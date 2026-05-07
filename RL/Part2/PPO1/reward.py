"""
Reward 计算模块：格式奖励 + 执行奖励 + LLM 评估奖励

支持拆分为：
  - compute_intermediate_reward: 中间步骤奖励（格式 + 代码执行）
  - compute_final_reward: 最终结果奖励（LLM 评估答案正确性）
  - compute_reward: 综合奖励（= intermediate + final）
"""
import re
import torch
from typing import Optional, Dict, List
from sandbox import extract_code, execute_code


# ------------------------------------------------------------------------------
# 规则基础奖励（保留作为 fallback / 中间轮奖励）
# ------------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    """
    从模型 response 中提取最终答案。
    优先匹配完整 <answer>...</answer>，其次匹配未闭合 <answer>...，再次匹配 \boxed{...}
    """
    # 1. 匹配完整 <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        ans = match.group(1).strip()
        boxed = re.search(r'\\boxed\{(.*?)\}', ans)
        if boxed:
            return boxed.group(1).strip()
        return ans

    # 2. 匹配未闭合的 <answer>...
    match = re.search(r'<answer>(.*)', response, re.DOTALL)
    if match:
        ans = match.group(1).strip()
        boxed = re.search(r'\\boxed\{(.*?)\}', ans)
        if boxed:
            return boxed.group(1).strip()
        return ans

    # 3. 匹配 \boxed{}
    match = re.search(r'\\boxed\{(.*?)\}', response)
    if match:
        return match.group(1).strip()

    # 4. 兜底：取最后 20 个非空字符
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def normalize_number(s: str) -> float | None:
    """尝试把字符串解析为数字，失败返回 None"""
    try:
        # 去掉逗号、空格
        s = s.replace(',', '').replace(' ', '')
        return float(s)
    except (ValueError, TypeError):
        return None


def match_answer(pred: str, gt: str, data_type: str) -> bool:
    """
    判断预测答案是否匹配 Ground Truth。
    data_type: 'comp' | '2round' | 'single'
    """
    # 统一去掉 \boxed{} 包装
    boxed_pred = re.search(r'\\boxed\{(.*?)\}', pred)
    if boxed_pred:
        pred = boxed_pred.group(1).strip()
    boxed_gt = re.search(r'\\boxed\{(.*?)\}', gt)
    if boxed_gt:
        gt = boxed_gt.group(1).strip()

    pred = pred.strip()
    gt = gt.strip()

    if data_type == 'comp':
        # 数学题：数值比较（允许 1% 相对误差或绝对误差 1e-3）
        pred_num = normalize_number(pred)
        gt_num = normalize_number(gt)
        if pred_num is not None and gt_num is not None:
            rel_err = abs(pred_num - gt_num) / (abs(gt_num) + 1e-8)
            abs_err = abs(pred_num - gt_num)
            return rel_err < 0.01 or abs_err < 1e-3
        # 字符串精确匹配兜底
        return pred.lower() == gt.lower()

    elif data_type == '2round':
        # 视觉问答：关键词包含匹配
        pred_lower = pred.lower()
        gt_lower = gt.lower()
        # 如果 GT 包含在预测中，或预测包含在 GT 中
        return gt_lower in pred_lower or pred_lower in gt_lower

    elif data_type == 'single':
        # 选择题：提取字母后精确匹配
        pred_letter = re.search(r'\b([A-Da-d])\b', pred)
        gt_letter = re.search(r'\b([A-Da-d])\b', gt)
        if pred_letter and gt_letter:
            return pred_letter.group(1).upper() == gt_letter.group(1).upper()
        return pred.lower() == gt.lower()

    return pred.lower() == gt.lower()


# ------------------------------------------------------------------------------
# LLM 奖励模型（基于 Qwen3.5-4B）
# ------------------------------------------------------------------------------

class LLMRewardModel:
    """
    使用 Qwen3.5-4B 作为奖励模型，对模型回答与 Ground Truth 进行一致性评分。
    
    评分方式：
      - 构建评估 prompt，让 LLM 输出 0-10 的整数分数
      - 10 分 = 完全正确，0 分 = 完全错误
      - 最终 reward 映射到 [-1.0, 1.0] 范围
    """

    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 默认使用 CPU 以节省显存；若传入 cuda，则初始化在 CPU，score 时临时搬到 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_on_gpu = False if self.device=="cpu" else True  # 标记模型当前是否在 GPU 上
        print(f"[LLM-RM] Loading reward model from: {model_path}")

        # RM 只做纯文本评估，直接用 tokenizer 更可靠（可控制 padding_side）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        
        # 兼容 transformers 4.x 和 5.x
        # 为节省显存，默认加载到 CPU；需要 GPU 时在 score() 中临时搬运
        try:
            # transformers >= 5.0
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, trust_remote_code=True
            )
        except TypeError:
            # transformers < 5.0
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        self.model = self.model.to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"[LLM-RM] Loaded successfully on CPU. Device: {self.device}")

    def _build_prompt(self, question: str, pred_answer: str, gt_answer: str) -> str:
        """为单条样本构建评估 prompt。"""
        system_msg = "Rate answer similarity. Brief."

        user_msg = (
            f"A: {pred_answer}\n"
            f"B: {gt_answer}\n"
            f"How similar? 0-10.\n"
            f"0=completely different, 10=exactly the same.\n"
            f"FINAL_SCORE:"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # 兼容不同版本的 apply_chat_template
        apply_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        text = self.tokenizer.apply_chat_template(
        messages,
        **apply_kwargs,
        enable_thinking=False,
        chat_template_kwargs={"enable_thinking": False}
    )

        return text

    def _move_model_to_device(self, target_device: torch.device):
        """将模型搬运到目标设备，并维护 _model_on_gpu 状态。"""
        if target_device.type == "cuda":
            if not self._model_on_gpu:
                self.model = self.model.to(target_device)
                self._model_on_gpu = True
        else:
            if self._model_on_gpu:
                self.model = self.model.to("cpu")
                self._model_on_gpu = False
                torch.cuda.empty_cache()

    @torch.no_grad()
    def score(
        self,
        questions: List[str],
        predictions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """
        对一批样本进行评分。
        若初始化时 device=cuda，则 score 前临时把模型搬上 GPU，结束后搬回 CPU；
        若 device=cpu，则全程在 CPU 上运行，不占用显存。

        Args:
            questions: 原始问题列表
            predictions: 模型提取出的最终答案列表
            ground_truths: 标准答案列表

        Returns:
            每个样本的 reward 标量，范围 [-1.0, 1.0]
        """
        texts = [
            self._build_prompt(q, p, g)
            for q, p, g in zip(questions, predictions, ground_truths)
        ]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        # 根据配置决定运行设备
        try:
            if self.device.type == "cuda":
                self._move_model_to_device(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # CPU 模式：全程在 CPU 上，不触碰 GPU
                inputs = {k: v for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            # 若临时上了 GPU，用完搬回 CPU 释放显存
            if self.device.type == "cuda":
                self._move_model_to_device(torch.device("cpu"))

        # GPU常驻
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=10,
        #         do_sample=False,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #     )

        # 只解码新生成的 token
        gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        rewards = []
        for i, text in enumerate(response_texts):
            score = self._parse_score(text)
            reward = (score / 5.0) - 1.0
            rewards.append(reward)
            # 打印 RM 原始输出，方便调试
            # print(f"[LLM-RM] Q{i}: raw_response='{text.strip()}' | parsed_score={score} | reward={reward:.3f}")

        return rewards

    def _parse_score(self, text: str) -> float:
        """从模型输出中解析 0-10 的整数分数。"""
        original = text.strip()
        # 有些模型会输出 <think>...</think> 后再给分数，先去掉 think 块
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not text:
            text = original

        # 优先匹配开头的数字
        match = re.search(r'^\s*(\d+)', text)
        if match:
            score = float(match.group(1))
            # print(f"[LLM-RM-PARSE] raw='{original}' | cleaned='{text}' | matched_start={score}")
            return max(0.0, min(10.0, score))

        # 匹配 "Score: 10"、"score is 10" 等格式
        match = re.search(r'[Ss]core\s*[:=]?\s*(\d+)', text)
        if match:
            score = float(match.group(1))
            # print(f"[LLM-RM-PARSE] raw='{original}' | cleaned='{text}' | matched_score_keyword={score}")
            return max(0.0, min(10.0, score))

        # 兜底：搜索任意数字
        match = re.search(r'(\d+)', text)
        if match:
            score = float(match.group(1))
            # print(f"[LLM-RM-PARSE] raw='{original}' | cleaned='{text}' | matched_anywhere={score}")
            return max(0.0, min(10.0, score))

        # print(f"[LLM-RM-PARSE] raw='{original}' | cleaned='{text}' | NO_DIGIT_FOUND -> fallback 0")
        return 0.0


# ------------------------------------------------------------------------------
# 拆分的 Reward 函数
# ------------------------------------------------------------------------------

def compute_intermediate_reward(response: str, code_result: Optional[dict] = None) -> float:
    """
    中间步骤奖励：格式 + 代码执行。

    Args:
        response: 模型当前轮的输出
        code_result: execute_code() 的返回结果 dict，若为 None 表示没有找到代码

    Returns:
        中间奖励标量
    """
    reward = 0.0

    # (1) 格式奖励
    if '<code>' in response:
        reward += 0.05
    if '<think>' in response:
        reward += 0.02
    # 惩罚模型错误输出 <sandbox_output>（这是系统反馈专用标签，模型不应输出）
    if '<sandbox_output>' in response:
        reward -= 0.2

    # (2) 执行奖励
    if code_result is not None:
        if code_result['success']:
            # 代码没报错，但还要检查是否输出了有效图片路径
            has_image = bool(code_result.get('extracted_images'))
            if has_image:
                reward += 0.15  # 成功执行 + 输出了有效图片
            else:
                reward -= 0.10  # 成功执行但没输出有效图片（路径缺失或错误），视为失败
        else:
            reward -= 0.10  # 代码报错

    return reward


def compute_final_reward(
    response: str,
    ground_truth: str,
    data_type: str,
    question: Optional[str] = None,
    llm_rm: Optional[LLMRewardModel] = None,
) -> float:
    """
    最终结果奖励：使用 LLM 奖励模型评估答案正确性。

    Args:
        response: 模型最终轮的输出（应包含 <answer>）
        ground_truth: 标准答案
        data_type: 数据类型（保留用于接口兼容，本函数不再使用规则匹配）
        question: 原始问题文本（用于 LLM 评估）
        llm_rm: LLM 奖励模型实例

    Returns:
        最终奖励标量

    Raises:
        ValueError: 如果 llm_rm 为 None 或未提供 question。
    """
    if llm_rm is None:
        raise ValueError(
            "llm_rm (LLM Reward Model) is required but not provided. "
            "Please set use_llm_rm=True and provide a valid rm_model_path."
        )
    if question is None:
        raise ValueError(
            "question is required for LLM-based evaluation. "
            "Please ensure the dataset provides question text."
        )

    pred = extract_answer(response)
    rewards = llm_rm.score([question], [pred], [ground_truth])
    return rewards[0]


def compute_reward(
    response: str,
    ground_truth: str,
    data_type: str,
    question: Optional[str] = None,
    llm_rm: Optional[LLMRewardModel] = None,
) -> float:
    """
    综合 sequence-level reward（= intermediate + final）。
    兼容旧接口，单轮场景直接调用。
    """
    # 中间奖励
    code = extract_code(response)
    if code:
        result = execute_code(code, timeout=30)
        r_int = compute_intermediate_reward(response, result)
    else:
        r_int = compute_intermediate_reward(response, None)

    # 最终奖励
    r_final = compute_final_reward(response, ground_truth, data_type, question, llm_rm)

    return r_int + r_final


if __name__ == "__main__":
    # 自测
    print("=== Reward 模块自测 ===")

    # comp 测试
    r1 = "<think>...</think><answer>\\boxed{45}</answer>"
    gt45 = r"\boxed{45}"
    print(f"comp: pred=\boxed{{45}}, gt=\boxed{{45}}, reward={compute_reward(r1, gt45, 'comp'):.2f}")

    r2 = "<think>...</think><answer>45.0</answer>"
    print(f"comp: pred=45.0, gt=\boxed{{45}}, reward={compute_reward(r2, gt45, 'comp'):.2f}")

    # 2round 测试
    r3 = "<answer>The tray is on the left of the image.</answer>"
    print(f"2round: pred=left..., gt=left..., reward={compute_reward(r3, 'The tray is on the left of the image.', '2round'):.2f}")

    # single 测试
    r4 = "<answer>B</answer>"
    print(f"single: pred=B, gt=B, reward={compute_reward(r4, 'B', 'single'):.2f}")

    # 错误测试
    r5 = "<answer>C</answer>"
    print(f"single: pred=C, gt=B, reward={compute_reward(r5, 'B', 'single'):.2f}")
