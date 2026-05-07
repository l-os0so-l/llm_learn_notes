"""
PPO 训练主脚本（通用多轮交互版）
支持命令行运行 和 Jupyter Notebook 直接调用 main(config)。

训练流程（每步一个 episode，最多 max_rounds 轮）：
  for round in 1..max_rounds:
      模型根据当前历史生成 response
      if response 包含 <answer>:
          结束 episode，计算 final_reward
          break
      else:
          提取并执行代码
          把执行结果加入历史，继续下一轮
  if 达到 max_rounds 仍无 <answer>:
      判负，final_reward = -1.0

Reward 设计：
  中间轮: r_t = intermediate_reward(格式分 + 代码执行分)   ← 小r
  最终轮: r_T = final_reward(答案正确性) 或 -1.0(超时惩罚)  ← 大R / 惩罚

Return 反传（discounted）：
  G_T = r_T
  G_{T-1} = r_{T-1} + γ * G_T
  ...
  G_1 = r_1 + γ * G_2

PPO Update: 每轮分别做 ppo_step，用该轮的 G_t 作为 reward target。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 无GUI后端
import matplotlib.pyplot as plt

import argparse
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict
from peft import LoraConfig
from ppo_trainer import PPOVLMTrainer
from reward import compute_intermediate_reward, compute_final_reward, LLMRewardModel
from sandbox import extract_code, execute_code, wrap_sandbox_output, has_answer, set_temp_dir
import shutil
is_print_mem = False


def gpu_mem(tag=""):
    global is_print_mem
    if is_print_mem:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] 已分配: {allocated:.2f} GB | 预留: {reserved:.2f} GB")


def diagnose_oom(tag="", trainer=None, local_vars=None):
    """在 OOM 时输出尽可能详细的显存诊断信息。"""
    import gc
    import sys
    print("\n" + "=" * 70)
    print(f"  OOM EMERGENCY DIAGNOSTICS @ {tag}")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\n[Basic Stats]")
        print(f"  memory_allocated:      {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"  memory_reserved:       {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        print(f"  max_memory_allocated:  {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        print(f"  max_memory_reserved:   {torch.cuda.max_memory_reserved() / 1e9:.3f} GB")
        print(f"\n{torch.cuda.memory_summary()}")

        print("\n[Top CUDA tensors by memory]")
        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size = obj.numel() * obj.element_size()
                    tensors.append((size, obj.shape, obj.dtype, type(obj).__name__))
            except Exception:
                pass
        tensors.sort(reverse=True)
        total_top = 0
        for rank, (size, shape, dtype, name) in enumerate(tensors[:50], 1):
            total_top += size
            print(f"  #{rank:2d} {size/1e9:.3f} GB | {str(shape):30s} | {dtype} | {name}")
        print(f"  Top 50 tensors total: {total_top/1e9:.3f} GB")
        print(f"  All CUDA tensors:     {sum(s for s,_,_,_ in tensors)/1e9:.3f} GB")

        if trainer is not None:
            print("\n[Model memory estimates]")
            for name, model in [("policy", trainer.policy), ("critic", trainer.critic), ("ref", trainer.ref)]:
                try:
                    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                    buf_size = sum(b.numel() * b.element_size() for b in model.buffers())
                    print(f"  {name:10s} params={param_size/1e9:.3f} GB | buffers={buf_size/1e9:.3f} GB")
                except Exception as e:
                    print(f"  {name:10s} error: {e}")

    if local_vars is not None:
        print("\n[Large objects in local scope]")
        big_locals = []
        for k, v in local_vars.items():
            try:
                if torch.is_tensor(v) and v.is_cuda:
                    size = v.numel() * v.element_size()
                    big_locals.append((size, k, str(v.shape), v.dtype))
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv) and vv.is_cuda:
                            size = vv.numel() * vv.element_size()
                            big_locals.append((size, f"{k}[{kk}]", str(vv.shape), vv.dtype))
            except Exception:
                pass
        big_locals.sort(reverse=True)
        for size, name, shape, dtype in big_locals[:30]:
            print(f"  {size/1e9:.3f} GB | {name:30s} | {shape} | {dtype}")

    print("\n" + "=" * 70)
    print("  END OF OOM DIAGNOSTICS")
    print("=" * 70 + "\n")
    sys.stdout.flush()

# ==================== 配置类 ====================
@dataclass
class TrainConfig:
    """训练配置（支持命令行 / Notebook / 代码直接修改）。"""
    model_path: str = "Part2/model/Qwen2.5-VL-3B-Instruct"
    data_dir: str = "Part2/data/rl_data"
    output_dir: str = "Part2/PPO/checkpoints"
    lr: float = 1e-5
    critic_lr: float = 1e-5
    batch_size: int = 8
    max_steps: int = 101
    save_interval: int = 33
    max_new_tokens: int = 512
    max_rounds: int = 3          # 最大交互轮数
    timeout_penalty: float = -1.3  # 达到 max_rounds 仍无答案的惩罚
    # PPO 超参数
    ppo_epochs: int = 3
    eps_clip: float = 0.1
    beta_kl: float = 0.01
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.95
    lam: float = 0.95
    debug: bool = False
    rm_model_path: str = "Part2/model/Qwen3.5-4B"               
    output_res_dir: str = "Part2/PPO/res"


def load_rl_data(data_dir: str, debug: bool = False):
    """加载 RL 数据，混洗后返回列表。每条: {question, answer, image, type}"""
    data = []
    for split in ["2round_local", "comp_local", "single_local"]:
        path = os.path.join(data_dir, f"{split}.json")
        data_type = split.replace("_local", "")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img = item.get("image", None)
                if isinstance(img, list) and len(img) == 0:
                    img = None
                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "image": img,
                    "type": data_type,
                })
    random.shuffle(data)
    if debug:
        print(f"[Debug] 使用 {min(10, len(data))} 条数据快速验证")
        data = data[:10]
    print(f"加载数据完成: 共 {len(data)} 条")
    return data


def sample_batch(data: list, batch_size: int):
    """随机采样一个 batch"""
    batch = random.sample(data, min(batch_size, len(data)))
    return {
        "questions": [d["question"] for d in batch],
        "answers": [d["answer"] for d in batch],
        "images": [d["image"] for d in batch],
        "types": [d["type"] for d in batch],
    }


def build_initial_messages(batch: dict) -> list:
    """构建初始对话历史。返回 list，每个元素是一条样本的 messages。
    注意：原始数据的 question 字段已包含完整指令（含 <think>/<answer> 格式要求），
    无需额外添加 system prompt，避免冲突。"""
    messages_list = []
    for q, img in zip(batch["questions"], batch["images"]):
        content = []
        if img is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": q})
        messages_list.append([
            {"role": "user", "content": content},
        ])
    return messages_list


# ------------------------------------------------------------------------------
# Padding 辅助函数：把不同长度的 episode 数据拼成 batch
# ------------------------------------------------------------------------------
def _pad_2d(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    """
    对一组 2D tensor [1, seq_len] 做 right padding，返回 [batch, max_len]。
    """
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] < max_len:
            pad = torch.full((t.shape[0], max_len - t.shape[1]), pad_value,
                             dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def _build_round_batch(turns: List[Dict], pad_token_id: int) -> Dict:
    """
    把同轮次的多个 episode turn 拼成一个 batch。
    自动处理 sequences、inputs 的 padding。

    Args:
        turns: list of dict，每个 dict 包含 'seq', 'inp', 'plen', 'lp_old', 'lp_ref', 'val', 'G'
        pad_token_id: padding token id

    Returns:
        dict，可直接传给 ppo_step
    """
    # sequences
    sequences = _pad_2d([t['seq'] for t in turns], pad_token_id)

    # inputs: 主要是 input_ids, attention_mask, pixel_values 等
    inputs_list = [t['inp'] for t in turns]
    max_seq_len = max(inp['input_ids'].shape[1] for inp in inputs_list)

    merged_inputs = {}
    # input_ids + attention_mask 需要 padding
    ids_list, mask_list = [], []
    for inp in inputs_list:
        ids = inp['input_ids']
        mask = inp.get('attention_mask', torch.ones_like(ids))
        if ids.shape[1] < max_seq_len:
            pad = torch.full((ids.shape[0], max_seq_len - ids.shape[1]), pad_token_id,
                             dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, pad], dim=1)
            mask_pad = torch.zeros((mask.shape[0], max_seq_len - mask.shape[1]),
                                   dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, mask_pad], dim=1)
        ids_list.append(ids)
        mask_list.append(mask)
    merged_inputs['input_ids'] = torch.cat(ids_list, dim=0)
    merged_inputs['attention_mask'] = torch.cat(mask_list, dim=0)

    # pixel_values, image_grid_thw：按样本切片后存储，现在需要按 batch 维拼接
    for key in ['pixel_values', 'image_grid_thw']:
        tensors = [t['inp'][key] for t in turns if key in t['inp'] and t['inp'][key] is not None]
        if tensors:
            merged_inputs[key] = torch.cat(tensors, dim=0)

    # image_rotary_emb 通常不存在；若存在则假设同轮次来源相同，沿用第一个 turn
    for key in ['image_rotary_emb']:
        if key in inputs_list[0]:
            merged_inputs[key] = inputs_list[0][key]

    # prompt_len：同轮次的 prompt_len 理论上相同（因为 history 结构相同），取第一个
    prompt_len = turns[0]['plen']

    # logprobs / values / resp_mask：需要 padding 到相同 response_len
    max_resp_len = max(t['lp_old'].shape[1] for t in turns)
    lp_old_list, val_list = [], []
    lp_ref_list = []
    resp_mask_list = []
    for t in turns:
        lp_old, val = t['lp_old'], t['val']
        lp_ref = t.get('lp_ref')
        resp_mask = t['resp_mask']
        if lp_old.shape[1] < max_resp_len:
            pad_len = max_resp_len - lp_old.shape[1]
            pad = torch.full((lp_old.shape[0], pad_len), 0.0,
                             dtype=lp_old.dtype, device=lp_old.device)
            lp_old = torch.cat([lp_old, pad], dim=1)
            lp_ref = torch.cat([lp_ref, pad], dim=1)
            # value 的 response 部分也需要 padding
            val_pad = torch.full((val.shape[0], pad_len), 0.0,
                                 dtype=val.dtype, device=val.device)
            val = torch.cat([val, val_pad], dim=1)
            # response mask padding
            mask_pad = torch.zeros(resp_mask.shape[0], pad_len,
                                   dtype=resp_mask.dtype, device=resp_mask.device)
            resp_mask = torch.cat([resp_mask, mask_pad], dim=1)
        lp_old_list.append(lp_old)
        lp_ref_list.append(lp_ref)
        val_list.append(val)
        resp_mask_list.append(resp_mask)

    lp_old = torch.cat(lp_old_list, dim=0)
    val = torch.cat(val_list, dim=0)
    resp_mask = torch.cat(resp_mask_list, dim=0)

    # rewards
    rewards = torch.tensor([t['G'] for t in turns],
                           dtype=sequences.dtype, device=sequences.device)

    result = {
        'sequences': sequences,
        'original_inputs': merged_inputs,
        'prompt_len': prompt_len,
        'old_logprobs': lp_old,
        'values': val,
        'rewards': rewards,
        'response_mask': resp_mask,
    }
    result['ref_logprobs'] = torch.cat(lp_ref_list, dim=0)
    return result


# ------------------------------------------------------------------------------
# 主训练入口
# ------------------------------------------------------------------------------
def main(config: Optional[TrainConfig] = None):
    if config is None:
        config = TrainConfig()

    os.makedirs(config.output_dir, exist_ok=True)

    # ==================== LoRA 配置 ====================
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ==================== 加载数据 ====================
    dataset = load_rl_data(config.data_dir, debug=config.debug)
    # gpu_mem() ##################################################################
    # ==================== 初始化 Trainer ====================
    trainer = PPOVLMTrainer(
        model_path=config.model_path,
        policy_lora_config=lora_config,
        critic_lora_config=lora_config,
        lr=config.lr,
        critic_lr=config.critic_lr,
        ppo_epochs=config.ppo_epochs,
        eps_clip=config.eps_clip,
        beta_kl=config.beta_kl,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        gamma=config.gamma,
        lam=config.lam,
        max_new_tokens=config.max_new_tokens,
    )
    # gpu_mem("after_init_trainer")############################################################################
    pad_token_id = trainer.processor.tokenizer.pad_token_id
    # gpu_mem()   #################################################################################
    # ==================== 初始化 LLM 奖励模型 ====================

    llm_rm = LLMRewardModel(config.rm_model_path)
    # gpu_mem("after_init_rm")  ###########################################################################################

    # 调试日志文件
    debug_log_path = os.path.join(config.output_res_dir, "debug_ppo_episodes.log")
    os.makedirs(config.output_res_dir, exist_ok=True)
    def _debug_log(text: str):
        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("[DEBUG] " + text.replace("\n", "\n[DEBUG] "))


    metrics_history = {
        "steps": [],
        "total_loss": [],
        "r_final_mean": [],
        "timeout_rate": [],
        "avg_rounds": [],
    }

    # ==================== 日志系统初始化 ====================
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.output_res_dir, "logs", run_timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"train_{run_timestamp}.log")
    metrics_file_path = os.path.join(log_dir, f"metrics_history_{run_timestamp}.json")

    def _log_print(msg: str):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg)

    # ==================== 训练循环 ====================
    _log_print(f"\n[Run Timestamp] {run_timestamp}")
    _log_print("\n" + "=" * 50)
    _log_print("开始 PPO 多轮交互训练")
    _log_print(f"  max_rounds={config.max_rounds}, timeout_penalty={config.timeout_penalty}")
    _log_print("=" * 50)

    # 沙盒图片输出临时目录根路径
    sandbox_temp_root = os.path.join(config.output_res_dir, "sandbox_temp")
    os.makedirs(sandbox_temp_root, exist_ok=True)
    last_temp_dir = None

    for step in range(config.max_steps):
        # ---- 每 step 管理临时目录 ----
        # 清理上一批次的临时目录
        if last_temp_dir is not None and os.path.exists(last_temp_dir):
            try:
                shutil.rmtree(last_temp_dir)
            except Exception as e:
                _log_print(f"[WARN] 清理旧临时目录失败: {e}")
        # 创建本批次的临时目录
        step_temp_dir = os.path.join(sandbox_temp_root, f"step_{step:05d}")
        os.makedirs(step_temp_dir, exist_ok=True)
        set_temp_dir(step_temp_dir)
        last_temp_dir = step_temp_dir

        batch = sample_batch(dataset, config.batch_size)
        batch_size = len(batch["questions"])

        # ==========================================================
        # Phase 1: 运行 episodes（每个样本最多 max_rounds 轮）
        # ==========================================================
        episodes = [[] for _ in range(batch_size)]  # episodes[i] = list of turns
        messages_list = build_initial_messages(batch)
        active = list(range(batch_size))  # 还未得到 answer 的样本索引

        for round_idx in range(config.max_rounds):
            if not active:
                break

            # 只保留 active 样本的 messages 和 images
            active_messages = [messages_list[i] for i in active]
            active_images = [batch["images"][i] for i in active]
            gpu_mem(f"before_generate_step{step}_round{round_idx}")
            try:
                resp, lp_old, lp_ref, val, seq, plen, inp, resp_mask = trainer.generate_from_messages(
                    active_messages, active_images
                )
            except torch.OutOfMemoryError as e:
                diagnose_oom(
                    f"generate_from_messages_step{step}_round{round_idx}",
                    trainer=trainer,
                    local_vars={"active_messages": active_messages, "active_images": active_images},
                )
                raise
            gpu_mem(f"after_generate_step{step}_round{round_idx}")
            # print(f"[Len] seq_len={seq.shape[1]}, prompt_len={plen}, resp_len={seq.shape[1]-plen}")
            # gpu_mem()   ##################################################################################
            
            # 存储当前轮数据（暂存，等 episode 结束后再填 reward / G）
            # 使用 image_ranges 正确切片 pixel_values / image_grid_thw，避免 BS>1 时的数据污染
            batch_level_keys = {'pixel_values', 'image_grid_thw', 'image_rotary_emb'}
            image_ranges = inp.get('image_ranges', [None] * len(active))
            for idx, i in enumerate(active):
                turn_inp = {}
                for k, v in inp.items():
                    if k == 'image_ranges':
                        continue
                    elif k in batch_level_keys:
                        if k == 'pixel_values' and image_ranges[idx] is not None:
                            r = image_ranges[idx]['pixel_values']
                            turn_inp[k] = v[r[0]:r[1]]
                        elif k == 'image_grid_thw' and image_ranges[idx] is not None:
                            r = image_ranges[idx]['image_grid_thw']
                            turn_inp[k] = v[r[0]:r[1]]
                        else:
                            # fallback：image_rotary_emb 或不具备 image_ranges 的情况
                            turn_inp[k] = v
                    else:
                        turn_inp[k] = v[idx:idx+1]

                turn = {
                    'resp': resp[idx],
                    'seq': seq[idx:idx+1],
                    'lp_old': lp_old[idx:idx+1],
                    'val': val[idx:idx+1],
                    'plen': plen,
                    'inp': turn_inp,
                    'round': round_idx + 1,
                    'resp_mask': resp_mask[idx:idx+1],
                }
                turn['lp_ref'] = lp_ref[idx:idx+1]
                episodes[i].append(turn)
                

            # 检查哪些样本已有 answer
            still_active = []
            for idx, i in enumerate(active):
                if has_answer(resp[idx]):
                    # 已有答案，该轮就是最终轮，reward 后面统一算
                    pass
                else:
                    # 没答案，执行代码，准备下一轮
                    code = extract_code(resp[idx])
                    if code:
                        # 每个样本分配独立临时子目录，防止 batch 内文件覆盖
                        sample_temp_dir = os.path.join(step_temp_dir, f"sample_{i}")
                        os.makedirs(sample_temp_dir, exist_ok=True)
                        set_temp_dir(sample_temp_dir)
                        
                        result = execute_code(code, timeout=30)
                        sandbox_result = wrap_sandbox_output(result)
                        r_int = compute_intermediate_reward(resp[idx], result)
                    else:
                        sandbox_result = {
                            "text": (
                                "<sandbox_output>\n"
                                "(No code provided. Please continue reasoning and provide your final answer.)\n"
                                "</sandbox_output>"
                            ),
                            "images": [],
                        }
                        r_int = compute_intermediate_reward(resp[idx], None)

                    #  intermediate reward 先存上（中间轮确定已知）
                    episodes[i][-1]['r'] = r_int

                    # 更新 messages 加入 assistant 和 user turn
                    messages_list[i] = messages_list[i].copy()
                    messages_list[i].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": resp[idx]}]
                    })
                    # sandbox 输出支持图片：若有 base64 图片，解码后嵌入 content
                    user_content = [{"type": "text", "text": sandbox_result["text"]}]
                    for img in sandbox_result.get("images", []):
                        user_content.insert(0, {"type": "image", "image": img})
                    messages_list[i].append({
                        "role": "user",
                        "content": user_content
                    })
                    # 截断历史，只保留初始 user message + 最近 N 轮交互
                    MAX_HISTORY_ROUNDS = 1
                    # 没有 system prompt，初始上下文只有 1 条 (initial user)
                    initial_len = 1
                    max_total_len = initial_len + MAX_HISTORY_ROUNDS * 2
                    if len(messages_list[i]) > max_total_len:
                        messages_list[i] = messages_list[i][:initial_len] + messages_list[i][-(MAX_HISTORY_ROUNDS * 2):]
                    still_active.append(i)

            active = still_active

        # ==========================================================
        # Phase 2: 计算每轮的 reward 和 return（从后向前反传）
        # ==========================================================
        for i in range(batch_size):
            ep = episodes[i]
            last_turn = ep[-1]
            resp = last_turn['resp']
            ans_flag = has_answer(resp)

            # _debug_log(f"\n========== Step {step} | Sample {i} ==========")
            # _debug_log(f"Q: {batch['questions'][i]}")
            # _debug_log(f"GT: {batch['answers'][i]} | type: {batch['types'][i]}")
            # _debug_log(f"Total Rounds: {len(ep)}")

            # # 打印每一轮的输出（中间轮 + 最终轮）
            # for turn_idx, turn in enumerate(ep):
            #     is_final = (turn_idx == len(ep) - 1)
            #     role = "FINAL" if is_final else f"Round {turn_idx + 1}"
            #     turn_resp = turn['resp']
            #     _debug_log(f"\n>>> [{role}] ---")
            #     if len(turn_resp) > 800:
            #         _debug_log(f"RESP [head]:\n{turn_resp[:400]}")
            #         _debug_log(f"RESP [tail]:\n{turn_resp[-400:]}")
            #     else:
            #         _debug_log(f"RESP:\n{turn_resp}")
            #     # 中间轮的 intermediate reward（如果有）
            #     if not is_final and 'r' in turn:
            #         _debug_log(f"Intermediate Reward: {turn['r']:.3f}")
    
            # # 最终轮的 reward
            # _debug_log(f"\n>>> [SUMMARY] has_answer={ans_flag}")
            if ans_flag:
                # gpu_mem(f"before_rm_score_step{step}") ################################################
                from reward import extract_answer
                pred_ans = extract_answer(resp)
                # _debug_log(f"EXTRACTED_ANSWER: {pred_ans}")
                last_turn['r'] = compute_final_reward(
                    last_turn['resp'],
                    batch["answers"][i],
                    batch["types"][i],
                    question=batch["questions"][i],
                    llm_rm=llm_rm,
                )
                # _debug_log(f"FINAL REWARD: {last_turn['r']:.3f}")
                # gpu_mem(f"after_rm_score_step{step}") ####################################################
            else:
                # 达到 max_rounds 仍无答案，惩罚
                last_turn['r'] = config.timeout_penalty
                # _debug_log(f"FINAL REWARD: {last_turn['r']:.3f}  (TIMEOUT / NO <answer>)")

            # Return 反传: G_t = r_t + gamma * G_{t+1}
            for t in reversed(range(len(ep))):
                if t == len(ep) - 1:
                    ep[t]['G'] = ep[t]['r']
                else:
                    ep[t]['G'] = ep[t]['r'] + config.gamma * ep[t+1]['G']

        # ==========================================================
        # Phase 3: PPO Update（按轮次分组，同轮次拼 batch）
        # ==========================================================
        max_len = max(len(ep) for ep in episodes)
        all_stats = []

        for epoch in range(config.ppo_epochs):
            for round_idx in range(max_len):
                # gpu_mem()   ##################################################################
                torch.cuda.empty_cache()
                round_turns = [ep[round_idx] for ep in episodes if round_idx < len(ep)]
                if not round_turns:
                    continue
                gpu_mem(f"before_ppo_step{step}_epoch{epoch}_round{round_idx}")
                # 拼 batch（自动 padding）
                batch_data = _build_round_batch(round_turns, pad_token_id)
                try:
                    stats = trainer.ppo_step(
                        sequences=batch_data['sequences'],
                        original_inputs=batch_data['original_inputs'],
                        prompt_len=batch_data['prompt_len'],
                        old_logprobs=batch_data['old_logprobs'],
                        values=batch_data['values'],
                        rewards=batch_data['rewards'],
                        response_mask=batch_data.get('response_mask'),
                        ref_logprobs=batch_data['ref_logprobs'],
                    )
                except torch.OutOfMemoryError as e:
                    diagnose_oom(
                        f"ppo_step_step{step}_epoch{epoch}_round{round_idx}",
                        trainer=trainer,
                        local_vars=batch_data,
                    )
                    raise
                all_stats.append(stats)
                # gpu_mem(f"after_ppo_step{step}_epoch{epoch}_round{round_idx}") ##################################################

        # 平均所有轮的 stats
        avg_stats = {
            k: sum(s[k] for s in all_stats) / len(all_stats)
            for k in all_stats[0]
        } if all_stats else {}

        # ==========================================================
        # 日志
        # ==========================================================
        if step % 3 == 0 and avg_stats:
            # gpu_mem()   ##########################################################################
            # 统计信息
            num_rounds_list = [len(ep) for ep in episodes]
            avg_rounds = sum(num_rounds_list) / len(num_rounds_list)
            timeout_rate = sum(1 for ep in episodes if not has_answer(ep[-1]['resp'])) / len(episodes)

            G0_list = [ep[0]['G'] for ep in episodes]
            r_int_list = [ep[t]['r'] for ep in episodes for t in range(len(ep) - 1) if 'r' in ep[t]]
            r_final_list = [ep[-1]['r'] for ep in episodes]

            metrics_history["steps"].append(step)
            metrics_history["total_loss"].append(avg_stats.get("total_loss", 0))
            metrics_history["r_final_mean"].append(sum(r_final_list) / len(r_final_list))
            metrics_history["timeout_rate"].append(timeout_rate)
            metrics_history["avg_rounds"].append(avg_rounds)
            
            _log_print(
                f"Step {step:04d} | "
                f"loss={avg_stats.get('total_loss', 0):.4f} | "
                f"policy={avg_stats.get('policy_loss', 0):.4f} | "
                f"value={avg_stats.get('value_loss', 0):.4f} | "
                f"entropy={avg_stats.get('entropy', 0):.4f} | "
                f"kl={avg_stats.get('kl', 0):.4f} | "
                f"avg_rounds={avg_rounds:.1f} | "
                f"timeout={timeout_rate:.1%} | "
                f"G1_mean={sum(G0_list)/len(G0_list):.3f} | "
                f"r_int_mean={sum(r_int_list)/max(len(r_int_list),1):.3f} | "
                f"r_final_mean={sum(r_final_list)/len(r_final_list):.3f}"
            )

            # 打印一个 sample（取轮数最多的或第一个）
            longest_ep = max(episodes, key=lambda ep: len(ep))
            final_resp = longest_ep[-1]['resp']
            _log_print(f"  Sample (R{len(longest_ep)}): {final_resp[:120]}...")

            # 保存当前已收集的指标数据（覆盖写入，确保始终是最新完整数据）
            with open(metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, ensure_ascii=False, indent=2)

        # ==========================================================
        # 保存 checkpoint
        # ==========================================================
        if step % config.save_interval == 0:
            save_path = os.path.join(config.output_dir, f"step_{step + 1}")
            os.makedirs(save_path, exist_ok=True)
            trainer.policy.save_pretrained(os.path.join(save_path, "policy"))
            trainer.critic.save_pretrained(os.path.join(save_path, "critic"))
            _log_print(f"[Saved] Checkpoint -> {save_path}")

    # 保存最终模型
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    trainer.policy.save_pretrained(os.path.join(final_path, "policy"))
    trainer.critic.save_pretrained(os.path.join(final_path, "critic"))
    _log_print(f"\n训练完成！最终模型保存到: {final_path}")

    # ==================== 绘制并保存指标图 ====================
    if metrics_history["steps"]:
        os.makedirs(config.output_res_dir, exist_ok=True)
        plot_save_path = os.path.join(config.output_res_dir, "training_metrics.png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("PPO Training Metrics", fontsize=14, fontweight="bold")

        # 1) Total Loss
        ax = axes[0, 0]
        ax.plot(metrics_history["steps"], metrics_history["total_loss"], "b-o", markersize=4, linewidth=1.2)
        ax.set_title("Total Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # 2) Final Reward Mean
        ax = axes[0, 1]
        ax.plot(metrics_history["steps"], metrics_history["r_final_mean"], "g-o", markersize=4, linewidth=1.2)
        ax.set_title("Final Reward Mean")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

        # 3) Timeout Rate
        ax = axes[1, 0]
        ax.plot(metrics_history["steps"], metrics_history["timeout_rate"], "r-o", markersize=4, linewidth=1.2)
        ax.set_title("Timeout Rate")
        ax.set_xlabel("Step")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # 4) Average Rounds
        ax = axes[1, 1]
        ax.plot(metrics_history["steps"], metrics_history["avg_rounds"], "m-o", markersize=4, linewidth=1.2)
        ax.set_title("Average Rounds")
        ax.set_xlabel("Step")
        ax.set_ylabel("Rounds")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
        plt.close()
        _log_print(f"[Plot] 训练指标图已保存到: {plot_save_path}")
    else:
        _log_print("[Plot] 没有收集到足够的数据用于绘图")



# ==================== 命令行入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=TrainConfig.model_path)
    parser.add_argument("--data_dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr, help="Policy 学习率")
    parser.add_argument("--critic_lr", type=float, default=TrainConfig.critic_lr, help="Critic 学习率")
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--save_interval", type=int, default=TrainConfig.save_interval)
    parser.add_argument("--max_new_tokens", type=int, default=TrainConfig.max_new_tokens)
    parser.add_argument("--max_rounds", type=int, default=TrainConfig.max_rounds, help="最大交互轮数")
    parser.add_argument("--timeout_penalty", type=float, default=TrainConfig.timeout_penalty, help="超时无答案的惩罚")
    parser.add_argument("--ppo_epochs", type=int, default=TrainConfig.ppo_epochs)
    parser.add_argument("--eps_clip", type=float, default=TrainConfig.eps_clip)
    parser.add_argument("--beta_kl", type=float, default=TrainConfig.beta_kl)
    parser.add_argument("--value_coef", type=float, default=TrainConfig.value_coef)
    parser.add_argument("--entropy_coef", type=float, default=TrainConfig.entropy_coef)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--lam", type=float, default=TrainConfig.lam)
    parser.add_argument("--debug", action="store_true", default=TrainConfig.debug)
    parser.add_argument("--rm_model_path", type=str, default=TrainConfig.rm_model_path, help="LLM 奖励模型路径")
    cli_args = parser.parse_args()

    config = TrainConfig(**vars(cli_args))
    main(config)
