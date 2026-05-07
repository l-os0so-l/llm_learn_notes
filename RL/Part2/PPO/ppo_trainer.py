"""
PPO 训练器（VLM 完整版）
- Policy Model: Qwen VL + LoRA（可训练）
- Reference Model: Qwen VL（冻结，用于 KL 参考分布）
- Critic Model: Qwen VL + LoRA + ValueHead（可训练，支持 GAE）

完整 PPO 流程：
  1. Token-level KL penalty 融入 reward
  2. GAE(γ, λ) 计算 advantage / return
  3. Per-token PPO-clip + Value Clipping + Entropy Bonus
  4. Policy / Critic 独立优化器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from typing import List, Optional, Tuple, Dict
import os
from PIL import Image

class ValueHead(nn.Module):
    """线性分数头，将 hidden state 映射为标量 value。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.dense.weight, std=0.01)
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [..., hidden_size]
        return self.dense(hidden_states).squeeze(-1)


class VLMCritic(nn.Module):
    """
    基于 VLM 的 Critic，输出每个 token 的 state value。
    结构上 = Base VLM (带 LoRA) + ValueHead。
    """

    def __init__(self, base_model: Qwen2_5_VLForConditionalGeneration):
        super().__init__()
        self.base_model = base_model
        # 兼容 transformers 5.x：Qwen2.5-VL 的 hidden_size 可能在 text_config 中
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(base_model.config, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot find hidden_size in model config")
        # value_head 默认在 CPU + float32，需要显式对齐 base_model 的设备和 dtype
        param = next(base_model.parameters())
        self.value_head = ValueHead(hidden_size).to(device=param.device, dtype=param.dtype)

    def forward(self, **kwargs) -> torch.Tensor:
        # 返回 per-token values: [batch, seq_len]
        outputs = self.base_model(**kwargs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]          # [batch, seq_len, hidden_size]
        values = self.value_head(last_hidden)            # [batch, seq_len]
        return values

    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory)
        torch.save(self.value_head.state_dict(), os.path.join(save_directory, "value_head.pt"))

    def load_pretrained(self, load_directory: str):
        value_head_path = os.path.join(load_directory, "value_head.pt")
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(
                torch.load(value_head_path, map_location="cpu")
            )


class PPOVLMTrainer:
    def __init__(
        self,
        model_path: str,
        policy_lora_config: LoraConfig,
        critic_lora_config: Optional[LoraConfig] = None,
        lr: float = 1e-5,
        critic_lr: Optional[float] = None,
        ppo_epochs: int = 4,
        eps_clip: float = 0.2,
        beta_kl: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.ppo_epochs = ppo_epochs
        self.eps_clip = eps_clip
        self.beta_kl = beta_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if critic_lora_config is None:
            critic_lora_config = policy_lora_config
        if critic_lr is None:
            critic_lr = lr

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # ==================== Policy Model ====================
        print("Loading Policy Model...")
        base_policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        # 关键：先开 GC，再套 LoRA
        # base_policy.gradient_checkpointing_enable()
        # base_policy.enable_input_require_grads()
        self.policy = get_peft_model(base_policy, policy_lora_config)
        # 强制把 LoRA 参数对齐到 base model 的 dtype（防止 fp32 浪费）
        for p in self.policy.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype)
        self.policy.print_trainable_parameters()


        # ==================== Reference Model ====================
        print("Loading Reference Model...")
        # 为节省显存，reference 初始化在 CPU，仅在计算 KL 时临时搬到 GPU
        self.ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        # ==================== Critic Model ====================
        print("Loading Critic Model...")
        critic_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        
        # critic_base.gradient_checkpointing_enable()
        # critic_base.enable_input_require_grads()
        critic_base = get_peft_model(critic_base, critic_lora_config)
        for p in critic_base.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype)
        self.critic = VLMCritic(critic_base)
        critic_base.print_trainable_parameters()

        # ==================== Processor & Optimizers ====================
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 256 * 256

        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]

        self.policy_optimizer = torch.optim.AdamW(policy_params, lr=lr)
        self.critic_optimizer = torch.optim.AdamW(critic_params, lr=critic_lr)

    # ------------------------------------------------------------------
    # 输入构建
    # ------------------------------------------------------------------
    def _build_inputs(self, prompts: List[str], images: Optional[List] = None):
        """
        用 processor 构建模型输入。
        返回 dict 包含 input_ids, attention_mask, pixel_values 等。
        """
        has_images = images is not None and any(img is not None for img in images)

        if has_images:
            messages = []
            for p, img in zip(prompts, images):
                content = []
                if img is not None:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": p})
                messages.append([{"role": "user", "content": content}])
        else:
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": p}]}]
                for p in prompts
            ]

        text = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        if has_images:
            pil_images = []
            for img in images:
                if isinstance(img, str):
                    
                    pil_images.append(
                        Image.open(img.lstrip("./").replace("/", os.sep)).convert("RGB")
                    )
                elif img is not None:
                    pil_images.append(img)
            inputs = self.processor(text=text, images=pil_images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

        return {k: v.to(self.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Policy 输出：per-token logprobs + entropy
    # ------------------------------------------------------------------
    def _compute_policy_outputs(
        self,
        model,
        sequences: torch.Tensor,
        original_inputs: Dict,
        prompt_len: int,
        no_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对完整序列重新前向传播，计算 response 部分的 per-token logprobs 和 entropy。

        返回:
            log_probs: [batch, response_len]
            entropy:   [batch, response_len]
        """
        inputs = {
            "input_ids": sequences,
            "attention_mask": original_inputs.get(
                "attention_mask", torch.ones_like(sequences)
            ),
        }
        for key in ["pixel_values", "image_grid_thw", "image_rotary_emb"]:
            if key in original_inputs:
                inputs[key] = original_inputs[key]

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            outputs = model(**inputs)

        logits = outputs.logits                                 # [batch, seq_len, vocab_size]
        response_len = sequences.shape[1] - prompt_len
        start = prompt_len - 1
        end = prompt_len + response_len - 1

        # response 部分对应的 logits（预测 response tokens）
        response_logits = logits[:, start:end, :]               # [batch, response_len, vocab_size]
        response_target_ids = sequences[:, prompt_len:prompt_len + response_len]  # [batch, response_len]

        log_probs_dist = F.log_softmax(response_logits, dim=-1)                       # [batch, response_len, vocab_size]
        log_probs = log_probs_dist.gather(dim=-1, index=response_target_ids.unsqueeze(-1)).squeeze(-1)

        probs = torch.exp(log_probs_dist)
        entropy = -(probs * log_probs_dist).sum(dim=-1)         # [batch, response_len]

        return log_probs, entropy

    # ------------------------------------------------------------------
    # Critic 输出：per-token values
    # ------------------------------------------------------------------
    def _compute_values(
        self,
        sequences: torch.Tensor,
        original_inputs: Dict,
        prompt_len: int,
        no_grad: bool = True,
    ) -> torch.Tensor:
        """
        计算每个 token 的 value。

        返回:
            values: [batch, seq_len]
        """
        inputs = {
            "input_ids": sequences,
            "attention_mask": original_inputs.get(
                "attention_mask", torch.ones_like(sequences)
            ),
        }
        for key in ["pixel_values", "image_grid_thw", "image_rotary_emb"]:
            if key in original_inputs:
                inputs[key] = original_inputs[key]

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            values = self.critic(**inputs)

        return values

    # ------------------------------------------------------------------
    # GAE 计算
    # ------------------------------------------------------------------
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        values: torch.Tensor,
        prompt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 GAE(γ, λ) 计算 advantages 和 returns。

        Args:
            rewards:      [batch]          外部 sequence-level reward
            old_logprobs: [batch, response_len]  policy 的 per-token logprob
            ref_logprobs: [batch, response_len]  reference 的 per-token logprob
            values:       [batch, seq_len]       critic 的 per-token value（完整序列）
            prompt_len:   int

        Returns:
            advantages: [batch, response_len]（已归一化）
            returns:    [batch, response_len]
        """
        batch_size, response_len = old_logprobs.shape
        device = rewards.device

        # 提取 response 部分的 value
        response_values = values[:, prompt_len:prompt_len + response_len]  # [batch, response_len]

        # Token-level KL penalty
        kl_penalty = old_logprobs - ref_logprobs                      # [batch, response_len]
        token_rewards = -self.beta_kl * kl_penalty
        token_rewards[:, -1] += rewards                                   # 外部 reward 仅加在末尾

        # GAE（从后向前）
        advantages = torch.zeros_like(token_rewards)
        returns = torch.zeros_like(token_rewards)
        gae = 0

        for t in reversed(range(response_len)):
            if t == response_len - 1:
                next_value = torch.zeros(batch_size, device=device, dtype=token_rewards.dtype)
            else:
                next_value = response_values[:, t + 1]
            # td err
            delta = token_rewards[:, t] + self.gamma * next_value - response_values[:, t]
            gae = delta + self.gamma * self.lam * gae
            advantages[:, t] = gae
            returns[:, t] = gae + response_values[:, t]

        # Advantage 归一化（batch-level，增加稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ------------------------------------------------------------------
    # 从对话历史构建输入（支持多轮交互）
    # ------------------------------------------------------------------
    def _build_inputs_from_messages(
        self,
        messages_list: List[List[dict]],
        images_list: Optional[List] = None,
    ) -> Dict:
        """
        从已有对话历史构建模型输入。
        messages_list: 每个元素是一条样本的完整对话历史（list of message dicts）。
        支持从 messages 的 content 中自动提取内嵌图片 {"type": "image", "image": ...}。

        返回的 dict 中额外包含 'image_ranges': List[Optional[Dict]]，
        用于标识每个样本在 pixel_values / image_grid_thw 中的索引范围，
        解决 BS>1 时 PPO update 的数据污染问题。
        """
        from PIL import Image

        # 从 messages 中提取内嵌图片，并清理 content 中的 "image" 键
        extracted_images = []  # List[List[Image]]
        processed_messages = []

        for messages in messages_list:
            sample_imgs = []
            new_messages = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for item in msg["content"]:
                        if item.get("type") == "image" and "image" in item:
                            img = item["image"]
                            if isinstance(img, str):
                                sample_imgs.append(
                                    Image.open(img.lstrip("./").replace("/", os.sep)).convert("RGB")
                                )
                            elif img is not None:
                                sample_imgs.append(img)
                            # 传给 apply_chat_template 时只保留 type 标记
                            new_content.append({"type": "image"})
                        else:
                            new_content.append(item)
                    new_messages.append({**msg, "content": new_content})
                else:
                    new_messages.append(msg)
            processed_messages.append(new_messages)
            extracted_images.append(sample_imgs)

        text = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in processed_messages
        ]

        # 合并外部传入的图片和 messages 中提取的图片
        if images_list is not None:
            for i, ext_imgs in enumerate(extracted_images):
                if i < len(images_list) and images_list[i] is not None:
                    external = images_list[i]
                    if not isinstance(external, list):
                        external = [external]
                    # 外部图片（如题目原图）放前面，sandbox 图片放后面
                    ext_imgs = external + ext_imgs
                extracted_images[i] = ext_imgs

        # 统一限制图片尺寸，防止 vision tower patch 数爆炸导致 OOM
        MAX_IMAGE_SIDE = 256
        def _resize_if_needed(img):
            if max(img.size) > MAX_IMAGE_SIDE:
                ratio = MAX_IMAGE_SIDE / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img

        has_images = any(len(imgs) > 0 for imgs in extracted_images)
        if has_images:
            pil_images = []
            for imgs in extracted_images:
                for img in imgs:
                    if isinstance(img, str):
                        img = Image.open(img.lstrip("./").replace("/", os.sep)).convert("RGB")
                    img = _resize_if_needed(img)
                    pil_images.append(img)
            inputs = self.processor(text=text, images=pil_images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

        # ========== 计算每个样本在 pixel_values / image_grid_thw 中的索引范围 ==========
        num_images_per_sample = [len(imgs) for imgs in extracted_images]
        total_images = sum(num_images_per_sample)
        if 'image_grid_thw' in inputs and total_images > 0:
            grid = inputs['image_grid_thw']  # [total_images, 3]
            # Qwen2.5-VL 的 pixel_values 第一维是 total_patches
            # 每张图片的 patch 数 = T * H * W（grid 的三维乘积）
            num_patches_per_image = grid[:, 0] * grid[:, 1] * grid[:, 2]  # [total_images]

            image_cumsum = [0]
            for n in num_images_per_sample:
                image_cumsum.append(image_cumsum[-1] + n)

            patch_cumsum = [0]
            for i in range(len(num_images_per_sample)):
                start_img = image_cumsum[i]
                end_img = image_cumsum[i + 1]
                num_patches = num_patches_per_image[start_img:end_img].sum().item()
                patch_cumsum.append(patch_cumsum[-1] + num_patches)

            inputs['image_ranges'] = [
                {
                    'pixel_values': (patch_cumsum[i], patch_cumsum[i + 1]),
                    'image_grid_thw': (image_cumsum[i], image_cumsum[i + 1]),
                }
                for i in range(len(num_images_per_sample))
            ]
        else:
            inputs['image_ranges'] = [None] * len(num_images_per_sample)

        # 转移到目标设备（跳过非 tensor 的辅助字段）
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Rollout：从对话历史继续生成（支持多轮）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_from_messages(
        self,
        messages_list: List[List[dict]],
        images_list: Optional[List] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Dict, torch.Tensor]:
        """
        从已有对话历史继续采样生成，同时获取旧策略的 logprobs、ref logprobs、critic values。

        返回:
            responses:      List[str]
            old_logprobs:   [batch, response_len]
            ref_logprobs:   [batch, response_len]
            values:         [batch, seq_len]
            sequences:      [batch, seq_len]
            prompt_len:     int
            inputs:         Dict（含更新后的 attention_mask）
            response_mask:  [batch, response_len]（1=有效token, 0=padding）
        """
        inputs = self._build_inputs_from_messages(messages_list, images_list)
        prompt_len = inputs["input_ids"].shape[1]

        # 临时移除 image_ranges，避免传给 generate 导致不认识的 kwargs 报错
        image_ranges = inputs.pop('image_ranges', None)

        # 采样生成
        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # 生成完成后把 image_ranges 塞回去，供调用方在存储 turn 时使用
        inputs['image_ranges'] = image_ranges

        sequences = outputs.sequences
        batch_size, seq_len = sequences.shape
        response_len = seq_len - prompt_len

        # 更新 attention_mask 覆盖完整序列（prompt + response）
        if "attention_mask" in inputs:
            prompt_attention = inputs["attention_mask"]
            response_attention = torch.ones(
                batch_size, response_len,
                device=prompt_attention.device,
                dtype=prompt_attention.dtype,
            )
            inputs["attention_mask"] = torch.cat([prompt_attention, response_attention], dim=1)

        # 解码 response
        response_ids = sequences[:, prompt_len:]
        responses = self.processor.batch_decode(response_ids, skip_special_tokens=True)

        # 计算 per-token logprobs 和 values
        old_logprobs, _ = self._compute_policy_outputs(
            self.policy, sequences, inputs, prompt_len, no_grad=True
        )
        # Reference 临时上 GPU 计算 KL，计算完强制搬回 CPU 释放显存
        try:
            self.ref.to(self.device)
            ref_logprobs, _ = self._compute_policy_outputs(
                self.ref, sequences, inputs, prompt_len, no_grad=True
            )
        finally:
            self.ref.to("cpu")
            torch.cuda.empty_cache()
        values = self._compute_values(sequences, inputs, prompt_len, no_grad=True)

        # response mask：标记有效 token，排除 generate 产生的 padding
        pad_id = self.processor.tokenizer.pad_token_id
        response_mask = torch.ones(batch_size, response_len, dtype=sequences.dtype, device=sequences.device)
        if pad_id is not None:
            for i in range(batch_size):
                pad_positions = (response_ids[i] == pad_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    response_mask[i, pad_positions[0]:] = 0.0

        return responses, old_logprobs, ref_logprobs, values, sequences, prompt_len, inputs, response_mask

    # ------------------------------------------------------------------
    # 快捷方式：从初始 prompt 生成（单轮场景兼容）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        images: Optional[List] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Dict, torch.Tensor]:
        """兼容旧接口的快捷方式，内部调用 generate_from_messages。"""
        has_images = images is not None and any(img is not None for img in images)
        messages_list = []
        for p, img in zip(prompts, images if images else [None] * len(prompts)):
            content = []
            if has_images and img is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": p})
            messages_list.append([{"role": "user", "content": content}])
        return self.generate_from_messages(messages_list, images)

    # ------------------------------------------------------------------
    # PPO 更新
    # ------------------------------------------------------------------
    def ppo_step(
        self,
        sequences: torch.Tensor,
        original_inputs: Dict,
        prompt_len: int,
        old_logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        ref_logprobs: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        执行PPO 更新（Policy + Critic）。

        返回 loss 统计信息 dict。
        """
        # 统一 dtype，防止 float32 辅助张量混入 bfloat16 计算图导致 backward 报错
        target_dtype = old_logprobs.dtype
        rewards = rewards.to(target_dtype)
        values = values.to(target_dtype)
        ref_logprobs = ref_logprobs.to(target_dtype)
        if response_mask is not None:
            response_mask = response_mask.to(target_dtype)

        advantages, returns = self._compute_gae(
            rewards, old_logprobs, ref_logprobs, values, prompt_len
        )

        # ========== Policy Update（单独 backward，算完立刻释放） ==========
        new_logprobs, entropy = self._compute_policy_outputs(
            self.policy, sequences, original_inputs, prompt_len, no_grad=False
        )
    
        ratio = torch.exp(new_logprobs - old_logprobs)
        if response_mask is not None:
            ratio = ratio * response_mask + (1 - response_mask)
            advantages_masked = advantages * response_mask
            entropy_masked = entropy * response_mask
            mask_sum = response_mask.sum()
            if mask_sum.item() == 0:
                mask_sum = 1.0
        else:
            advantages_masked = advantages
            entropy_masked = entropy
            mask_sum = old_logprobs.numel()
    
        surr1 = ratio * advantages_masked
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_masked
        policy_loss = -torch.min(surr1, surr2).sum() / mask_sum
        entropy_loss = -self.entropy_coef * entropy_masked.sum() / mask_sum
    
        policy_total = policy_loss + entropy_loss
    
        self.policy_optimizer.zero_grad()
        policy_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
    
        # ========== Critic Update（单独 backward） ==========
        new_values_all = self._compute_values(
            sequences, original_inputs, prompt_len, no_grad=False
        )
        new_values = new_values_all[:, prompt_len:prompt_len + old_logprobs.shape[1]]
        # 简化版
        value_loss = F.mse_loss(new_values, returns, reduction="none")
        if response_mask is not None:
            value_loss = value_loss * response_mask
        value_loss = 0.5 * value_loss.sum() / mask_sum

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
    
        # 日志统计（无梯度，不增加显存）
        with torch.no_grad():
            kl_div = ((old_logprobs - ref_logprobs) * (response_mask if response_mask is not None else 1.0)).sum().item() / mask_sum
            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_masked.sum().item() / mask_sum,
                "kl": kl_div,
                "total_loss": (policy_total + self.value_coef * value_loss).item(),
            }

        return stats
