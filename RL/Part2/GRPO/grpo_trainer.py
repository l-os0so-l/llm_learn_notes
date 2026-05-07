"""
GRPO 训练器（VLM 完整版）
- Policy Model: Qwen VL + LoRA（可训练）
- Reference Model: Qwen VL（冻结，用于 KL 参考分布）
- 无 Critic Model：通过组内相对奖励标准化估计 advantage

GRPO 核心差异（对比 PPO）：
  1. 移除 Critic / ValueHead，减少显存和计算开销
  2. Advantage = (reward - group_mean) / (group_std + eps)
  3. Loss 为 sample-level：每条 response 内 token 平均后再跨样本平均
  4. KL penalty 直接放入 objective（无 token-level reward shaping）
  5. 无 entropy bonus、无 value loss、无 GAE

尽可能复用 PPO 的输入构建、生成、logprob 计算逻辑。
"""
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from typing import List, Optional, Tuple, Dict
import os
from PIL import Image


class GRPOVLMTrainer:
    def __init__(
        self,
        model_path: str,
        policy_lora_config: LoraConfig,
        lr: float = 1e-5,
        grpo_epochs: int = 4,
        eps_clip: float = 0.2,
        beta_kl: float = 0.01,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.grpo_epochs = grpo_epochs
        self.eps_clip = eps_clip
        self.beta_kl = beta_kl
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.policy = get_peft_model(base_policy, policy_lora_config)
        for p in self.policy.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype)
        self.policy.print_trainable_parameters()

        # ==================== Reference Model ====================
        print("Loading Reference Model...")
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

        # ==================== Processor & Optimizer ====================
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 256 * 256

        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.policy_optimizer = torch.optim.AdamW(policy_params, lr=lr)

    # ------------------------------------------------------------------
    # 输入构建（与 PPO 完全一致）
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

        response_logits = logits[:, start:end, :]               # [batch, response_len, vocab_size]
        response_target_ids = sequences[:, prompt_len:prompt_len + response_len]  # [batch, response_len]

        log_probs_dist = F.log_softmax(response_logits, dim=-1)                       # [batch, response_len, vocab_size]
        log_probs = log_probs_dist.gather(dim=-1, index=response_target_ids.unsqueeze(-1)).squeeze(-1)

        probs = torch.exp(log_probs_dist)
        entropy = -(probs * log_probs_dist).sum(dim=-1)         # [batch, response_len]

        return log_probs, entropy

    # ------------------------------------------------------------------
    # 从对话历史构建输入（支持多轮交互，与 PPO 完全一致）
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
        解决 BS>1 时 GRPO update 的数据污染问题。
        """
        from PIL import Image

        extracted_images = []
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

        if images_list is not None:
            for i, ext_imgs in enumerate(extracted_images):
                if i < len(images_list) and images_list[i] is not None:
                    external = images_list[i]
                    if not isinstance(external, list):
                        external = [external]
                    ext_imgs = external + ext_imgs
                extracted_images[i] = ext_imgs

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

        num_images_per_sample = [len(imgs) for imgs in extracted_images]
        total_images = sum(num_images_per_sample)
        if 'image_grid_thw' in inputs and total_images > 0:
            grid = inputs['image_grid_thw']
            num_patches_per_image = grid[:, 0] * grid[:, 1] * grid[:, 2]

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
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, int, Dict, torch.Tensor]:
        """
        从已有对话历史继续采样生成，同时获取旧策略的 logprobs、ref logprobs。

        返回:
            responses:      List[str]
            old_logprobs:   [batch, response_len]
            ref_logprobs:   [batch, response_len]
            sequences:      [batch, seq_len]
            prompt_len:     int
            inputs:         Dict（含更新后的 attention_mask）
            response_mask:  [batch, response_len]（1=有效token, 0=padding）
        """
        inputs = self._build_inputs_from_messages(messages_list, images_list)
        prompt_len = inputs["input_ids"].shape[1]

        image_ranges = inputs.pop('image_ranges', None)

        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        inputs['image_ranges'] = image_ranges

        sequences = outputs.sequences
        batch_size, seq_len = sequences.shape
        response_len = seq_len - prompt_len

        if "attention_mask" in inputs:
            prompt_attention = inputs["attention_mask"]
            response_attention = torch.ones(
                batch_size, response_len,
                device=prompt_attention.device,
                dtype=prompt_attention.dtype,
            )
            inputs["attention_mask"] = torch.cat([prompt_attention, response_attention], dim=1)

        response_ids = sequences[:, prompt_len:]
        responses = self.processor.batch_decode(response_ids, skip_special_tokens=True)

        old_logprobs, _ = self._compute_policy_outputs(
            self.policy, sequences, inputs, prompt_len, no_grad=True
        )
        try:
            self.ref.to(self.device)
            ref_logprobs, _ = self._compute_policy_outputs(
                self.ref, sequences, inputs, prompt_len, no_grad=True
            )
        finally:
            self.ref.to("cpu")
            torch.cuda.empty_cache()

        pad_id = self.processor.tokenizer.pad_token_id
        response_mask = torch.ones(batch_size, response_len, dtype=sequences.dtype, device=sequences.device)
        if pad_id is not None:
            for i in range(batch_size):
                pad_positions = (response_ids[i] == pad_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    response_mask[i, pad_positions[0]:] = 0.0

        return responses, old_logprobs, ref_logprobs, sequences, prompt_len, inputs, response_mask

    # ------------------------------------------------------------------
    # 快捷方式：从初始 prompt 生成（单轮场景兼容）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        images: Optional[List] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, int, Dict, torch.Tensor]:
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
    # GRPO Advantage：组内标准化
    # ------------------------------------------------------------------
    @staticmethod
    def compute_grpo_advantages(
        rewards: torch.Tensor,
        group_size: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        计算 GRPO 的组内相对 advantage。

        Args:
            rewards: [batch]  每条样本（episode）的总奖励
            group_size: int   每个 prompt 对应的 episode 数量
            eps: float        数值稳定常数

        Returns:
            advantages: [batch]  标准化后的 advantage

        说明：
            batch 必须能被 group_size 整除，前 group_size 个元素对应第 1 个 prompt 的 G 个 episode，
            接下来 group_size 个对应第 2 个 prompt，以此类推。
        """
        batch_size = rewards.shape[0]
        assert batch_size % group_size == 0, f"batch_size {batch_size} 必须能被 group_size {group_size} 整除"
        num_groups = batch_size // group_size

        advantages = torch.zeros_like(rewards)
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group_rewards = rewards[start:end]
            mean = group_rewards.mean()
            std = group_rewards.std()
            if std.item() < eps:
                std = torch.tensor(1.0, dtype=std.dtype, device=std.device)
            advantages[start:end] = (group_rewards - mean) / std

        return advantages

    def grpo_step(
        self,
        sequences: torch.Tensor,
        original_inputs: Dict,
        prompt_len: int,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        执行多轮 GRPO 更新（仅 Policy，无 Critic）。

        Args:
            advantages: [batch]  每条样本（turn）共享的 episode-level advantage

        返回 loss 统计信息 dict。
        """
        target_dtype = old_logprobs.dtype
        ref_logprobs = ref_logprobs.to(target_dtype)
        advantages = advantages.to(target_dtype)
        if response_mask is not None:
            response_mask = response_mask.to(target_dtype)

        new_logprobs, _ = self._compute_policy_outputs(
            self.policy, sequences, original_inputs, prompt_len, no_grad=False
        )

        ratio = torch.exp(new_logprobs - old_logprobs)

        # 扩展 advantages 到 token 维度
        advantages_expanded = advantages.unsqueeze(-1)  # [batch, 1]

        # KL divergence: Schulman 2020 unbiased estimator
        # D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
        log_ratio = ref_logprobs - new_logprobs  # log(pi_ref / pi_theta)
        ratio_kl = torch.exp(log_ratio)          # pi_ref / pi_theta
        kl_penalty = (ratio_kl - log_ratio - 1)

        if response_mask is not None:
            ratio = ratio * response_mask + (1 - response_mask)
            advantages_masked = advantages_expanded * response_mask
            kl_penalty = kl_penalty * response_mask
            mask_sum = response_mask.sum()
            if mask_sum.item() == 0:
                mask_sum = 1.0
        else:
            advantages_masked = advantages_expanded
            mask_sum = old_logprobs.numel()

        # PPO-clip（token-level 计算，然后 sample-level 平均）
        surr1 = ratio * advantages_masked
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_masked
        policy_loss = -torch.min(surr1, surr2).sum() / mask_sum

        # KL penalty（token-level 然后平均）
        kl_loss = kl_penalty.sum() / mask_sum
        total_loss = policy_loss + self.beta_kl * kl_loss

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        with torch.no_grad():
            stats = {
                "policy_loss": policy_loss.item(),
                "kl": kl_loss.item(),
                "total_loss": total_loss.item(),
            }

        return stats

    
    # ------------------------------------------------------------------
    # GRPO 更新
    # ------------------------------------------------------------------
    # def grpo_step(
    #     self,
    #     sequences: torch.Tensor,
    #     original_inputs: Dict,
    #     prompt_len: int,
    #     old_logprobs: torch.Tensor,
    #     ref_logprobs: torch.Tensor,
    #     advantages: torch.Tensor,
    #     response_mask: Optional[torch.Tensor] = None,
    # ) -> Dict:
    #     """
    #     执行多轮 GRPO 更新（仅 Policy，无 Critic）。

    #     Args:
    #         advantages: [batch]  每条样本（turn）共享的 episode-level advantage

    #     返回 loss 统计信息 dict。
    #     """
    #     target_dtype = old_logprobs.dtype
    #     ref_logprobs = ref_logprobs.to(target_dtype)
    #     advantages = advantages.to(target_dtype)
    #     if response_mask is not None:
    #         response_mask = response_mask.to(target_dtype)

    #     new_logprobs, _ = self._compute_policy_outputs(
    #         self.policy, sequences, original_inputs, prompt_len, no_grad=False
    #     )

    #     ratio = torch.exp(new_logprobs - old_logprobs)

    #     # 扩展 advantages 到 token 维度
    #     advantages_expanded = advantages.unsqueeze(-1)  # [batch, 1]

    #     if response_mask is not None:
    #         ratio = ratio * response_mask + (1 - response_mask)
    #         advantages_masked = advantages_expanded * response_mask
    #         kl_penalty = (ref_logprobs - new_logprobs) * response_mask
    #         mask_sum = response_mask.sum()
    #         if mask_sum.item() == 0:
    #             mask_sum = 1.0
    #     else:
    #         advantages_masked = advantages_expanded
    #         kl_penalty = ref_logprobs - new_logprobs
    #         mask_sum = old_logprobs.numel()

    #     # PPO-clip（token-level 计算，然后 sample-level 平均）
    #     surr1 = ratio * advantages_masked
    #     surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_masked
    #     policy_loss = -torch.min(surr1, surr2).sum() / mask_sum

    #     # KL penalty（token-level 然后平均）
    #     kl_loss = kl_penalty.sum() / mask_sum
    #     total_loss = policy_loss + self.beta_kl * kl_loss

    #     self.policy_optimizer.zero_grad()
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
    #     self.policy_optimizer.step()

    #     with torch.no_grad():
    #         stats = {
    #             "policy_loss": policy_loss.item(),
    #             "kl": kl_loss.item(),
    #             "total_loss": total_loss.item(),
    #         }

    #     return stats

    