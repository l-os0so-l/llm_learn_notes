"""
GRPO 训练主脚本（通用多轮交互版）
支持命令行运行 和 Jupyter Notebook 直接调用 main(config)。

训练流程（每步处理一个 batch，每个 prompt 采样 G 个 episode）：
  1. 将 batch 中每个 prompt 复制 G 份，得到 expanded_batch
  2. 对每个 episode（共 batch_size * G 个）进行最多 max_rounds 轮交互：
     - 模型根据当前历史生成 response
     - 若包含 <answer>，episode 结束，计算 final_reward
     - 否则执行代码，将结果加入历史，继续下一轮
  3. 若达到 max_rounds 仍无 <answer>，final_reward = timeout_penalty
  4. 每个 episode 的总 reward = sum(所有 turn 的 reward)
  5. 对每个 prompt 的 G 个 episode，标准化 reward 得到 advantage
  6. 同一 episode 内的所有 turn 共享该 advantage
  7. 按轮次分组，调用 grpo_step 进行 Policy 更新

与 PPO 的核心差异：
  - 无 Critic、无 Value Loss、无 GAE
  - Advantage 来自组内相对标准化（GRPO）
  - 每个 prompt 采样 group_size 个独立 episode
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict
from peft import LoraConfig
from grpo_trainer import GRPOVLMTrainer
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
            for name, model in [("policy", trainer.policy), ("ref", trainer.ref)]:
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
    output_dir: str = "Part2/GRPO/checkpoints"
    lr: float = 1e-5
    batch_size: int = 2
    max_steps: int = 101
    save_interval: int = 33
    max_new_tokens: int = 512
    max_rounds: int = 3
    timeout_penalty: float = -1.3
    # GRPO 超参数
    group_size: int = 4          # 每个 prompt 采样的 episode 数量
    grpo_epochs: int = 3
    eps_clip: float = 0.1
    beta_kl: float = 0.01
    debug: bool = False
    rm_model_path: str = "Part2/model/Qwen3.5-4B"
    output_res_dir: str = "Part2/GRPO/res"


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
    """构建初始对话历史。返回 list，每个元素是一条样本的 messages。"""
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


def _build_round_batch_grpo(turns: List[Dict], pad_token_id: int) -> Dict:
    """
    把同轮次的多个 episode turn 拼成一个 batch（GRPO 专用）。
    与 PPO 版本相比，去除了 values 字段，保留了 advantages 字段。

    Args:
        turns: list of dict，每个 dict 包含:
            'seq', 'inp', 'plen', 'lp_old', 'lp_ref', 'advantage', 'resp_mask'
        pad_token_id: padding token id

    Returns:
        dict，可直接传给 grpo_step
    """
    sequences = _pad_2d([t['seq'] for t in turns], pad_token_id)

    inputs_list = [t['inp'] for t in turns]
    max_seq_len = max(inp['input_ids'].shape[1] for inp in inputs_list)

    merged_inputs = {}
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

    for key in ['pixel_values', 'image_grid_thw']:
        tensors = [t['inp'][key] for t in turns if key in t['inp'] and t['inp'][key] is not None]
        if tensors:
            merged_inputs[key] = torch.cat(tensors, dim=0)

    for key in ['image_rotary_emb']:
        if key in inputs_list[0]:
            merged_inputs[key] = inputs_list[0][key]

    prompt_len = turns[0]['plen']

    max_resp_len = max(t['lp_old'].shape[1] for t in turns)
    lp_old_list, lp_ref_list, resp_mask_list = [], [], []
    for t in turns:
        lp_old, lp_ref = t['lp_old'], t.get('lp_ref')
        resp_mask = t['resp_mask']
        if lp_old.shape[1] < max_resp_len:
            pad_len = max_resp_len - lp_old.shape[1]
            pad = torch.full((lp_old.shape[0], pad_len), 0.0,
                             dtype=lp_old.dtype, device=lp_old.device)
            lp_old = torch.cat([lp_old, pad], dim=1)
            lp_ref = torch.cat([lp_ref, pad], dim=1)
            mask_pad = torch.zeros(resp_mask.shape[0], pad_len,
                                   dtype=resp_mask.dtype, device=resp_mask.device)
            resp_mask = torch.cat([resp_mask, mask_pad], dim=1)
        lp_old_list.append(lp_old)
        lp_ref_list.append(lp_ref)
        resp_mask_list.append(resp_mask)

    lp_old = torch.cat(lp_old_list, dim=0)
    resp_mask = torch.cat(resp_mask_list, dim=0)

    advantages = torch.tensor([t['advantage'] for t in turns],
                              dtype=sequences.dtype, device=sequences.device)

    result = {
        'sequences': sequences,
        'original_inputs': merged_inputs,
        'prompt_len': prompt_len,
        'old_logprobs': lp_old,
        'advantages': advantages,
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

    # ==================== 初始化 Trainer ====================
    trainer = GRPOVLMTrainer(
        model_path=config.model_path,
        policy_lora_config=lora_config,
        lr=config.lr,
        grpo_epochs=config.grpo_epochs,
        eps_clip=config.eps_clip,
        beta_kl=config.beta_kl,
        max_new_tokens=config.max_new_tokens,
    )
    pad_token_id = trainer.processor.tokenizer.pad_token_id

    # ==================== 初始化 LLM 奖励模型 ====================
    llm_rm = LLMRewardModel(config.rm_model_path)

    # 调试日志文件
    debug_log_path = os.path.join(config.output_res_dir, "debug_grpo_episodes.log")
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
    _log_print("开始 GRPO 多轮交互训练")
    _log_print(f"  batch_size={config.batch_size}, group_size={config.group_size}")
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
        group_size = config.group_size
        expanded_size = batch_size * group_size

        # ==========================================================
        # Phase 1: 扩展 batch，为每个 prompt 采样 G 个 episode
        # ==========================================================
        expanded_questions = [q for q in batch["questions"] for _ in range(group_size)]
        expanded_answers = [a for a in batch["answers"] for _ in range(group_size)]
        expanded_images = [img for img in batch["images"] for _ in range(group_size)]
        expanded_types = [t for t in batch["types"] for _ in range(group_size)]

        # episode 索引映射：expanded_idx -> (prompt_idx, group_idx)
        def prompt_idx(expanded_idx: int) -> int:
            return expanded_idx // group_size

        episodes = [[] for _ in range(expanded_size)]
        messages_list = build_initial_messages({
            "questions": expanded_questions,
            "images": expanded_images,
        })
        active = list(range(expanded_size))

        for round_idx in range(config.max_rounds):
            if not active:
                break

            active_messages = [messages_list[i] for i in active]
            active_images = [expanded_images[i] for i in active]
            gpu_mem(f"before_generate_step{step}_round{round_idx}")
            try:
                resp, lp_old, lp_ref, seq, plen, inp, resp_mask = trainer.generate_from_messages(
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
                            turn_inp[k] = v
                    else:
                        turn_inp[k] = v[idx:idx+1]

                turn = {
                    'resp': resp[idx],
                    'seq': seq[idx:idx+1],
                    'lp_old': lp_old[idx:idx+1],
                    'plen': plen,
                    'inp': turn_inp,
                    'round': round_idx + 1,
                    'resp_mask': resp_mask[idx:idx+1],
                    'episode_idx': i,  # 记录所属 episode
                }
                turn['lp_ref'] = lp_ref[idx:idx+1]
                episodes[i].append(turn)

            still_active = []
            for idx, i in enumerate(active):
                if has_answer(resp[idx]):
                    pass
                else:
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

                    episodes[i][-1]['r'] = r_int

                    messages_list[i] = messages_list[i].copy()
                    messages_list[i].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": resp[idx]}]
                    })
                    user_content = [{"type": "text", "text": sandbox_result["text"]}]
                    for img in sandbox_result.get("images", []):
                        user_content.insert(0, {"type": "image", "image": img})
                    messages_list[i].append({
                        "role": "user",
                        "content": user_content
                    })
                    MAX_HISTORY_ROUNDS = 1
                    initial_len = 1
                    max_total_len = initial_len + MAX_HISTORY_ROUNDS * 2
                    if len(messages_list[i]) > max_total_len:
                        messages_list[i] = messages_list[i][:initial_len] + messages_list[i][-(MAX_HISTORY_ROUNDS * 2):]
                    still_active.append(i)

            active = still_active

        # ==========================================================
        # Phase 2: 计算每轮 reward 和每个 episode 的总 reward
        # ==========================================================
        episode_total_rewards = [0.0] * expanded_size

        for i in range(expanded_size):
            ep = episodes[i]
            last_turn = ep[-1]
            resp = last_turn['resp']
            ans_flag = has_answer(resp)

            if ans_flag:
                last_turn['r'] = compute_final_reward(
                    last_turn['resp'],
                    expanded_answers[i],
                    expanded_types[i],
                    question=expanded_questions[i],
                    llm_rm=llm_rm,
                )
            else:
                last_turn['r'] = config.timeout_penalty

            episode_total_rewards[i] = sum(turn.get('r', 0.0) for turn in ep)

        # ==========================================================
        # Phase 3: GRPO Advantage 计算（按 prompt 分组标准化）
        # ==========================================================
        rewards_tensor = torch.tensor(episode_total_rewards, dtype=torch.float32)
        advantages = GRPOVLMTrainer.compute_grpo_advantages(
            rewards_tensor, group_size=group_size
        )
        advantages_list = advantages.tolist()

        # 将 advantage 写入每个 episode 的每个 turn
        for i in range(expanded_size):
            adv = advantages_list[i]
            for turn in episodes[i]:
                turn['advantage'] = adv

        # ==========================================================
        # Phase 4: GRPO Update（按轮次分组，同轮次拼 batch）
        # ==========================================================
        max_len = max(len(ep) for ep in episodes)
        all_stats = []

        for epoch in range(config.grpo_epochs):
            for round_idx in range(max_len):
                torch.cuda.empty_cache()
                round_turns = [ep[round_idx] for ep in episodes if round_idx < len(ep)]
                if not round_turns:
                    continue
                gpu_mem(f"before_grpo_step{step}_epoch{epoch}_round{round_idx}")
                batch_data = _build_round_batch_grpo(round_turns, pad_token_id)
                try:
                    stats = trainer.grpo_step(
                        sequences=batch_data['sequences'],
                        original_inputs=batch_data['original_inputs'],
                        prompt_len=batch_data['prompt_len'],
                        old_logprobs=batch_data['old_logprobs'],
                        ref_logprobs=batch_data['ref_logprobs'],
                        advantages=batch_data['advantages'],
                        response_mask=batch_data.get('response_mask'),
                    )
                except torch.OutOfMemoryError as e:
                    diagnose_oom(
                        f"grpo_step_step{step}_epoch{epoch}_round{round_idx}",
                        trainer=trainer,
                        local_vars=batch_data,
                    )
                    raise
                all_stats.append(stats)

        avg_stats = {
            k: sum(s[k] for s in all_stats) / len(all_stats)
            for k in all_stats[0]
        } if all_stats else {}

        # ==========================================================
        # 日志
        # ==========================================================
        if step % 3 == 0 and avg_stats:
            num_rounds_list = [len(ep) for ep in episodes]
            avg_rounds = sum(num_rounds_list) / len(num_rounds_list)
            timeout_rate = sum(1 for ep in episodes if not has_answer(ep[-1]['resp'])) / len(episodes)

            r_final_list = [ep[-1]['r'] for ep in episodes]
            # 按 prompt 分组计算平均总 reward
            prompt_avg_rewards = []
            for b in range(batch_size):
                group_rewards = episode_total_rewards[b * group_size:(b + 1) * group_size]
                prompt_avg_rewards.append(sum(group_rewards) / len(group_rewards))

            metrics_history["steps"].append(step)
            metrics_history["total_loss"].append(avg_stats.get("total_loss", 0))
            metrics_history["r_final_mean"].append(sum(r_final_list) / len(r_final_list))
            metrics_history["timeout_rate"].append(timeout_rate)
            metrics_history["avg_rounds"].append(avg_rounds)

            _log_print(
                f"Step {step:04d} | "
                f"loss={avg_stats.get('total_loss', 0):.4f} | "
                f"policy={avg_stats.get('policy_loss', 0):.4f} | "
                f"kl={avg_stats.get('kl', 0):.4f} | "
                f"avg_rounds={avg_rounds:.1f} | "
                f"timeout={timeout_rate:.1%} | "
                f"r_final_mean={sum(r_final_list)/len(r_final_list):.3f} | "
                f"prompt_reward_mean={sum(prompt_avg_rewards)/len(prompt_avg_rewards):.3f}"
            )

            longest_ep = max(episodes, key=lambda ep: len(ep))
            final_resp = longest_ep[-1]['resp']
            _log_print(f"  Sample (R{len(longest_ep)}): {final_resp[:120]}...")

            with open(metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, ensure_ascii=False, indent=2)

        # ==========================================================
        # 保存 checkpoint
        # ==========================================================
        if step % config.save_interval == 0:
            save_path = os.path.join(config.output_dir, f"step_{step + 1}")
            os.makedirs(save_path, exist_ok=True)
            trainer.policy.save_pretrained(os.path.join(save_path, "policy"))
            _log_print(f"[Saved] Checkpoint -> {save_path}")

    # 保存最终模型
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    trainer.policy.save_pretrained(os.path.join(final_path, "policy"))
    _log_print(f"\n训练完成！最终模型保存到: {final_path}")

    # ==================== 绘制并保存指标图 ====================
    if metrics_history["steps"]:
        os.makedirs(config.output_res_dir, exist_ok=True)
        plot_save_path = os.path.join(config.output_res_dir, "training_metrics.png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("GRPO Training Metrics", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(metrics_history["steps"], metrics_history["total_loss"], "b-o", markersize=4, linewidth=1.2)
        ax.set_title("Total Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(metrics_history["steps"], metrics_history["r_final_mean"], "g-o", markersize=4, linewidth=1.2)
        ax.set_title("Final Reward Mean")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(metrics_history["steps"], metrics_history["timeout_rate"], "r-o", markersize=4, linewidth=1.2)
        ax.set_title("Timeout Rate")
        ax.set_xlabel("Step")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

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
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--save_interval", type=int, default=TrainConfig.save_interval)
    parser.add_argument("--max_new_tokens", type=int, default=TrainConfig.max_new_tokens)
    parser.add_argument("--max_rounds", type=int, default=TrainConfig.max_rounds, help="最大交互轮数")
    parser.add_argument("--timeout_penalty", type=float, default=TrainConfig.timeout_penalty, help="超时无答案的惩罚")
    parser.add_argument("--group_size", type=int, default=TrainConfig.group_size, help="每个 prompt 采样的 episode 数")
    parser.add_argument("--grpo_epochs", type=int, default=TrainConfig.grpo_epochs)
    parser.add_argument("--eps_clip", type=float, default=TrainConfig.eps_clip)
    parser.add_argument("--beta_kl", type=float, default=TrainConfig.beta_kl)
    parser.add_argument("--debug", action="store_true", default=TrainConfig.debug)
    parser.add_argument("--rm_model_path", type=str, default=TrainConfig.rm_model_path, help="LLM 奖励模型路径")
    cli_args = parser.parse_args()

    config = TrainConfig(**vars(cli_args))
    main(config)
