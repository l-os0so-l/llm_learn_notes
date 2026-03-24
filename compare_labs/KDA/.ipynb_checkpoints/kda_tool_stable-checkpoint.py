import os

import matplotlib.pyplot as plt
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch import nn


class GPTDatasetV1(Dataset):
    """与 tool.py 相同"""
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids)-max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, tokenizer_name='gpt2', max_length=256, stride=128, 
                         batch_size=32, shuffle=True, num_workers=0, drop_last=True):
    if type(tokenizer_name) == str:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
    else:
        tokenizer = tokenizer_name
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           drop_last=drop_last, num_workers=num_workers)
    return dataloader, tokenizer


def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def tokens_to_text(tokens, tokenizer):
    tokens = tokens.squeeze(0).tolist()
    return tokenizer.decode(tokens)


def check_nan_inf(tensor, name="tensor"):
    """检查张量是否包含 NaN 或 Inf"""
    if torch.isnan(tensor).any():
        print(f"警告: {name} 包含 NaN!")
        return True
    if torch.isinf(tensor).any():
        print(f"警告: {name} 包含 Inf!")
        return True
    return False


# 全局复用 criterion 以提高效率
_criterion = nn.CrossEntropyLoss()

def calculate_loss(inputs, targets, model, device, check_numerics=False):
    """计算损失 - 适用于 KimiModel，添加数值检查"""
    inputs, targets = inputs.to(device), targets.to(device)
    
    # 使用混合精度上下文（如果外部未包裹）
    logits = model(inputs)

    # 检查 logits 数值
    if check_numerics and check_nan_inf(logits, "logits"):
        print(f"logits 范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        # 尝试修复（临时措施）
        logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                            torch.zeros_like(logits), logits)

    # 展平为 [B*T, vocab_size] 和 [B*T]
    loss = _criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    # 检查 loss 数值
    if check_numerics and (torch.isnan(loss) or torch.isinf(loss)):
        print(f"警告: loss = {loss.item()}")

    return loss


def calculate_loss_batch(dataloader, model, device, num_batches=None, check_numerics=False):
    """计算批次平均损失，添加数值保护"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    if len(dataloader) == 0:
        return None
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            try:
                loss = calculate_loss(inputs, targets, model, device, check_numerics)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                else:
                    print(f"警告: 批次 {i} 的损失为 NaN/Inf，已跳过")
            except RuntimeError as e:
                print(f"警告: 批次 {i} 计算损失时出错: {e}")
                continue

    if was_training:
        model.train()

    if valid_batches == 0:
        print("警告: 所有批次都无效，返回 None")
        return None

    return total_loss / valid_batches


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, 
             topk=None, eos_token_id=50256, use_state=False):
    """
    为 KimiModel 定制的生成函数
    添加数值稳定性保护
    """
    is_training = model.training
    model.eval()

    states = None
    if use_state and hasattr(model, 'init_states'):
        states = model.init_states(idx.size(0), idx.device)

    for step in range(max_new_tokens):
        with torch.no_grad():
            if use_state and states is not None:
                # 增量解码
                if step == 0:
                    # 第一次使用完整的上下文（或截断到 context_size）
                    idx_used = idx[:, -context_size:]
                    logits, states = model(idx_used, states=None, return_states=True)
                else:
                    # 后续只输入最新的 token，复用状态
                    logits, states = model(idx[:, -1:], states=states, return_states=True)
                logits = logits[:, -1, :]
            else:
                # 普通解码（每次重新计算整个上下文）
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond)[:, -1, :]

            # 数值保护：防止 logits 过大导致 softmax 溢出
            logits = logits.clamp(min=-50, max=50)

        # Top-k 采样
        if topk is not None and topk > 0:
            topk_logits, _ = torch.topk(logits, min(topk, logits.size(-1)))
            kth_val = topk_logits[:, [-1]]
            logits = torch.where(
                logits < kth_val,
                torch.full_like(logits, float('-inf')),
                logits
            )

        # 温度采样
        if temperature > 0.0:
            logits = logits / temperature
            # 再次裁剪防止温度除法后数值过大
            logits = logits.clamp(min=-50, max=50)
            probs = torch.softmax(logits, dim=-1)
            # 数值保护：确保概率有效
            probs = torch.where(torch.isnan(probs) | (probs < 0), 
                               torch.zeros_like(probs), probs)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            # 防止除零
            probs_sum = torch.where(probs_sum == 0, torch.ones_like(probs_sum), probs_sum)
            probs = probs / probs_sum  # 重新归一化
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat([idx, next_token], dim=1)

        # EOS 检查（仅支持 batch_size=1）
        if idx.size(0) == 1 and next_token.item() == eos_token_id:
            break

    if is_training:
        model.train()
    return idx


def generate_text_and_print(model, text, tokenizer, device, context_size, 
                            max_new_tokens=20, temperature=0.0, topk=None,
                            use_state=False):
    """生成文本并打印"""
    was_training = model.training
    model.eval()
    encoded = text_to_tokens(text, tokenizer).to(device)
    generated_tokens = generate(model, encoded, max_new_tokens, 
                                context_size=context_size, 
                                temperature=temperature, topk=topk,
                                use_state=use_state)
    generated_text = tokens_to_text(generated_tokens, tokenizer)
    print(f"Input text: {text}")
    print(f"Generated text: {generated_text}")
    if was_training:
        model.train()


def train_f(model, optimizer, train_loader, test_loader, epochs, eval_freq, 
            device, tokenizer, start_text, cfg):
    """
    训练函数 - 已修复数值稳定性问题
    关键修改：
    1. 添加了全面的 NaN/Inf 检查
    2. 增强了梯度裁剪
    3. 支持 AMP (自动混合精度)
    """
    model.to(device)
    model.train()
    train_losses, test_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 0

    # 数值检查开关（可在 cfg 中设置）
    check_numerics = cfg.get("check_numerics", True)
    grad_clip = cfg.get("grad_clip", 1.0)
    use_amp = cfg.get("use_amp", False)

    # 初始化 GradScaler 用于混合精度
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"开始训练... 梯度裁剪阈值: {grad_clip}")
    print(f"数值检查: {'开启' if check_numerics else '关闭'}")
    print(f"混合精度 (AMP): {'开启' if use_amp else '关闭'}")

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            tokens_seen += inputs.numel()
            inputs, targets = inputs.to(device), targets.to(device)
            
            if global_step % eval_freq == 0:
                print(f"Step {global_step}, Tokens seen: {tokens_seen}")

            # 前向传播（支持 AMP）
            try:
                optimizer.zero_grad()
                
                # with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = calculate_loss(inputs, targets, model, device, check_numerics)

                # 检查 loss 有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: Epoch {epoch+1} Batch {batch_idx} 损失无效 ({loss.item()})，跳过此批次")
                    # 清空可能存在的缩放器状态
                    if use_amp:
                        scaler.update()
                    continue

                # 反向传播（使用 scaler 处理梯度）
                if use_amp:
                    scaler.scale(loss).backward()
                    # 在裁剪前先解缩放
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # 梯度裁剪（必须在 backward 之后，step 之前）
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    if check_numerics:
                        grad_norm_val = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm
                        if grad_norm_val == float('inf') or grad_norm_val != grad_norm_val:  # inf or nan
                            print(f"警告: 梯度范数为 {grad_norm_val}，检测到数值问题")

                # 检查梯度是否包含 NaN（非 AMP 模式已裁剪，AMP 模式在 unscale 后已裁剪）
                has_nan_grad = False
                if check_numerics and not use_amp:  # AMP 模式下 check_nan_inf 可能误报
                    for name, param in model.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            print(f"警告: 参数 {name} 的梯度包含 NaN/Inf")
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f"警告: 检测到 NaN/Inf 梯度，跳过此优化步骤")
                    optimizer.zero_grad()  # 清空无效梯度
                    if use_amp:
                        scaler.update()
                    continue

                # 优化步骤
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                

            except RuntimeError as e:
                print(f"错误: 训练步骤失败 - {e}")
                print("尝试继续训练...")
                optimizer.zero_grad()
                if use_amp:
                    scaler.update()
                continue

            # 评估
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    # 评估时也使用 AMP 以节省显存（可选）
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        train_loss = calculate_loss_batch(train_loader, model, device, 
                                                         num_batches=128, check_numerics=check_numerics)
                        test_loss = calculate_loss_batch(test_loader, model, device, 
                                                        num_batches=None, check_numerics=check_numerics)
                model.train()                
                train_losses.append(train_loss if train_loss is not None else float('inf'))
                test_losses.append(test_loss if test_loss is not None else float('inf'))
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} Step {global_step}: "
                        f"Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
            
            global_step += 1

        # Epoch 结束后的生成测试
        print(f"\nEpoch {epoch+1} completed. Total tokens seen: {tokens_seen}")
        print(f"Start text: {start_text}")
        print("Generating text...")
        try:
            generate_text_and_print(
                model, start_text, tokenizer, device, 
                context_size=cfg["context_size"], 
                max_new_tokens=cfg.get("max_new_tokens", 20),
                temperature=cfg.get("temperature", 0.8),
                topk=cfg.get("topk", None),
                use_state=cfg.get("use_state", False)
            )
        except Exception as e:
            print(f"生成文本时出错: {e}")
        print("-" * 50 + "\n")

    print("训练完成！")
    return train_losses, test_losses, track_tokens_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label='loss', save_fig=False, save_dir=None):
    # 修复：初始化 save_dir_ 防止 UnboundLocalError
    save_dir_ = None
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_seen, train_values, label=f'Train {label}')
    ax1.plot(epochs_seen, val_values, label=f'Val {label}', linestyle='-.')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(label)
    ax1.legend(loc='best')
    
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)  # 隐形图，仅用于对齐刻度
    ax2.set_xlabel('Tokens Seen')
    
    plt.title(f'Train and Val {label} vs Epochs and Tokens Seen')
    
    if save_fig:
        if save_dir is not None:
            save_dir_ = f"{save_dir}/lowest_test_{label}_{min(val_values):.4f}"
            if not os.path.exists(save_dir_):
                os.makedirs(save_dir_)
            save_path = f"{save_dir_}/{label}_plot.png"
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
    plt.show()
    return save_dir_