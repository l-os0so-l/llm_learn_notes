import os

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt

class GPTDatasetV1(Dataset):
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
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0) # 添加batch维度

def tokens_to_text(tokens, tokenizer):
    tokens = tokens.squeeze(0).tolist()  # 移除batch维度并转换为列表
    return tokenizer.decode(tokens)

def calculate_loss(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits.flatten(0, 1), targets.flatten())
    return loss

def calculate_loss_batch(dataloader, model, device, num_batches=None):
    was_training = model.training 
    model.eval()
    total_loss = 0.0
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
            loss = calculate_loss(inputs, targets, model, device)
            total_loss += loss.item()
    if was_training:
        model.train()
    return total_loss / num_batches if num_batches > 0 else None

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, topk=None, eos_token_id=50256):
    is_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx_used = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_used)[:, -1, :]
        if topk is not None:
            topk_logits, _ = torch.topk(logits, topk)
            logits = torch.where(
                condition=logits < topk_logits[..., [-1]],
                input=torch.tensor(float('-inf')),
                other=logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        idx = torch.cat([idx, next_token], dim=1)
        
        if next_token.item() == eos_token_id:
            break
    if is_training:
        model.train()
    return idx

def generate_text_and_print(model, text, tokenizer, device, context_size, max_new_tokens=20, temperature=0.0, topk=None):
    was_training = model.training
    model.eval()
    encoded = text_to_tokens(text, tokenizer).to(device)
    generated_tokens = generate(model, encoded, max_new_tokens, context_size=context_size, temperature=temperature, topk=topk)
    generated_text = tokens_to_text(generated_tokens, tokenizer)
    print(f"Input text: {text}")
    print(f"Generated text: {generated_text}")
    if was_training:
        model.train()

def train_f(model, optimizer, train_loader, test_loader, epochs, eval_freq, device, tokenizer, start_text, cfg):
    model.to(device)
    model.train()
    train_losses, test_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 0
    for epoch in range(epochs):
        for inputs,targets in train_loader:
            tokens_seen += inputs.numel()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = calculate_loss(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            
            if global_step % eval_freq == 0:
                train_loss = calculate_loss_batch(train_loader, model, device, num_batches=128)
                test_loss = calculate_loss_batch(test_loader, model, device)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, stpe {global_step} : Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
                
            global_step += 1
        print(f"\nEpoch {epoch+1} completed. Total tokens seen: {tokens_seen}")
        print(f"start text is: {start_text}")
        print("Generating text after epoch completion:")
        generate_text_and_print(model, start_text, tokenizer, device, context_size=cfg["context_size"], max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"], topk=cfg["topk"])
        print("-" * 50+"\n")

        
    return  train_losses, test_losses, track_tokens_seen

def plot_values(epochs_seen, examples_seen, train_values, val_values, label='loss', save_fig=False, save_dir=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_seen, train_values, label=f'Train {label}')
    ax1.plot(epochs_seen, val_values, label=f'Val {label}', linestyle='-.')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(label)
    ax1.legend(loc='best')
    
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel('Examples Seen')
    plt.title(f'Train and Val {label} vs Epochs and Examples Seen')
    if save_fig:
        if save_dir is not None:
            save_dir_ = f"{save_dir}/lowest_test_{label}_{min(val_values):.4f}"
            if not os.path.exists(save_dir_):
                os.makedirs(save_dir_)
            save_path = f"{save_dir_}/{label}_plot.png"
            plt.savefig(save_path)
    plt.show()
    return save_dir_