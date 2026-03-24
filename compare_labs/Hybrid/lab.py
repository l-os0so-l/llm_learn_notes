import torch
from attention import HybridKimiModel
from kda_tool_stable import create_dataloader_v1
from kda_tool_stable import train_f, generate_text_and_print, plot_values

# 修改后的配置（兼容两种模块）
cfg = {
    "vocab_size": 50257,
    "hidden_size": 768,
    "n_heads": 12,           # KDA使用
    "num_heads": 12,         # 传统Attention使用（兼容）
    "n_layers": 8,           # 必须是4的倍数才能完美实现3:1比例
    "chunk_size": 32,
    "context_size": 1024,
    "drop_rate": 0.1,
    "use_short_conv": True,
    "conv_size": 4,
    "qkv_bias": False,       # 传统Attention使用
    "max_new_tokens": 50,
    "temperature": 0.8,
    "topk": 40,
    "grad_clip": 1.0,
    "check_numerics": True,
    "use_amp": False,
    "use_state": True,       # 混合模型支持部分状态（KDA层有，Attention层无）
}

def main_hybrid(batch_size=2, num_epochs=5, lr=1e-4, weight_decay=1e-2, eval_freq=50, save_fig=False, file_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载（复用你的代码）
    try:
        with open("../gutenberg_60books/training_data_1.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
    except FileNotFoundError:
        train_text = "Once upon a time, there was a little girl named Alice. " * 1000
    
    train_loader, tokenizer = create_dataloader_v1(
        train_text, "gpt2", cfg["context_size"], cfg["context_size"], 
        batch_size=batch_size, shuffle=True
    )
    
    try:
        with open("../gutenberg_10books/training_data_1.txt", "r", encoding="utf-8") as f:
            test_text = f.read()
    except FileNotFoundError:
        test_text = train_text[-100000:]
    
    test_loader, _ = create_dataloader_v1(
        test_text, tokenizer, cfg["context_size"], cfg["context_size"], 
        batch_size=batch_size, shuffle=False
    )
    
    # 初始化混合模型
    print("Initializing Hybrid Model (3 KDA + 1 Attention per block)...")
    model = HybridKimiModel(cfg).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay, 
        betas=(0.9, 0.95)
    )
    
    
    
    start_text = "Once upon a time, "
    
    print(f"\nStart text: {start_text}")
    print("Generating text (initial random weights)...")
    try:
        generate_text_and_print(
            model, start_text, tokenizer, device, 
            context_size=cfg["context_size"], 
            max_new_tokens=cfg["max_new_tokens"], 
            temperature=cfg["temperature"], 
            topk=cfg["topk"],
            use_state=cfg["use_state"]
        )
    except Exception as e:
        print(f"初始生成出错: {e}")
    print("-" * 50 + "\n")
    
    # 训练
    import time
    start = time.time()
    train_losses, test_losses, track_tokens_seen = train_f(
        model, optimizer, train_loader, test_loader, num_epochs, 
        eval_freq, device, tokenizer, start_text, cfg
    )
    end = time.time()

    epochs_indices = torch.linspace(0, num_epochs, len(train_losses))
    save_dir = plot_values(epochs_indices, track_tokens_seen, train_losses, test_losses, label='loss', save_fig=save_fig, save_dir=file_dir)
    if file_dir is not None and save_fig:
        print(f"Plots saved to {file_dir}")
        with open(f"{save_dir}/log.txt", "w") as f:
            f.write(f"lr: {lr}\n"
                    f"weight_decay: {weight_decay}\n"
                    f"batch_size: {batch_size}\n"
                    f"chunk_size: {cfg['chunk_size']}\n"
                    f"context_size: {cfg['context_size']}\n"
                    f"num_epochs: {num_epochs}\n"
                    f"eval_freq: {eval_freq}\n"
                    f"final_train_loss: {train_losses[-1]:.4f}\n"
                    f"final_test_loss: {test_losses[-1]:.4f}\n"
                    f"time_spent_on_train: {end-start:.2f} s\n") 
    
    print(f"\n训练完成！总耗时: {end-start:.2f}秒")
    return model, train_losses, test_losses

if __name__ == "__main__":
    main_hybrid(cfg, batch_size=2, num_epochs=5, lr=1e-4, eval_freq=50)