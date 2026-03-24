import torch
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
from kda_stable import KimiModel
from kda_tool_stable import (
    create_dataloader_v1,
    generate_text_and_print,
    train_f,
    calculate_loss_batch,
    plot_values,
    )
import time

cfg = {
    "vocab_size": 50257,      
    "hidden_size": 768,       
    "n_heads": 12,           
    "n_layers": 8,            
    "chunk_size": 32,         
    "context_size": 1024,    
    "drop_rate": 0.1,         
    "use_short_conv": True,  
    "conv_size": 4,           
    "max_new_tokens": 50,    
    "temperature": 0.8,      
    "topk": 40,               
    "grad_clip": 1.0,        
    "check_numerics": True,   
    "use_amp": False,          
    "use_state": True,       
}

def main(batch_size=16, num_epochs=5, lr=1e-4, eval_freq=1, weight_decay=1e-2, save_fig=False, file_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    try:
        with open("../gutenberg_60books/training_data_1.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
    except FileNotFoundError:
        print("警告: 训练文件未找到，使用示例文本")
        train_text = "Once upon a time, there was a little girl named Alice. " * 1000
    
    train_loader, tokenizer = create_dataloader_v1(
        train_text, "gpt2", cfg["context_size"], cfg["context_size"], 
        batch_size=batch_size, shuffle=True
    )

    try:
        with open("../gutenberg_10books/training_data_1.txt", "r", encoding="utf-8") as f:
            test_text = f.read()
    except FileNotFoundError:
        print("警告: 测试文件未找到，使用训练文本的一部分")
        test_text = train_text[-100000:]  # 使用最后一部分作为测试
    
    test_loader, _ = create_dataloader_v1(
        test_text, tokenizer, cfg["context_size"], cfg["context_size"], 
        batch_size=batch_size, shuffle=False
    ) 

    # 初始化模型和优化器
    model = KimiModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    
    start_text = "Once upon a time, "

    # 初始生成测试（随机初始化）
    print(f"Start text: {start_text}")
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
        print(f"初始生成出错（正常，因为模型未训练）: {e}")
    print("-" * 50 + "\n")
    
    # 训练
    start = time.time()
    train_losses, test_losses, track_tokens_seen = train_f(
        model, optimizer, train_loader, test_loader, num_epochs, 
        eval_freq, device, tokenizer, start_text, cfg
    )
    end = time.time()
    
    # 绘制结果
    epochs_indices = torch.linspace(0, num_epochs, len(train_losses))
    save_dir = plot_values(
        epochs_indices,           # x1: epoch 索引
        track_tokens_seen,        # x2: tokens seen
        train_losses, 
        test_losses, 
        label='loss', 
        save_fig=save_fig, 
        save_dir=file_dir
    )
    

    if file_dir is not None and save_fig and save_dir is not None:
        print(f"Plots saved to {save_dir}")
        log_path = f"{save_dir}/log.txt"
        with open(log_path, "w") as f:
            f.write(f"lr: {lr}\n"
                    f"weight_decay: {weight_decay}\n"
                    f"chunk_size: {cfg['chunk_size']}\n"
                    f"context_size: {cfg['context_size']}\n"
                    f"batch_size: {batch_size}\n"
                    f"num_epochs: {num_epochs}\n"
                    f"eval_freq: {eval_freq}\n"
                    f"final_train_loss: {train_losses[-1]:.4f}\n"
                    f"final_test_loss: {test_losses[-1]:.4f}\n"
                    f"time_spent_on_train: {end-start:.4f} s\n")
        print(f"Log saved to {log_path}")
    elif file_dir is not None:
        print(f"File dir provided but save_fig=False, no files saved")


if __name__ == "__main__":
    main(
        batch_size=16, 
        num_epochs=5, 
        lr=1e-4, 
        eval_freq=50, 
        weight_decay=1e-2, 
        save_fig=False,  
        file_dir="./kda_results"
    )