import torch
import time
from attention import GPTModel
from tool import (
    create_dataloader_v1,
    generate_text_and_print,
    train_f,
    calculate_loss_batch,
    plot_values,
    )



cfg = {
    "vocab_size": 50257,
    "context_size": 1024,
    "num_layers": 8,          # 改为与 KDA 相同的层数，确保公平对比
    "num_heads": 12,
    "hidden_size": 768,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "max_new_tokens": 50,
    "temperature": 0.8,       # 添加缺失的 temperature
    "topk": 40,
}

def main(batch_size=16, num_epochs=5,lr=1e-4, eval_freq=1, weight_decay=1e-2, save_fig=False, file_dir=None):
    device = torch.device("cuda")

    with open("../gutenberg_60books/training_data_1.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    train_loader, tokenizer = create_dataloader_v1(train_text, "gpt2", cfg["context_size"], cfg["context_size"], batch_size=batch_size, shuffle=True)

    with open("../gutenberg_10books/training_data_1.txt", "r", encoding="utf-8") as f:
        test_text = f.read()
    test_loader, _ = create_dataloader_v1(test_text, tokenizer, cfg["context_size"], cfg["context_size"], batch_size=batch_size, shuffle=False) 

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_text = "Once upon a time, "

    # initial_train_loss = calculate_loss_batch(train_loader, model, device, num_batches=64)
    # initial_test_loss = calculate_loss_batch(test_loader, model, device, num_batches=64)
    # print(f"Initial Train Loss: {initial_train_loss:.4f}")
    # print(f"Initial Test Loss: {initial_test_loss:.4f}")
    print(f"Start text: {start_text}")
    print("Generating text...")
    generate_text_and_print(model, start_text, tokenizer, device, context_size=cfg["context_size"], max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"], topk=cfg["topk"])
    print("-" * 50 + "\n")
    start = time.time()
    train_losses, test_losses, track_tokens_seen = train_f(model, optimizer, train_loader, test_loader, num_epochs, eval_freq, device, tokenizer, start_text, cfg)
    end = time.time()
    epochs_indices = torch.linspace(0, num_epochs, len(train_losses))
    save_dir = plot_values(epochs_indices, track_tokens_seen, train_losses, test_losses, label='loss', save_fig=save_fig, save_dir=file_dir)
    if file_dir is not None and save_fig:
        print(f"Plots saved to {file_dir}")
        with open(f"{save_dir}/log.txt", "w") as f:
            f.write(f"lr: {lr}\n"
                    f"weight_decay: {weight_decay}\n"
                    f"batch_size: {batch_size}\n"
                    f"num_epochs: {num_epochs}\n"
                    f"eval_freq: {eval_freq}\n"
                    f"final_train_loss: {train_losses[-1]:.4f}\n"
                    f"final_test_loss: {test_losses[-1]:.4f}\n"
                    f"time_spent_on_train: {end-start:.2f} s\n")   

if __name__ == "__main__":
    # train数据集，618个batch， 1024长度
    main(batch_size=64, num_epochs=5, lr=1e-4, eval_freq=128, weight_decay=1e-2, save_fig=False, file_dir="./gpt_results")