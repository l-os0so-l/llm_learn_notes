from huggingface_hub import snapshot_download

# snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="./Part2/model/Qwen2.5-VL-3B-Instruct")

snapshot_download("Qwen/Qwen3.5-4B", local_dir="./Part2/model/Qwen3.5-4B", resume_download=True)