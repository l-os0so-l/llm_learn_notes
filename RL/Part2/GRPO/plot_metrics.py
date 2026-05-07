"""
独立脚本：读取 PPO 训练指标 JSON 文件并画图。

用法:
    # 自动查找最新的 metrics 文件
    python Part2/PPO/plot_metrics.py

    # 指定具体的 metrics 文件
    python Part2/PPO/plot_metrics.py --metrics_path Part2/PPO/res/logs/20260425_200355/metrics_history_20260425_200355.json
"""
import os
import json
import argparse
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latest_metrics(root_dir: str = "Part2/GRPO/res/logs") -> str | None:
    """
    在 logs 目录下递归查找所有 metrics_history_*.json 文件，
    返回修改时间最新的那个。
    """
    pattern = os.path.join(root_dir, "**", "metrics_history_*.json")
    candidates = glob(pattern, recursive=True)
    if not candidates:
        return None
    # 按文件修改时间排序，取最新
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def plot_and_save(metrics_path: str):
    """读取指标文件并画图，保存到 metrics 文件所在目录。"""
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_history = json.load(f)

    if not metrics_history.get("steps"):
        print("[Warn] 没有收集到足够的数据用于绘图。")
        return

    # 保存到 metrics 文件所在的文件夹
    output_dir = os.path.dirname(os.path.abspath(metrics_path))
    plot_save_path = os.path.join(output_dir, "training_metrics.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GRPO Training Metrics", fontsize=14, fontweight="bold")

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

    print(f"[Plot] 训练指标图已保存到: {plot_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training metrics from JSON file.")
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="Path to metrics_history_*.json. If not provided, auto-find the latest one.",
    )
    args = parser.parse_args()

    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = find_latest_metrics()
        if metrics_path:
            print(f"[Auto-detect] 使用最新的指标文件: {metrics_path}")
        else:
            print("[Error] 未找到任何 metrics_history_*.json 文件。")
            print("        请确认训练已运行并生成了指标文件，或通过 --metrics_path 手动指定。")
            return

    if not os.path.exists(metrics_path):
        print(f"[Error] 指定的文件不存在: {metrics_path}")
        return

    plot_and_save(metrics_path)


if __name__ == "__main__":
    main()
