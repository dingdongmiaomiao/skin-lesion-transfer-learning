import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    固定随机种子，尽量保证实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 设置
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    """
    如果目录不存在，则创建目录。
    """
    os.makedirs(path, exist_ok=True)


def save_metrics_csv(history: Dict[str, List[float]], save_path: str):
    """
    保存训练过程指标到 CSV 文件。
    """
    ensure_dir(os.path.dirname(save_path))

    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def plot_training_curves(history: Dict[str, List[float]], save_dir: str):
    """
    绘制并保存 Loss 曲线和 AUC 曲线。
    """
    ensure_dir(save_dir)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["valid_loss"], marker="o", label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_loss_curve.png"), dpi=300)
    plt.close()

    # AUC curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["valid_auc"], marker="o", label="Valid AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_auc_curve.png"), dpi=300)
    plt.close()


def get_device():
    """
    自动选择 GPU 或 CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")