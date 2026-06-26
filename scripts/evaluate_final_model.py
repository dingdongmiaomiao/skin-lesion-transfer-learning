import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# 允许从 scripts/ 目录导入 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import ISICDataset, get_valid_transforms
from src.models import build_model
from src.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate final model with confusion matrix and ROC curve"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/ISIC_2020/ground_truth.csv",
        help="ground_truth.csv 路径",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/ISIC_2020/train",
        help="图像文件夹路径",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="最终模型 best_model.pth 路径",
    )

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="默认分类阈值",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录",
    )

    return parser.parse_args()


def make_abs_path(path: str):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_model(checkpoint_path: str, model_name: str, device):
    """
    加载训练好的模型。

    注意：
    这里只做评估，所以 freeze=True/False 不影响 forward。
    为了完整匹配 full_finetune 的结构，这里使用 freeze=False。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name_from_ckpt = checkpoint.get("model_name", model_name)

    model = build_model(
        model_name=model_name_from_ckpt,
        pretrained=False,
        freeze=False,
        dropout=0.0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_auc = checkpoint.get("best_auc", None)
    epoch = checkpoint.get("epoch", None)

    print(f"已加载模型: {model_name_from_ckpt}")
    print(f"checkpoint epoch: {epoch}")
    print(f"checkpoint best_auc: {best_auc}")

    return model


@torch.no_grad()
def predict(model, dataloader, device):
    """
    在验证集上预测，返回真实标签、预测概率和 image_name。
    """
    all_labels = []
    all_probs = []
    all_image_names = []

    progress_bar = tqdm(dataloader, desc="Evaluate", leave=False)

    for images, labels, image_names in progress_bar:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().reshape(-1).tolist())
        all_image_names.extend(list(image_names))

    return (
        np.array(all_labels).astype(int),
        np.array(all_probs).astype(float),
        all_image_names,
    )


class ISICDatasetWithName(ISICDataset):
    """
    在原 ISICDataset 基础上额外返回 image_name，方便保存预测结果。
    """

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image_name = str(self.dataframe.iloc[idx]["image_name"])
        return image, label, image_name


def compute_metrics(y_true, y_prob, threshold: float):
    """
    根据指定 threshold 计算分类指标。
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "auc": auc,
        "accuracy": acc,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_youden_threshold(y_true, y_prob):
    """
    使用 Youden Index 寻找较优分类阈值。

    Youden Index = sensitivity + specificity - 1
                 = TPR - FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr

    best_idx = int(np.argmax(youden_index))
    best_threshold = float(thresholds[best_idx])

    return best_threshold


def plot_confusion_matrix(cm, save_path: str, title: str):
    """
    使用 matplotlib 绘制混淆矩阵。
    """
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign (0)", "Malignant (1)"])
    plt.yticks(tick_marks, ["Benign (0)", "Malignant (1)"])

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path: str):
    """
    绘制 ROC 曲线。
    """
    ensure_dir(os.path.dirname(save_path))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Validation Set")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    args.csv_path = make_abs_path(args.csv_path)
    args.image_dir = make_abs_path(args.image_dir)
    args.checkpoint_path = make_abs_path(args.checkpoint_path)
    args.output_dir = make_abs_path(args.output_dir)

    device = get_device()
    print(f"使用设备: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"final_eval_{args.model}_{timestamp}"

    table_dir = os.path.join(args.output_dir, "tables", run_name)
    figure_dir = os.path.join(args.output_dir, "figures", run_name)

    ensure_dir(table_dir)
    ensure_dir(figure_dir)

    # 读取数据，并使用与训练相同的 8:2 分层划分
    df = pd.read_csv(args.csv_path)
    df = df[["image_name", "target"]].copy()
    df["target"] = df["target"].astype(int)

    _, valid_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )

    valid_df = valid_df.reset_index(drop=True)

    print(f"验证集样本数: {len(valid_df)}")
    print("验证集标签分布:")
    print(valid_df["target"].value_counts())

    valid_dataset = ISICDatasetWithName(
        dataframe=valid_df,
        image_dir=args.image_dir,
        transform=get_valid_transforms(args.img_size),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model,
        device=device,
    )

    y_true, y_prob, image_names = predict(
        model=model,
        dataloader=valid_loader,
        device=device,
    )

    # 默认 threshold=0.5
    default_metrics = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
    )

    # Youden threshold
    youden_threshold = find_youden_threshold(y_true, y_prob)

    # 处理极端情况：roc_curve 有时会返回 inf threshold
    if not np.isfinite(youden_threshold):
        youden_threshold = args.threshold

    youden_metrics = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=youden_threshold,
    )

    # 保存预测结果
    pred_df = pd.DataFrame(
        {
            "image_name": image_names,
            "target": y_true,
            "prob_malignant": y_prob,
            "pred_threshold_0.5": (y_prob >= args.threshold).astype(int),
            "pred_youden": (y_prob >= youden_threshold).astype(int),
        }
    )

    pred_path = os.path.join(table_dir, "final_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 保存指标
    metrics_df = pd.DataFrame(
        [
            {"setting": "threshold_0.5", **default_metrics},
            {"setting": "youden_threshold", **youden_metrics},
        ]
    )

    metrics_path = os.path.join(table_dir, "final_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # 保存 JSON，方便报告引用
    metrics_json_path = os.path.join(table_dir, "final_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold_0.5": default_metrics,
                "youden_threshold": youden_metrics,
                "youden_threshold_value": youden_threshold,
                "checkpoint_path": args.checkpoint_path,
                "validation_size": int(len(y_true)),
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    # 绘图：ROC
    roc_path = os.path.join(figure_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_prob, roc_path)

    # 绘图：默认阈值混淆矩阵
    cm_default = np.array(
        [
            [default_metrics["tn"], default_metrics["fp"]],
            [default_metrics["fn"], default_metrics["tp"]],
        ]
    )

    cm_default_path = os.path.join(
        figure_dir,
        "confusion_matrix_threshold_0.5.png",
    )

    plot_confusion_matrix(
        cm=cm_default,
        save_path=cm_default_path,
        title="Confusion Matrix (Threshold = 0.5)",
    )

    # 绘图：Youden 阈值混淆矩阵
    cm_youden = np.array(
        [
            [youden_metrics["tn"], youden_metrics["fp"]],
            [youden_metrics["fn"], youden_metrics["tp"]],
        ]
    )

    cm_youden_path = os.path.join(
        figure_dir,
        "confusion_matrix_youden_threshold.png",
    )

    plot_confusion_matrix(
        cm=cm_youden,
        save_path=cm_youden_path,
        title=f"Confusion Matrix (Youden Threshold = {youden_threshold:.4f})",
    )

    print("\n最终模型评估完成。")

    print("\n默认阈值 0.5 指标:")
    for k, v in default_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\nYouden 阈值指标:")
    for k, v in youden_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n输出文件:")
    print(f"预测结果: {pred_path}")
    print(f"指标表格: {metrics_path}")
    print(f"指标 JSON: {metrics_json_path}")
    print(f"ROC 曲线: {roc_path}")
    print(f"混淆矩阵 threshold=0.5: {cm_default_path}")
    print(f"混淆矩阵 Youden threshold: {cm_youden_path}")


if __name__ == "__main__":
    main()