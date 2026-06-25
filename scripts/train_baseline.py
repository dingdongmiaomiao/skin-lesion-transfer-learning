import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 允许从 scripts/ 目录导入 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import ISICDataset, get_train_transforms, get_valid_transforms
from src.models import build_model, count_total_parameters, count_trainable_parameters
from src.utils import (
    ensure_dir,
    get_device,
    plot_training_curves,
    save_metrics_csv,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline training for ISIC skin lesion classification"
    )

    # 数据路径
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
        help="训练图像文件夹路径",
    )

    # 训练设置
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # baseline 默认冻结 backbone，只训练分类头
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=True,
        help="是否冻结 backbone。baseline 默认冻结。",
    )

    # 输出路径
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录",
    )

    # 其他
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="是否使用混合精度训练。仅 CUDA 下生效。",
    )

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples

    return epoch_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Valid", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.extend(probs.detach().cpu().numpy().reshape(-1).tolist())
        all_labels.extend(labels.detach().cpu().numpy().reshape(-1).tolist())

    epoch_loss = running_loss / total_samples

    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0

    return epoch_loss, epoch_auc


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"使用设备: {device}")

    use_amp = bool(args.amp and device.type == "cuda")
    print(f"混合精度训练 AMP: {use_amp}")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"baseline_{args.model}_{timestamp}"

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", exp_name)
    log_dir = os.path.join(args.output_dir, "logs", exp_name)
    figure_dir = os.path.join(args.output_dir, "figures", exp_name)
    table_dir = os.path.join(args.output_dir, "tables", exp_name)

    for d in [checkpoint_dir, log_dir, figure_dir, table_dir]:
        ensure_dir(d)

    # 读取标注文件
    df = pd.read_csv(args.csv_path)

    # 只保留必要列，避免原始 csv 里有很多无关字段
    df = df[["image_name", "target"]].copy()
    df["target"] = df["target"].astype(int)

    print("数据集总样本数:", len(df))
    print("标签分布:")
    print(df["target"].value_counts())

    # 分层划分训练集和验证集
    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )

    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(valid_df)}")

    # Dataset
    train_dataset = ISICDataset(
        dataframe=train_df,
        image_dir=args.image_dir,
        transform=get_train_transforms(args.img_size),
    )

    valid_dataset = ISICDataset(
        dataframe=valid_df,
        image_dir=args.image_dir,
        transform=get_valid_transforms(args.img_size),
    )

    # Windows 下如果 DataLoader 报错，可以把 num_workers 改成 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 模型
    model = build_model(
        model_name=args.model,
        pretrained=True,
        freeze=args.freeze,
        dropout=0.0,
    )
    model = model.to(device)

    print(f"模型: {args.model}")
    print(f"总参数量: {count_total_parameters(model):,}")
    print(f"可训练参数量: {count_trainable_parameters(model):,}")

    # 类别不平衡处理：给恶性样本更高权重
    num_pos = int((train_df["target"] == 1).sum())
    num_neg = int((train_df["target"] == 0).sum())

    if num_pos > 0:
        pos_weight_value = num_neg / num_pos
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = GradScaler(enabled=use_amp)

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_auc": [],
        "lr": [],
    }

    best_auc = -1.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        print("=" * 60)
        print(f"Epoch [{epoch}/{args.epochs}] | lr = {current_lr:.6e}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        valid_loss, valid_auc = validate_one_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(valid_auc)

        print(
            f"Train Loss: {train_loss:.5f} | "
            f"Valid Loss: {valid_loss:.5f} | "
            f"Valid AUC: {valid_auc:.5f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_auc"].append(valid_auc)
        history["lr"].append(current_lr)

        # 保存最佳模型
        if valid_auc > best_auc:
            best_auc = valid_auc

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "args": vars(args),
                },
                best_model_path,
            )

            print(f"保存最佳模型: {best_model_path} | best_auc = {best_auc:.5f}")

        # 每轮保存 last model
        torch.save(
            {
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "args": vars(args),
            },
            last_model_path,
        )

        # 每轮都更新日志和曲线，防止中途停止后没有结果
        metrics_path = os.path.join(table_dir, "baseline_metrics.csv")
        save_metrics_csv(history, metrics_path)
        plot_training_curves(history, figure_dir)

    print("=" * 60)
    print("训练完成")
    print(f"最佳验证 AUC: {best_auc:.5f}")
    print(f"最佳模型路径: {best_model_path}")
    print(f"日志表格路径: {os.path.join(table_dir, 'baseline_metrics.csv')}")
    print(f"结果图路径: {figure_dir}")


if __name__ == "__main__":
    main()
