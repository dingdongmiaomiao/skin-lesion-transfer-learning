import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 允许从 scripts/ 目录导入 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import ISICDataset, get_train_transforms, get_valid_transforms
from src.models import build_model, count_total_parameters, count_trainable_parameters
from src.utils import ensure_dir, get_device, save_metrics_csv, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tuning experiments for optimizer and scheduler comparison"
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

    # 实验设置
    parser.add_argument(
        "--experiment_set",
        type=str,
        default="all",
        choices=["all", "optimizer", "scheduler"],
        help="选择运行全部实验、优化器对比实验或学习率调度器对比实验",
    )
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # 工序2默认冻结 backbone，只比较优化器和学习率调度器
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=True,
        help="是否冻结 backbone。工序2默认冻结，只训练分类头。",
    )

    # 调试用：只抽取部分数据跑流程，正式实验保持 1.0
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="抽样比例。调试可用 0.05，正式实验用 1.0。",
    )

    # 输出路径
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录",
    )

    # AMP 设置
    parser.add_argument(
        "--amp",
        type=int,
        default=1,
        help="是否使用混合精度训练。1 表示使用，0 表示关闭。仅 CUDA 下生效。",
    )

    return parser.parse_args()


def make_abs_path(path: str):
    """
    将相对路径转换为项目根目录下的绝对路径。
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def create_grad_scaler(device, use_amp: bool):
    """
    创建 GradScaler。

    新版 PyTorch 推荐：
    GradScaler("cuda", enabled=True)

    为了兼容旧版本 PyTorch，这里做一次 try-except。
    """
    try:
        return GradScaler(device.type, enabled=use_amp)
    except TypeError:
        return GradScaler(enabled=use_amp)


def create_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
):
    """
    创建优化器。

    工序2为了控制变量：
    - Adam / AdamW / SGD 使用相同初始学习率；
    - SGD 使用 momentum=0.9 和 nesterov=True。
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )

    raise ValueError(f"不支持的优化器: {optimizer_name}")


def create_scheduler(
    scheduler_name: str,
    optimizer,
    epochs: int,
    lr: float,
):
    """
    创建学习率调度器。

    fixed:
        不使用调度器，学习率固定。

    cosine:
        使用 CosineAnnealingLR。

    plateau:
        使用 ReduceLROnPlateau，根据验证 AUC 调整学习率。
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "fixed":
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=lr * 0.01,
        )

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

    raise ValueError(f"不支持的学习率调度器: {scheduler_name}")


def step_scheduler(scheduler, scheduler_name: str, valid_auc: float):
    """
    不同 scheduler 的 step 方式不同。
    """
    if scheduler is None:
        return

    if scheduler_name == "plateau":
        scheduler.step(valid_auc)
    else:
        scheduler.step()


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
):
    """
    训练一个 epoch。
    """
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
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
    """
    验证一个 epoch，返回验证 Loss 和验证 AUC。
    """
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

    valid_loss = running_loss / total_samples

    try:
        valid_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        valid_auc = 0.0

    return valid_loss, valid_auc


def build_experiment_plan(experiment_set: str):
    """
    构建工序2实验列表。

    all 模式实际运行 5 组：
    1. Adam + ReduceLROnPlateau
    2. AdamW + ReduceLROnPlateau
    3. SGD + momentum + ReduceLROnPlateau
    4. Adam + Fixed LR
    5. Adam + CosineAnnealingLR

    其中 Adam + ReduceLROnPlateau 同时作为：
    - 优化器对比中的 Adam 组；
    - 调度器对比中的 ReduceLROnPlateau 组。
    """
    experiment_set = experiment_set.lower()

    if experiment_set == "optimizer":
        return [
            {
                "experiment_type": "optimizer",
                "name": "optimizer_adam_plateau",
                "display_name": "Adam",
                "optimizer": "adam",
                "scheduler": "plateau",
            },
            {
                "experiment_type": "optimizer",
                "name": "optimizer_adamw_plateau",
                "display_name": "AdamW",
                "optimizer": "adamw",
                "scheduler": "plateau",
            },
            {
                "experiment_type": "optimizer",
                "name": "optimizer_sgd_plateau",
                "display_name": "SGD+momentum",
                "optimizer": "sgd",
                "scheduler": "plateau",
            },
        ]

    if experiment_set == "scheduler":
        return [
            {
                "experiment_type": "scheduler",
                "name": "scheduler_fixed_adam",
                "display_name": "Fixed LR",
                "optimizer": "adam",
                "scheduler": "fixed",
            },
            {
                "experiment_type": "scheduler",
                "name": "scheduler_cosine_adam",
                "display_name": "CosineAnnealingLR",
                "optimizer": "adam",
                "scheduler": "cosine",
            },
            {
                "experiment_type": "scheduler",
                "name": "scheduler_plateau_adam",
                "display_name": "ReduceLROnPlateau",
                "optimizer": "adam",
                "scheduler": "plateau",
            },
        ]

    if experiment_set == "all":
        return [
            {
                "experiment_type": "optimizer_and_scheduler",
                "name": "adam_plateau",
                "display_name": "Adam / ReduceLROnPlateau",
                "optimizer": "adam",
                "scheduler": "plateau",
            },
            {
                "experiment_type": "optimizer",
                "name": "adamw_plateau",
                "display_name": "AdamW",
                "optimizer": "adamw",
                "scheduler": "plateau",
            },
            {
                "experiment_type": "optimizer",
                "name": "sgd_plateau",
                "display_name": "SGD+momentum",
                "optimizer": "sgd",
                "scheduler": "plateau",
            },
            {
                "experiment_type": "scheduler",
                "name": "adam_fixed",
                "display_name": "Fixed LR",
                "optimizer": "adam",
                "scheduler": "fixed",
            },
            {
                "experiment_type": "scheduler",
                "name": "adam_cosine",
                "display_name": "CosineAnnealingLR",
                "optimizer": "adam",
                "scheduler": "cosine",
            },
        ]

    raise ValueError(f"不支持的 experiment_set: {experiment_set}")


def plot_auc_compare(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str,
):
    """
    绘制多组实验的验证 AUC 对比曲线。
    """
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(9, 5))

    for name, history in histories.items():
        epochs = history["epoch"]
        aucs = history["valid_auc"]
        plt.plot(epochs, aucs, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_single_experiment(
    exp_config: Dict,
    args,
    train_loader,
    valid_loader,
    criterion,
    device,
    use_amp: bool,
    checkpoint_base_dir: str,
    table_base_dir: str,
):
    """
    运行单组实验。
    """
    set_seed(args.seed)

    exp_name = exp_config["name"]
    optimizer_name = exp_config["optimizer"]
    scheduler_name = exp_config["scheduler"]

    print("\n" + "=" * 80)
    print(f"开始实验: {exp_name}")
    print(f"实验类型: {exp_config['experiment_type']}")
    print(f"优化器: {optimizer_name}")
    print(f"学习率调度器: {scheduler_name}")
    print("=" * 80)

    exp_checkpoint_dir = os.path.join(checkpoint_base_dir, exp_name)
    exp_table_dir = os.path.join(table_base_dir, exp_name)

    ensure_dir(exp_checkpoint_dir)
    ensure_dir(exp_table_dir)

    # 保存实验配置
    config_path = os.path.join(exp_table_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_config": exp_config,
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    # 每组实验都重新初始化模型，保证公平
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

    optimizer = create_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = create_scheduler(
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        epochs=args.epochs,
        lr=args.lr,
    )

    scaler = create_grad_scaler(device, use_amp)

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_auc": [],
        "lr": [],
    }

    best_auc = -1.0
    best_epoch = -1

    best_model_path = os.path.join(exp_checkpoint_dir, "best_model.pth")
    last_model_path = os.path.join(exp_checkpoint_dir, "last_model.pth")
    metrics_path = os.path.join(exp_table_dir, "metrics.csv")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        print("-" * 60)
        print(
            f"[{exp_name}] Epoch [{epoch}/{args.epochs}] | "
            f"lr = {current_lr:.6e}"
        )

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

        step_scheduler(scheduler, scheduler_name, valid_auc)

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

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "experiment_config": exp_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "args": vars(args),
                },
                best_model_path,
            )

            print(
                f"保存最佳模型: {best_model_path} | "
                f"best_auc = {best_auc:.5f}"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_name": args.model,
                "experiment_config": exp_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "args": vars(args),
            },
            last_model_path,
        )

        # 每个 epoch 保存一次日志，避免中途停止后没有结果
        save_metrics_csv(history, metrics_path)

    result = {
        "experiment_type": exp_config["experiment_type"],
        "name": exp_name,
        "display_name": exp_config["display_name"],
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "final_auc": history["valid_auc"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "best_model_path": best_model_path,
        "metrics_path": metrics_path,
        "history": history,
    }

    print("\n实验完成:")
    print(f"实验名: {exp_name}")
    print(f"Best AUC: {best_auc:.5f}")
    print(f"Best Epoch: {best_epoch}")

    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    args.csv_path = make_abs_path(args.csv_path)
    args.image_dir = make_abs_path(args.image_dir)
    args.output_dir = make_abs_path(args.output_dir)

    device = get_device()
    print(f"使用设备: {device}")

    use_amp = bool(args.amp == 1 and device.type == "cuda")
    print(f"混合精度训练 AMP: {use_amp}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tuning_{args.model}_{timestamp}"

    checkpoint_base_dir = os.path.join(args.output_dir, "checkpoints", run_name)
    table_base_dir = os.path.join(args.output_dir, "tables", run_name)
    figure_base_dir = os.path.join(args.output_dir, "figures", run_name)

    for d in [checkpoint_base_dir, table_base_dir, figure_base_dir]:
        ensure_dir(d)

    # 读取数据
    df = pd.read_csv(args.csv_path)
    df = df[["image_name", "target"]].copy()
    df["target"] = df["target"].astype(int)

    if args.sample_frac <= 0 or args.sample_frac > 1:
        raise ValueError("--sample_frac 必须在 (0, 1] 范围内")

    # 调试时可抽样，正式实验保持 1.0
    if args.sample_frac < 1.0:
        df, _ = train_test_split(
            df,
            train_size=args.sample_frac,
            stratify=df["target"],
            random_state=args.seed,
        )
        df = df.reset_index(drop=True)
        print(f"当前为抽样调试模式，sample_frac = {args.sample_frac}")

    print("数据集样本数:", len(df))
    print("标签分布:")
    print(df["target"].value_counts())

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )

    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(valid_df)}")

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

    # 类别不平衡处理
    num_pos = int((train_df["target"] == 1).sum())
    num_neg = int((train_df["target"] == 0).sum())

    if num_pos > 0:
        pos_weight_value = num_neg / num_pos
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"pos_weight: {pos_weight_value:.4f}")

    experiment_plan = build_experiment_plan(args.experiment_set)

    print("\n本次将运行以下实验:")
    for exp in experiment_plan:
        print(
            f"- {exp['name']}: "
            f"optimizer={exp['optimizer']}, scheduler={exp['scheduler']}"
        )

    all_results = []

    for exp_config in experiment_plan:
        result = run_single_experiment(
            exp_config=exp_config,
            args=args,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            checkpoint_base_dir=checkpoint_base_dir,
            table_base_dir=table_base_dir,
        )
        all_results.append(result)

    # 保存汇总表格
    summary_rows = []

    for r in all_results:
        summary_rows.append(
            {
                "experiment_type": r["experiment_type"],
                "name": r["name"],
                "display_name": r["display_name"],
                "optimizer": r["optimizer"],
                "scheduler": r["scheduler"],
                "best_auc": r["best_auc"],
                "best_epoch": r["best_epoch"],
                "final_auc": r["final_auc"],
                "final_valid_loss": r["final_valid_loss"],
                "best_model_path": r["best_model_path"],
                "metrics_path": r["metrics_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(table_base_dir, "tuning_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n汇总结果:")
    print(summary_df[["name", "optimizer", "scheduler", "best_auc", "best_epoch"]])
    print(f"\n汇总表格已保存: {summary_path}")

    # 绘制优化器对比曲线
    optimizer_histories = {}

    for r in all_results:
        if args.experiment_set == "optimizer":
            optimizer_histories[r["display_name"]] = r["history"]
        elif args.experiment_set == "all":
            if r["name"] == "adam_plateau":
                optimizer_histories["Adam"] = r["history"]
            elif r["name"] == "adamw_plateau":
                optimizer_histories["AdamW"] = r["history"]
            elif r["name"] == "sgd_plateau":
                optimizer_histories["SGD+momentum"] = r["history"]

    if len(optimizer_histories) > 0:
        optimizer_fig_path = os.path.join(
            figure_base_dir,
            "optimizer_auc_compare.png",
        )
        plot_auc_compare(
            histories=optimizer_histories,
            save_path=optimizer_fig_path,
            title="Optimizer Comparison on Validation AUC",
        )
        print(f"优化器 AUC 对比图已保存: {optimizer_fig_path}")

    # 绘制学习率调度器对比曲线
    scheduler_histories = {}

    for r in all_results:
        if args.experiment_set == "scheduler":
            scheduler_histories[r["display_name"]] = r["history"]
        elif args.experiment_set == "all":
            if r["name"] == "adam_fixed":
                scheduler_histories["Fixed LR"] = r["history"]
            elif r["name"] == "adam_cosine":
                scheduler_histories["CosineAnnealingLR"] = r["history"]
            elif r["name"] == "adam_plateau":
                scheduler_histories["ReduceLROnPlateau"] = r["history"]

    if len(scheduler_histories) > 0:
        scheduler_fig_path = os.path.join(
            figure_base_dir,
            "scheduler_auc_compare.png",
        )
        plot_auc_compare(
            histories=scheduler_histories,
            save_path=scheduler_fig_path,
            title="Learning Rate Scheduler Comparison on Validation AUC",
        )
        print(f"学习率调度器 AUC 对比图已保存: {scheduler_fig_path}")

    print("\n全部工序2实验完成。")
    print(f"结果表格目录: {table_base_dir}")
    print(f"结果图目录: {figure_base_dir}")
    print(f"模型权重目录: {checkpoint_base_dir}")


if __name__ == "__main__":
    main()