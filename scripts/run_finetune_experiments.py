import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

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
        description="Run fine-tuning strategy experiments for ISIC classification"
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

    # 模型与训练设置
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # 微调学习率
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-5,
        help="backbone 微调学习率",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=1e-4,
        help="分类头学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减",
    )

    # 解冻最后 N 个参数层
    parser.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=10,
        help="解冻 backbone 最后 N 个参数层",
    )

    # 选择实验
    parser.add_argument(
        "--experiment_set",
        type=str,
        default="all",
        choices=["all", "last10", "full"],
        help="选择运行全部微调实验、只运行解冻最后10层、只运行全模型微调",
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
    创建 GradScaler，兼容新旧 PyTorch 写法。
    """
    try:
        return GradScaler(device.type, enabled=use_amp)
    except TypeError:
        return GradScaler(enabled=use_amp)


def is_head_parameter(model_name: str, param_name: str):
    """
    判断一个参数是否属于分类头。

    ResNet:
        fc.weight / fc.bias

    EfficientNet:
        classifier.xxx
    """
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return param_name.startswith("fc.")

    if model_name.startswith("efficientnet"):
        return param_name.startswith("classifier.")

    return False


def set_finetune_strategy(
    model: nn.Module,
    model_name: str,
    strategy: str,
    unfreeze_last_n: int = 10,
) -> List[str]:
    """
    设置微调策略。

    strategy:
    - unfreeze_last10:
        冻结大部分 backbone，只解冻 backbone 最后 N 个参数层和分类头。

    - full_finetune:
        解冻整个模型。

    返回：
    - trainable_names: 可训练参数名列表，用于保存和检查。
    """
    strategy = strategy.lower()

    # 先全部冻结
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []

    if strategy == "full_finetune":
        for name, param in model.named_parameters():
            param.requires_grad = True
            trainable_names.append(name)

        return trainable_names

    if strategy == "unfreeze_last10":
        # 1. 分类头永远解冻
        for name, param in model.named_parameters():
            if is_head_parameter(model_name, name):
                param.requires_grad = True
                trainable_names.append(name)

        # 2. 从 backbone 中找出非分类头参数
        backbone_params: List[Tuple[str, torch.nn.Parameter]] = []

        for name, param in model.named_parameters():
            if not is_head_parameter(model_name, name):
                backbone_params.append((name, param))

        # 3. 解冻 backbone 最后 N 个参数层
        last_params = backbone_params[-unfreeze_last_n:]

        for name, param in last_params:
            param.requires_grad = True
            trainable_names.append(name)

        return trainable_names

    raise ValueError(f"不支持的微调策略: {strategy}")


def build_param_groups(
    model: nn.Module,
    model_name: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
):
    """
    构建差异化学习率参数组。

    分类头使用 head_lr；
    backbone 使用 backbone_lr。
    """
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_head_parameter(model_name, name):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            }
        )

    if len(head_params) > 0:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        )

    return param_groups


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

    return running_loss / total_samples


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
    构建工序3实验列表。
    """
    experiment_set = experiment_set.lower()

    if experiment_set == "last10":
        return [
            {
                "name": "unfreeze_last10",
                "display_name": "Unfreeze Last 10 Layers",
                "strategy": "unfreeze_last10",
            }
        ]

    if experiment_set == "full":
        return [
            {
                "name": "full_finetune",
                "display_name": "Full Fine-tuning",
                "strategy": "full_finetune",
            }
        ]

    if experiment_set == "all":
        return [
            {
                "name": "unfreeze_last10",
                "display_name": "Unfreeze Last 10 Layers",
                "strategy": "unfreeze_last10",
            },
            {
                "name": "full_finetune",
                "display_name": "Full Fine-tuning",
                "strategy": "full_finetune",
            },
        ]

    raise ValueError(f"不支持的 experiment_set: {experiment_set}")


def plot_finetune_auc_compare(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str,
):
    """
    绘制微调策略 AUC 对比曲线。
    """
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(9, 5))

    for name, history in histories.items():
        epochs = history["epoch"]
        aucs = history["valid_auc"]
        plt.plot(epochs, aucs, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.title("Fine-tuning Strategy Comparison on Validation AUC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_trainable_params(trainable_names: List[str], save_path: str):
    """
    保存本组实验实际参与训练的参数名。
    """
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, "w", encoding="utf-8") as f:
        for name in trainable_names:
            f.write(name + "\n")


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
    运行单组微调实验。
    """
    set_seed(args.seed)

    exp_name = exp_config["name"]
    strategy = exp_config["strategy"]

    print("\n" + "=" * 80)
    print(f"开始微调实验: {exp_name}")
    print(f"微调策略: {strategy}")
    print("=" * 80)

    exp_checkpoint_dir = os.path.join(checkpoint_base_dir, exp_name)
    exp_table_dir = os.path.join(table_base_dir, exp_name)

    ensure_dir(exp_checkpoint_dir)
    ensure_dir(exp_table_dir)

    # 构建模型
    # 注意：这里 freeze=False，先构建完整可训练模型，再由 set_finetune_strategy 精确控制冻结/解冻。
    model = build_model(
        model_name=args.model,
        pretrained=True,
        freeze=False,
        dropout=0.0,
    )
    model = model.to(device)

    trainable_names = set_finetune_strategy(
        model=model,
        model_name=args.model,
        strategy=strategy,
        unfreeze_last_n=args.unfreeze_last_n,
    )

    print(f"模型: {args.model}")
    print(f"总参数量: {count_total_parameters(model):,}")
    print(f"可训练参数量: {count_trainable_parameters(model):,}")
    print("可训练参数名:")
    for name in trainable_names:
        print(f"  - {name}")

    trainable_params_path = os.path.join(exp_table_dir, "trainable_params.txt")
    save_trainable_params(trainable_names, trainable_params_path)

    # 保存实验配置
    config_path = os.path.join(exp_table_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_config": exp_config,
                "args": vars(args),
                "trainable_params": trainable_names,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    # 差异化学习率
    param_groups = build_param_groups(
        model=model,
        model_name=args.model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = create_grad_scaler(device, use_amp)

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_auc": [],
        "backbone_lr": [],
        "head_lr": [],
    }

    best_auc = -1.0
    best_epoch = -1

    best_model_path = os.path.join(exp_checkpoint_dir, "best_model.pth")
    last_model_path = os.path.join(exp_checkpoint_dir, "last_model.pth")
    metrics_path = os.path.join(exp_table_dir, "metrics.csv")

    for epoch in range(1, args.epochs + 1):
        current_backbone_lr = None
        current_head_lr = None

        for group in optimizer.param_groups:
            if group.get("name") == "backbone":
                current_backbone_lr = group["lr"]
            elif group.get("name") == "head":
                current_head_lr = group["lr"]

        if current_backbone_lr is None:
            current_backbone_lr = 0.0
        if current_head_lr is None:
            current_head_lr = 0.0

        print("-" * 60)
        print(
            f"[{exp_name}] Epoch [{epoch}/{args.epochs}] | "
            f"backbone_lr = {current_backbone_lr:.6e} | "
            f"head_lr = {current_head_lr:.6e}"
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
        history["backbone_lr"].append(current_backbone_lr)
        history["head_lr"].append(current_head_lr)

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
                    "trainable_params": trainable_names,
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
                "trainable_params": trainable_names,
            },
            last_model_path,
        )

        save_metrics_csv(history, metrics_path)

    result = {
        "name": exp_name,
        "display_name": exp_config["display_name"],
        "strategy": strategy,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "final_auc": history["valid_auc"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "trainable_params_count": count_trainable_parameters(model),
        "best_model_path": best_model_path,
        "metrics_path": metrics_path,
        "trainable_params_path": trainable_params_path,
        "history": history,
    }

    print("\n微调实验完成:")
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
    run_name = f"finetune_{args.model}_{timestamp}"

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

    print("\n本次将运行以下微调实验:")
    for exp in experiment_plan:
        print(f"- {exp['name']}: strategy={exp['strategy']}")

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
                "name": r["name"],
                "display_name": r["display_name"],
                "strategy": r["strategy"],
                "best_auc": r["best_auc"],
                "best_epoch": r["best_epoch"],
                "final_auc": r["final_auc"],
                "final_valid_loss": r["final_valid_loss"],
                "trainable_params_count": r["trainable_params_count"],
                "best_model_path": r["best_model_path"],
                "metrics_path": r["metrics_path"],
                "trainable_params_path": r["trainable_params_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(table_base_dir, "finetune_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n微调实验汇总结果:")
    print(
        summary_df[
            [
                "name",
                "strategy",
                "best_auc",
                "best_epoch",
                "trainable_params_count",
            ]
        ]
    )
    print(f"\n微调汇总表格已保存: {summary_path}")

    # 绘制微调策略 AUC 对比曲线
    histories = {}

    for r in all_results:
        histories[r["display_name"]] = r["history"]

    if len(histories) > 0:
        fig_path = os.path.join(
            figure_base_dir,
            "finetune_auc_compare.png",
        )
        plot_finetune_auc_compare(
            histories=histories,
            save_path=fig_path,
        )
        print(f"微调策略 AUC 对比图已保存: {fig_path}")

    print("\n全部工序3微调实验完成。")
    print(f"结果表格目录: {table_base_dir}")
    print(f"结果图目录: {figure_base_dir}")
    print(f"模型权重目录: {checkpoint_base_dir}")


if __name__ == "__main__":
    main()