import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# 允许从 scripts/ 目录导入 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_model
from src.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for final model"
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
        help="分类阈值，默认 0.5",
    )
    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=4,
        help="每类样本生成多少张 Grad-CAM",
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


class ISICGradCAMDataset(Dataset):
    """
    用于 Grad-CAM 的验证集 Dataset。

    返回：
    - model_input: 标准化后的输入
    - label: 标签
    - image_name: 图片名
    """

    def __init__(self, dataframe, image_dir, img_size=224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_name = str(row["image_name"])
        label = int(row["target"])

        image_path = os.path.join(self.image_dir, image_name + ".jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}")

        image = Image.open(image_path).convert("RGB")
        model_input = self.transform(image)

        return model_input, label, image_name


def load_model(checkpoint_path: str, model_name: str, device):
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

    print(f"已加载模型: {model_name_from_ckpt}")
    print(f"checkpoint epoch: {checkpoint.get('epoch', None)}")
    print(f"checkpoint best_auc: {checkpoint.get('best_auc', None)}")

    return model, model_name_from_ckpt


@torch.no_grad()
def predict_validation_set(model, dataloader, device):
    all_records = []

    progress_bar = tqdm(dataloader, desc="Predict validation set")

    for images, labels, image_names in progress_bar:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        labels = labels.numpy().reshape(-1)

        for image_name, label, prob in zip(image_names, labels, probs):
            all_records.append(
                {
                    "image_name": image_name,
                    "target": int(label),
                    "prob_malignant": float(prob),
                }
            )

    return pd.DataFrame(all_records)


def get_target_layer(model, model_name: str):
    """
    选择 Grad-CAM 的目标层。

    ResNet-18:
        使用 layer4 最后一个 BasicBlock 的 conv2。

    EfficientNet-B0:
        使用 features 最后一层。
    """
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return model.layer4[-1].conv2

    if model_name.startswith("efficientnet"):
        return model.features[-1]

    raise ValueError(f"暂不支持该模型的 Grad-CAM target layer: {model_name}")


class GradCAM:
    """
    简单 Grad-CAM 实现。

    对二分类单 logit 模型：
    - malignant score 使用 logit
    - benign score 使用 -logit
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, target_class: int = 1):
        """
        input_tensor: shape [1, C, H, W]
        target_class:
            1 表示解释“恶性”方向；
            0 表示解释“良性”方向。
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if target_class == 1:
            score = logits[:, 0].sum()
        else:
            score = (-logits[:, 0]).sum()

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def load_original_image(image_dir: str, image_name: str, img_size: int):
    """
    读取原图，并进行与验证集一致的 Resize + CenterCrop。
    返回 RGB uint8 图像。
    """
    image_path = os.path.join(image_dir, image_name + ".jpg")
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
        ]
    )

    image = transform(image)
    image = np.array(image)

    return image


def overlay_cam_on_image(rgb_image, cam, alpha=0.45):
    """
    将 CAM 叠加到 RGB 图像上。
    """
    h, w, _ = rgb_image.shape

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8((1 - alpha) * rgb_image + alpha * heatmap)

    return overlay, heatmap


def select_cases(pred_df: pd.DataFrame, threshold: float, samples_per_group: int):
    """
    选择 TP / FN / FP / TN 样本。

    排序逻辑：
    - TP: 恶性且预测恶性，优先选 prob 高的；
    - FN: 恶性但预测良性，优先选 prob 低的；
    - FP: 良性但预测恶性，优先选 prob 高的；
    - TN: 良性且预测良性，优先选 prob 低的。
    """
    df = pred_df.copy()
    df["pred"] = (df["prob_malignant"] >= threshold).astype(int)

    tp = df[(df["target"] == 1) & (df["pred"] == 1)].sort_values(
        "prob_malignant", ascending=False
    )

    fn = df[(df["target"] == 1) & (df["pred"] == 0)].sort_values(
        "prob_malignant", ascending=True
    )

    fp = df[(df["target"] == 0) & (df["pred"] == 1)].sort_values(
        "prob_malignant", ascending=False
    )

    tn = df[(df["target"] == 0) & (df["pred"] == 0)].sort_values(
        "prob_malignant", ascending=True
    )

    cases = {
        "TP_malignant_correct": tp.head(samples_per_group),
        "FN_malignant_missed": fn.head(samples_per_group),
        "FP_benign_false_alarm": fp.head(samples_per_group),
        "TN_benign_correct": tn.head(samples_per_group),
    }

    return cases


def preprocess_single_image(image_dir: str, image_name: str, img_size: int, device):
    """
    读取单张图片并转成模型输入。
    """
    image_path = os.path.join(image_dir, image_name + ".jpg")
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = transform(image).unsqueeze(0).to(device)

    return tensor


def save_single_gradcam_figure(
    original,
    heatmap,
    overlay,
    save_path: str,
    title: str,
):
    """
    保存单张 Grad-CAM 三联图。
    """
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_group_summary(group_images: List[Dict], save_path: str, group_name: str):
    """
    保存某一组样本的汇总图。

    每个样本一行：
    Original | Heatmap | Overlay
    """
    if len(group_images) == 0:
        return

    ensure_dir(os.path.dirname(save_path))

    n = len(group_images)

    plt.figure(figsize=(12, 4 * n))

    for i, item in enumerate(group_images):
        row_base = i * 3

        plt.subplot(n, 3, row_base + 1)
        plt.imshow(item["original"])
        plt.title(
            f"{item['image_name']}\n"
            f"True={item['target']}, Prob={item['prob_malignant']:.4f}"
        )
        plt.axis("off")

        plt.subplot(n, 3, row_base + 2)
        plt.imshow(item["heatmap"])
        plt.title("Heatmap")
        plt.axis("off")

        plt.subplot(n, 3, row_base + 3)
        plt.imshow(item["overlay"])
        plt.title("Overlay")
        plt.axis("off")

    plt.suptitle(group_name, fontsize=14)
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
    run_name = f"gradcam_{args.model}_{timestamp}"

    figure_dir = os.path.join(args.output_dir, "figures", run_name)
    table_dir = os.path.join(args.output_dir, "tables", run_name)

    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    # 读取数据并使用与训练一致的验证集划分
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

    valid_dataset = ISICGradCAMDataset(
        dataframe=valid_df,
        image_dir=args.image_dir,
        img_size=args.img_size,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 加载模型
    model, model_name = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model,
        device=device,
    )

    # 预测验证集
    pred_df = predict_validation_set(
        model=model,
        dataloader=valid_loader,
        device=device,
    )

    pred_df["pred"] = (pred_df["prob_malignant"] >= args.threshold).astype(int)

    pred_path = os.path.join(table_dir, "gradcam_selected_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    cases = select_cases(
        pred_df=pred_df,
        threshold=args.threshold,
        samples_per_group=args.samples_per_group,
    )

    # 保存被选中的样本列表
    selected_rows = []

    for group_name, group_df in cases.items():
        for _, row in group_df.iterrows():
            selected_rows.append(
                {
                    "group": group_name,
                    "image_name": row["image_name"],
                    "target": int(row["target"]),
                    "pred": int(row["pred"]),
                    "prob_malignant": float(row["prob_malignant"]),
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    selected_path = os.path.join(table_dir, "gradcam_selected_cases.csv")
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print("\n各组样本数量:")
    for group_name, group_df in cases.items():
        print(f"{group_name}: {len(group_df)}")

    # Grad-CAM
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    for group_name, group_df in cases.items():
        group_output_dir = os.path.join(figure_dir, group_name)
        ensure_dir(group_output_dir)

        group_images = []

        print(f"\n生成 {group_name} Grad-CAM...")

        for _, row in group_df.iterrows():
            image_name = row["image_name"]
            target = int(row["target"])
            pred = int(row["pred"])
            prob = float(row["prob_malignant"])

            # 通常我们解释“恶性”方向，因为项目关注模型如何判断恶性风险
            target_class_for_cam = 1

            input_tensor = preprocess_single_image(
                image_dir=args.image_dir,
                image_name=image_name,
                img_size=args.img_size,
                device=device,
            )

            cam = gradcam.generate(
                input_tensor=input_tensor,
                target_class=target_class_for_cam,
            )

            original = load_original_image(
                image_dir=args.image_dir,
                image_name=image_name,
                img_size=args.img_size,
            )

            overlay, heatmap = overlay_cam_on_image(
                rgb_image=original,
                cam=cam,
                alpha=0.45,
            )

            title = (
                f"{group_name} | {image_name} | "
                f"True={target}, Pred={pred}, Prob={prob:.4f}"
            )

            save_path = os.path.join(
                group_output_dir,
                f"{image_name}_gradcam.png",
            )

            save_single_gradcam_figure(
                original=original,
                heatmap=heatmap,
                overlay=overlay,
                save_path=save_path,
                title=title,
            )

            group_images.append(
                {
                    "image_name": image_name,
                    "target": target,
                    "pred": pred,
                    "prob_malignant": prob,
                    "original": original,
                    "heatmap": heatmap,
                    "overlay": overlay,
                }
            )

        summary_path = os.path.join(
            figure_dir,
            f"{group_name}_summary.png",
        )

        save_group_summary(
            group_images=group_images,
            save_path=summary_path,
            group_name=group_name,
        )

    gradcam.remove_hooks()

    print("\nGrad-CAM 生成完成。")
    print(f"选择样本表: {selected_path}")
    print(f"预测结果表: {pred_path}")
    print(f"Grad-CAM 图像目录: {figure_dir}")


if __name__ == "__main__":
    main()