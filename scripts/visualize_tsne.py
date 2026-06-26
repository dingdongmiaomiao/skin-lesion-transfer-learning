import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 允许从 scripts/ 目录导入 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_model
from src.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE feature visualization for final skin lesion model"
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
        "--max_samples",
        type=int,
        default=2500,
        help="整体 t-SNE 图最多抽样多少个验证集样本",
    )
    parser.add_argument(
        "--balanced_per_class",
        type=int,
        default=100,
        help="平衡 t-SNE 图中每个类别最多抽样多少个样本",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="t-SNE 前先用 PCA 降到多少维，加快计算并降低噪声",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity 参数",
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


class ISICFeatureDataset(Dataset):
    """
    用于特征提取的验证集 Dataset。
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
        image = self.transform(image)

        return image, label, image_name


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


class FeatureExtractor(torch.nn.Module):
    """
    从模型中提取分类头前一层特征。

    ResNet-18:
        输出 avgpool 后 flatten 的 512 维特征。

    EfficientNet-B0:
        输出 classifier 前的 pooled feature。
    """

    def __init__(self, model, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name.lower()

    def forward(self, x):
        if self.model_name.startswith("resnet"):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)

            logits = self.model.fc(features)

            return features, logits

        if self.model_name.startswith("efficientnet"):
            x = self.model.features(x)
            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)
            logits = self.model.classifier(features)

            return features, logits

        raise ValueError(f"暂不支持该模型的特征提取: {self.model_name}")


@torch.no_grad()
def extract_features(feature_extractor, dataloader, device):
    feature_extractor.eval()

    all_features = []
    all_labels = []
    all_probs = []
    all_image_names = []

    progress_bar = tqdm(dataloader, desc="Extract features")

    for images, labels, image_names in progress_bar:
        images = images.to(device, non_blocking=True)

        features, logits = feature_extractor(images)
        probs = torch.sigmoid(logits)

        all_features.append(features.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy().reshape(-1).tolist())
        all_labels.extend(labels.numpy().reshape(-1).tolist())
        all_image_names.extend(list(image_names))

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels).astype(int)
    probs = np.array(all_probs).astype(float)

    return features, labels, probs, all_image_names


def run_tsne(features, seed: int, pca_dim: int, perplexity: float):
    """
    先 PCA，再 t-SNE。
    """
    n_samples = features.shape[0]

    if n_samples <= 2:
        raise ValueError("t-SNE 至少需要多于 2 个样本")

    actual_pca_dim = min(pca_dim, features.shape[1], n_samples - 1)

    if actual_pca_dim >= 2:
        pca = PCA(n_components=actual_pca_dim, random_state=seed)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features

    actual_perplexity = min(perplexity, max(2, (n_samples - 1) / 3))

    print(f"t-SNE 样本数: {n_samples}")
    print(f"PCA dim: {actual_pca_dim}")
    print(f"t-SNE perplexity: {actual_perplexity:.2f}")

    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )

    embeddings = tsne.fit_transform(features_reduced)

    return embeddings


def plot_tsne(embedding_df, save_path: str, title: str):
    """
    绘制 t-SNE 散点图。
    """
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(7, 6))

    benign_df = embedding_df[embedding_df["target"] == 0]
    malignant_df = embedding_df[embedding_df["target"] == 1]

    plt.scatter(
        benign_df["tsne_1"],
        benign_df["tsne_2"],
        s=12,
        alpha=0.55,
        label="Benign (0)",
    )

    plt.scatter(
        malignant_df["tsne_1"],
        malignant_df["tsne_2"],
        s=28,
        alpha=0.9,
        label="Malignant (1)",
        marker="x",
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def sample_for_overall_tsne(df: pd.DataFrame, max_samples: int, seed: int):
    """
    用于整体 t-SNE 图的抽样。

    验证集良性样本远多于恶性样本，因此这里尽量保留所有恶性样本，
    再从良性样本中抽样，避免恶性点完全被淹没。
    """
    if len(df) <= max_samples:
        return df.copy().reset_index(drop=True)

    malignant_df = df[df["target"] == 1]
    benign_df = df[df["target"] == 0]

    keep_malignant = malignant_df

    benign_n = max_samples - len(keep_malignant)

    if benign_n <= 0:
        sampled_df = malignant_df.sample(
            n=max_samples,
            random_state=seed,
        )
    else:
        benign_n = min(benign_n, len(benign_df))
        sampled_benign = benign_df.sample(
            n=benign_n,
            random_state=seed,
        )

        sampled_df = pd.concat(
            [sampled_benign, keep_malignant],
            axis=0,
        )

    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sampled_df


def sample_balanced(df: pd.DataFrame, per_class: int, seed: int):
    """
    良性和恶性平衡抽样。
    """
    benign_df = df[df["target"] == 0]
    malignant_df = df[df["target"] == 1]

    n = min(len(benign_df), len(malignant_df), per_class)

    sampled_benign = benign_df.sample(n=n, random_state=seed)
    sampled_malignant = malignant_df.sample(n=n, random_state=seed)

    sampled_df = pd.concat(
        [sampled_benign, sampled_malignant],
        axis=0,
    )

    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sampled_df


def run_one_tsne_case(
    case_name: str,
    case_df: pd.DataFrame,
    image_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device,
    feature_extractor,
    seed: int,
    pca_dim: int,
    perplexity: float,
    table_dir: str,
    figure_dir: str,
    title: str,
):
    """
    运行一组 t-SNE 可视化。
    """
    print("\n" + "=" * 70)
    print(f"开始 t-SNE: {case_name}")
    print(f"样本数: {len(case_df)}")
    print("标签分布:")
    print(case_df["target"].value_counts())
    print("=" * 70)

    dataset = ISICFeatureDataset(
        dataframe=case_df,
        image_dir=image_dir,
        img_size=img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    features, labels, probs, image_names = extract_features(
        feature_extractor=feature_extractor,
        dataloader=loader,
        device=device,
    )

    embeddings = run_tsne(
        features=features,
        seed=seed,
        pca_dim=pca_dim,
        perplexity=perplexity,
    )

    embedding_df = pd.DataFrame(
        {
            "image_name": image_names,
            "target": labels,
            "prob_malignant": probs,
            "tsne_1": embeddings[:, 0],
            "tsne_2": embeddings[:, 1],
        }
    )

    table_path = os.path.join(table_dir, f"{case_name}_tsne_embeddings.csv")
    embedding_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    figure_path = os.path.join(figure_dir, f"{case_name}.png")
    plot_tsne(
        embedding_df=embedding_df,
        save_path=figure_path,
        title=title,
    )

    print(f"t-SNE 表格已保存: {table_path}")
    print(f"t-SNE 图像已保存: {figure_path}")

    return table_path, figure_path


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
    run_name = f"tsne_{args.model}_{timestamp}"

    table_dir = os.path.join(args.output_dir, "tables", run_name)
    figure_dir = os.path.join(args.output_dir, "figures", run_name)

    ensure_dir(table_dir)
    ensure_dir(figure_dir)

    # 读取数据，并使用和训练一致的验证集划分
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

    print(f"验证集总样本数: {len(valid_df)}")
    print("验证集标签分布:")
    print(valid_df["target"].value_counts())

    # 加载最终模型
    model, model_name = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model,
        device=device,
    )

    feature_extractor = FeatureExtractor(
        model=model,
        model_name=model_name,
    )

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # 1. 整体抽样 t-SNE
    sampled_df = sample_for_overall_tsne(
        df=valid_df,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    run_one_tsne_case(
        case_name="tsne_features_sampled",
        case_df=sampled_df,
        image_dir=args.image_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        feature_extractor=feature_extractor,
        seed=args.seed,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        table_dir=table_dir,
        figure_dir=figure_dir,
        title="t-SNE Visualization of Extracted Features (Sampled Validation Set)",
    )

    # 2. 平衡抽样 t-SNE
    balanced_df = sample_balanced(
        df=valid_df,
        per_class=args.balanced_per_class,
        seed=args.seed,
    )

    run_one_tsne_case(
        case_name="tsne_features_balanced",
        case_df=balanced_df,
        image_dir=args.image_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        feature_extractor=feature_extractor,
        seed=args.seed,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        table_dir=table_dir,
        figure_dir=figure_dir,
        title="t-SNE Visualization of Extracted Features (Class-balanced Samples)",
    )

    print("\n全部 t-SNE 可视化完成。")
    print(f"结果表格目录: {table_dir}")
    print(f"结果图目录: {figure_dir}")


if __name__ == "__main__":
    main()