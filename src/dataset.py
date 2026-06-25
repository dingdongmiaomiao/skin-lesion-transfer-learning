import os
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ISICDataset(Dataset):
    """
    ISIC 2020 皮肤病灶二分类数据集。

    ground_truth.csv 需要包含：
    - image_name: 图片名，不含扩展名
    - target: 标签，0 表示良性，1 表示恶性
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        if "image_name" not in self.dataframe.columns:
            raise ValueError("ground_truth.csv 中必须包含 image_name 列")

        if "target" not in self.dataframe.columns:
            raise ValueError("ground_truth.csv 中必须包含 target 列")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_name = str(row["image_name"])
        label = float(row["target"])

        image_path = os.path.join(self.image_dir, image_name + ".jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_train_transforms(img_size: int = 224):
    """
    baseline 训练集数据增强。

    注意：
    这里先使用比较基础的数据增强，不加入 Mixup / CutMix。
    Mixup / CutMix 留到后续消融实验工序。
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_valid_transforms(img_size: int = 224):
    """
    验证集不做随机增强，只做 resize、center crop 和标准化。
    """
    return transforms.Compose(
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