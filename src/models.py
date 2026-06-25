import torch
import torch.nn as nn
from torchvision import models


def freeze_backbone(model: nn.Module):
    """
    冻结模型主体参数。
    baseline 阶段只训练分类头。
    """
    for param in model.parameters():
        param.requires_grad = False


def build_model(
    model_name: str = "resnet18",
    pretrained: bool = True,
    freeze: bool = True,
    dropout: float = 0.0,
):
    """
    构建二分类模型。

    支持：
    - resnet18
    - efficientnet_b0

    输出：
    - 单个 logit，不加 sigmoid
    - 训练时使用 BCEWithLogitsLoss
    """

    model_name = model_name.lower()

    if model_name == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.resnet18(weights=weights)

        if freeze:
            freeze_backbone(model)

        in_features = model.fc.in_features

        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )
        else:
            model.fc = nn.Linear(in_features, 1)

    elif model_name == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.efficientnet_b0(weights=weights)

        if freeze:
            freeze_backbone(model)

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout if dropout > 0 else 0.2),
            nn.Linear(in_features, 1),
        )

    else:
        raise ValueError(
            f"不支持的模型: {model_name}，目前支持 resnet18 / efficientnet_b0"
        )

    return model


def count_trainable_parameters(model: nn.Module):
    """
    统计可训练参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module):
    """
    统计模型总参数数量。
    """
    return sum(p.numel() for p in model.parameters())