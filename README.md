# skin-lesion-transfer-learning
基于迁移学习的皮肤病灶良恶性识别（ISIC 2020），包含优化器/调度器对比、微调策略、消融实验、Grad-CAM 热力图和 t-SNE 可视化，PyTorch 实现。

# 基于迁移学习的皮肤病灶良恶性识别

## 目录

<details>
<summary>点击展开目录</summary>

- [1. 项目简介](#1-项目简介)
- [2. 项目特点](#2-项目特点)
- [3. 数据集说明](#3-数据集说明)
- [4. 环境配置](#4-环境配置)
  - [4.1 推荐环境](#41-推荐环境)
  - [4.2 安装依赖](#42-安装依赖)
- [5. 仓库结构](#5-仓库结构)
- [6. 实验流程](#6-实验流程)
  - [6.1 Baseline 训练](#61-baseline-训练)
  - [6.2 优化器与学习率调度器对比](#62-优化器与学习率调度器对比)
  - [6.3 微调策略对比](#63-微调策略对比)
- [7. 最终模型评估](#7-最终模型评估)
- [8. 可视化结果](#8-可视化结果)
- [9. 主要实验结果汇总](#9-主要实验结果汇总)
- [10. 团队分工](#10-团队分工)
- [11. AI 工具使用说明](#11-ai-工具使用说明)
- [12. 注意事项](#12-注意事项)
- [13. 项目展示](#13-项目展示)

</details>

## 1. 项目简介

本项目是数据科学与大数据技术课程大项目，选题为：

**基于迁移学习的皮肤病灶良恶性识别——以 ISIC 黑色素瘤数据为例**

项目使用 PyTorch 框架，基于 ImageNet 预训练卷积神经网络，对 ISIC 2020 / SIIM-ISIC Melanoma Classification 数据集中的皮肤镜图像进行二分类识别：

* `0`：良性病灶 Benign
* `1`：恶性病灶 Malignant

本项目重点不在于开发前端系统，而是完整实现一个机器学习实验流程，包括：

1. 数据读取与预处理；
2. Baseline 模型训练；
3. 优化器与学习率调度器对比；
4. 迁移学习微调策略对比；
5. 最终模型评估；
6. ROC 曲线、混淆矩阵、Grad-CAM、t-SNE 等可视化分析；
7. 实验结果总结与报告展示。

---

## 2. 项目特点

本项目具有以下特点：

* 使用 ResNet-18 作为主要模型，并加载 ImageNet 预训练权重；
* 采用迁移学习方法，比较不同微调策略对模型性能的影响；
* 针对恶性样本数量较少的问题，使用 `pos_weight` 缓解类别不平衡；
* 使用 AUC 作为主要评价指标，更适合类别不平衡的医学图像二分类任务；
* 通过 Grad-CAM 分析模型关注区域，提高模型可解释性；
* 通过 t-SNE 展示模型提取特征的分布情况；
* 保存完整训练日志、结果表格和可视化图像，便于复现实验过程。

---

## 3. 数据集说明

本项目使用 ISIC 2020 / SIIM-ISIC Melanoma Classification 数据集。

本地数据目录结构如下：

```text
data/
└── ISIC_2020/
    ├── ground_truth.csv
    └── train/
        ├── ISIC_xxxxxxx.jpg
        ├── ISIC_xxxxxxx.jpg
        └── ...
```

其中：

* `ground_truth.csv`：标签文件，至少包含以下两列：

  * `image_name`：图像名称，不含 `.jpg` 后缀；
  * `target`：标签，`0` 表示良性，`1` 表示恶性。
* `train/`：所有皮肤镜图像。

注意：由于图像数据集体积较大，本仓库不直接上传原始数据集。运行代码前，需要自行下载数据集，并按照上述目录结构放置。

---

## 4. 环境配置

### 4.1 推荐环境

```text
Python >= 3.9
PyTorch
torchvision
scikit-learn
pandas
numpy
matplotlib
Pillow
opencv-python
tqdm
```

### 4.2 安装依赖

建议先创建虚拟环境：

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，建议根据本机 CUDA 版本安装对应的 PyTorch GPU 版本。

---

## 5. 仓库结构

```text
skin-lesion-transfer-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── AI_LOG.md
├── data/
│   └── ISIC_2020/
│       ├── ground_truth.csv
│       └── train/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   └── utils.py
├── scripts/
│   ├── train_baseline.py
│   ├── run_tuning_experiments.py
│   ├── run_finetune_experiments.py
│   ├── evaluate_final_model.py
│   ├── generate_gradcam.py
│   └── visualize_tsne.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── tables/
└── report/
```

主要文件说明：

| 文件                                    | 作用                                  |
| ------------------------------------- | ----------------------------------- |
| `src/dataset.py`                      | ISIC 数据集读取与图像预处理                    |
| `src/models.py`                       | ResNet-18 / EfficientNet-B0 二分类模型构建 |
| `src/utils.py`                        | 随机种子、目录创建、指标保存、曲线绘制等工具函数            |
| `scripts/train_baseline.py`           | Baseline 训练脚本                       |
| `scripts/run_tuning_experiments.py`   | 优化器与学习率调度器对比实验                      |
| `scripts/run_finetune_experiments.py` | 微调策略对比实验                            |
| `scripts/evaluate_final_model.py`     | 最终模型评估、ROC 曲线与混淆矩阵                  |
| `scripts/generate_gradcam.py`         | Grad-CAM 可解释性分析                     |
| `scripts/visualize_tsne.py`           | t-SNE 特征可视化                         |
| `AI_LOG.md`                           | AI 工具使用与项目开发过程记录                    |

---

## 6. 实验流程

### 6.1 Baseline 训练

Baseline 使用 ImageNet 预训练 ResNet-18，冻结 backbone，仅训练最后的二分类分类头。

运行命令：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

如果 Windows 下 DataLoader 多进程报错，可使用：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 15 --batch_size 16 --img_size 224 --num_workers 0
```

Baseline 实验结果：

| 模型        | 训练策略               | Best AUC |
| --------- | ------------------ | -------: |
| ResNet-18 | 冻结 backbone，仅训练分类头 |  0.83784 |

---

### 6.2 优化器与学习率调度器对比

本阶段固定模型为 ResNet-18，冻结 backbone，仅训练分类头，比较不同优化器和学习率调度策略。

运行命令：

```bash
python scripts/run_tuning_experiments.py --experiment_set all --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

实验组如下：

| 实验组           | Optimizer      | Scheduler         | Best AUC | Best Epoch |
| ------------- | -------------- | ----------------- | -------: | ---------: |
| adam_plateau  | Adam           | ReduceLROnPlateau | 0.837860 |         15 |
| adamw_plateau | AdamW          | ReduceLROnPlateau | 0.837864 |         15 |
| sgd_plateau   | SGD + momentum | ReduceLROnPlateau | 0.822258 |         15 |
| adam_fixed    | Adam           | Fixed LR          | 0.837980 |         13 |
| adam_cosine   | Adam           | CosineAnnealingLR | 0.835348 |         13 |

实验结论：

* Adam 与 AdamW 表现非常接近，Best AUC 均约为 0.838；
* SGD + momentum 表现明显低于 Adam 系列优化器；
* Fixed LR 与 ReduceLROnPlateau 表现接近；
* 综合考虑后续微调稳定性，本项目后续采用 `AdamW + ReduceLROnPlateau` 作为默认训练策略。

---

### 6.3 微调策略对比

本阶段比较三种迁移学习策略：

1. 仅训练分类头；
2. 解冻最后 10 层；
3. 全模型微调。

运行命令：

```bash
python scripts/run_finetune_experiments.py --experiment_set all --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

实验结果：

| 微调策略      | Best AUC | Best Epoch |     可训练参数量 |
| --------- | -------: | ---------: | ---------: |
| 仅训练分类头    | 0.837864 |         15 |         较少 |
| 解冻最后 10 层 | 0.855562 |         15 |  4,853,761 |
| 全模型微调     | 0.885590 |          7 | 11,177,025 |

实验结论：

全模型微调取得最佳结果，Best AUC 达到 `0.885590`。相比仅训练分类头，全模型微调带来了约 `0.0477` 的 AUC 提升，说明 ISIC 皮肤镜图像与 ImageNet 自然图像存在一定领域差异，解冻更多网络层有助于模型学习医学图像特征。

最终模型选择：

```text
outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth
```

---

## 7. 最终模型评估

最终模型选用 full_finetune 中验证集 AUC 最优的模型。

运行命令：

```bash
python scripts/evaluate_final_model.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 4
```

### 7.1 ROC-AUC

最终模型在验证集上的 ROC-AUC 为：

```text
AUC = 0.885581
```

### 7.2 默认阈值 0.5 下的指标

| 指标                   |       数值 |
| -------------------- | -------: |
| AUC                  | 0.885581 |
| Accuracy             | 0.812406 |
| Precision            | 0.068913 |
| Recall / Sensitivity | 0.769231 |
| Specificity          | 0.813182 |
| F1                   | 0.126493 |
| TN                   |     5293 |
| FP                   |     1216 |
| FN                   |       27 |
| TP                   |       90 |

### 7.3 Youden 阈值下的指标

Youden 阈值为：

```text
0.383211
```

| 指标                   |       数值 |
| -------------------- | -------: |
| AUC                  | 0.885581 |
| Accuracy             | 0.758980 |
| Precision            | 0.060570 |
| Recall / Sensitivity | 0.871795 |
| Specificity          | 0.756952 |
| F1                   | 0.113270 |
| TN                   |     4927 |
| FP                   |     1582 |
| FN                   |       15 |
| TP                   |      102 |

结果分析：

在默认阈值 0.5 下，模型能够识别 117 个恶性样本中的 90 个，召回率为 76.92%。使用 Youden 阈值后，恶性样本召回率提升至 87.18%，漏诊数量由 27 降低至 15，但误报数量增加。

由于验证集中恶性样本数量远少于良性样本，类别分布极不平衡，因此 Precision 和 F1 分数较低。对于黑色素瘤筛查任务，AUC、Recall / Sensitivity 和 Specificity 比单纯 Accuracy 更具有参考意义。

---

## 8. 可视化结果

### 8.1 训练过程曲线

Baseline 训练过程输出：

```text
outputs/figures/baseline_resnet18_20260626_213923/
├── baseline_loss_curve.png
└── baseline_auc_curve.png
```

调优实验输出：

```text
outputs/figures/tuning_resnet18_20260626_222832/
├── optimizer_auc_compare.png
└── scheduler_auc_compare.png
```

微调策略实验输出：

```text
outputs/figures/finetune_resnet18_20260627_002934/
└── finetune_auc_compare.png
```

### 8.2 ROC 曲线与混淆矩阵

最终模型评估输出：

```text
outputs/figures/final_eval_resnet18_20260627_012535/
├── roc_curve.png
├── confusion_matrix_threshold_0.5.png
└── confusion_matrix_youden_threshold.png
```

### 8.3 Grad-CAM 可解释性分析

运行命令：

```bash
python scripts/generate_gradcam.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 4 --samples_per_group 4
```

Grad-CAM 输出：

```text
outputs/figures/gradcam_resnet18_20260627_013535/
├── TP_malignant_correct_summary.png
├── FN_malignant_missed_summary.png
├── FP_benign_false_alarm_summary.png
└── TN_benign_correct_summary.png
```

Grad-CAM 分析结论：

* 对于预测正确的恶性样本，模型热力区域大多集中在病灶主体或异常纹理区域；
* 对于漏诊恶性样本，热力图可能偏离病灶主体，说明模型未能捕捉关键恶性特征；
* 对于误报良性样本，模型常关注颜色较深、边界复杂或纹理变化明显的区域，说明模型可能将部分良性复杂色素区域误判为恶性风险。

### 8.4 t-SNE 特征可视化

运行命令：

```bash
python scripts/visualize_tsne.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 4
```

t-SNE 输出：

```text
outputs/figures/tsne_resnet18_20260627_014555/
├── tsne_features_sampled.png
└── tsne_features_balanced.png
```

t-SNE 分析结论：

在类别平衡抽样图中，恶性样本与良性样本呈现一定分离趋势，但仍存在明显交叠。这说明最终模型提取的深度特征具有一定判别能力，但皮肤病灶良恶性分类仍具有较高难度。

---

## 9. 主要实验结果汇总

| 实验阶段     | 最佳模型 / 策略                   | Best AUC |
| -------- | --------------------------- | -------: |
| Baseline | ResNet-18，仅训练分类头            |  0.83784 |
| 优化器对比    | AdamW + ReduceLROnPlateau   | 0.837864 |
| 微调策略对比   | Full Fine-tuning            | 0.885590 |
| 最终模型评估   | Full Fine-tuning best model | 0.885581 |

最终结论：

1. 迁移学习能够有效应用于皮肤病灶良恶性识别任务；
2. Adam / AdamW 优化器明显优于 SGD + momentum；
3. 全模型微调显著优于只训练分类头和解冻最后 10 层；
4. 最终模型在验证集上达到约 0.886 的 ROC-AUC；
5. 由于类别极度不平衡，模型 Precision 较低，但 Recall 和 AUC 具有较好的筛查意义；
6. Grad-CAM 和 t-SNE 结果表明模型具有一定可解释性和特征区分能力，但仍存在误报和漏诊问题。

---

## 10. 团队分工

本项目由两名成员合作完成。

| 成员   | 主要职责                                    |
| ---- | --------------------------------------- |
| zy_dingdongmiaomiao(3829) | 项目规划、实验设计、AI 交流记录、README、实验报告、结果分析与展示材料 |
| ym2333333(3853) | 代码运行、模型训练、实验结果生成、可视化结果输出、报错反馈与调试        |

说明：

由于zy(3829)同学的本地电脑性能有限，不适合长时间运行深度学习训练任务，因此主要负责项目设计、文档整理、实验分析与展示；ym(3853)同学主要负责模型训练与结果生成。两位成员均参与了项目设计与结果分析过程。

---

## 11. AI 工具使用说明

本项目在开发过程中使用了 AI 工具辅助完成：

* 项目难度评估；
* 实验流程设计；
* PyTorch 训练脚本生成；
* 代码调试与实验结果分析；
* README、实验报告和展示材料整理。

AI 工具交流与项目推进过程记录在：

```text
AI_LOG.md
```

该文件用于保存项目实施过程中的关键提问、分析、修改和结果总结。

---

## 12. 注意事项

1. 本仓库不上传原始 ISIC 图像数据；
2. 本仓库不上传 `.pth` 模型权重文件；
3. 若需要复现实验，请先下载数据集并按照指定目录结构放置；
4. 由于训练过程涉及随机划分、GPU 环境和数据增强，复现实验结果可能存在轻微波动；
5. 本项目结果仅用于课程实验与机器学习方法学习，不可直接用于临床诊断。

---

## 13. 项目展示

项目答辩材料：

```text
report/
└── defense_slides.pptx
```

最终实验报告：

```text
report/
└── project_report.md
```
