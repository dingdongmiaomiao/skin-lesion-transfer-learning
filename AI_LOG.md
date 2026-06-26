# AI 工具使用记录

本文件用于记录本项目在开发过程中与 AI 工具的交流、问题分析、代码修改和实验设计过程。

课程大项目要求开源仓库中保存与 AI 工具交流的记录文件，因此本项目将关键交流过程整理在此文件中。

---

## 记录 1：项目选题与整体规划

### 提问内容

我们计划完成一个机器学习课程大项目，选题为：

《基于迁移学习的皮肤病灶良恶性识别——以 ISIC 黑色素瘤数据为例》

项目使用 CNN 和迁移学习方法，对 ISIC 皮肤镜图像进行二分类，类别为良性和恶性。希望评估项目从零开始的难度，并规划两人分工。

### AI 辅助分析结果

AI 建议该项目难度为中等偏上，适合作为机器学习课程大项目。项目核心不应只是简单训练一个模型，而应形成完整实验流程，包括：

1. 数据读取与预处理；
2. 基线模型训练；
3. 优化器和学习率调度器对比；
4. 不同迁移学习微调策略对比；
5. 数据增强与正则化消融实验；
6. 混淆矩阵、t-SNE 和 Grad-CAM 等可视化分析；
7. README、实验报告和答辩材料整理。

### 分工调整

由于一名成员电脑性能有限，不适合长时间跑深度学习训练任务，因此项目分工调整为：

- 成员 A：负责项目规划、仓库文档、实验记录、结果分析、报告和答辩材料；
- 成员 B：负责主要代码运行、模型训练、实验结果生成和报错反馈。

这种分工可以保证两位成员都有真实贡献，并且适合课程对开源仓库和 Commit History 的要求。

---

## 记录 2：仓库初始化

### 提问内容

希望按工序逐步生成项目文件，不在一个对话中一次性完成全部内容。

### AI 辅助结果

AI 建议按照以下工序推进：

1. 工序0：仓库初始化与项目结构；
2. 工序1：数据读取与 baseline 训练；
3. 工序2：优化器与学习率调度对比；
4. 工序3：迁移学习微调策略对比；
5. 工序4：数据增强与正则化消融；
6. 工序5：Grad-CAM 可解释性分析；
7. 工序6：混淆矩阵、t-SNE 与最终结果汇总；
8. 工序7：README、实验报告、答辩材料整理。

当前完成了项目初始目录结构、`.gitignore` 和 `requirements.txt` 文件。

---

## 记录 3：小样本调试测试与 Baseline 运行检查

### 本阶段目标

在正式使用完整 ISIC 训练集进行模型训练之前，为了避免因数据路径、图片读取、CUDA 环境、模型加载或保存逻辑错误导致长时间训练失败，本阶段先使用小规模样本进行调试测试。

由于完整训练集包含 33126 张皮肤镜图像，若直接进行完整训练，一旦代码或环境存在问题，会浪费较多时间。因此，本阶段在 `train_baseline.py` 中加入了 `--debug_samples` 参数，用于只抽取少量样本进行快速测试。

### 新增调试参数

在 `scripts/train_baseline.py` 中新增参数：

```bash
--debug_samples
```

该参数用于控制调试时使用的样本数量。

* `--debug_samples 300`：只使用 300 张图像进行快速测试；
* `--debug_samples 1000`：只使用 1000 张图像进行进一步测试；
* 不添加该参数，或设置为 `-1`：使用完整训练集。

### 当前测试命令

本次小样本测试使用如下命令：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 1 --batch_size 8 --img_size 224 --num_workers 0 --debug_samples 300
```

其中各参数含义如下：

* `--model resnet18`：使用 ResNet-18 作为 baseline 模型；
* `--epochs 1`：只训练 1 个 epoch，用于快速检查代码流程；
* `--batch_size 8`：每次训练输入 8 张图片，适合显存较小的显卡；
* `--img_size 224`：将输入图像统一处理为 224×224；
* `--num_workers 0`：Windows 环境下优先使用单进程读取数据，降低 DataLoader 多进程报错概率；
* `--debug_samples 300`：只抽取 300 张图片进行调试，不使用完整训练集。

### 本次测试主要检查内容

本次测试并不追求模型准确率，主要用于检查以下内容：

1. `ground_truth.csv` 是否能够被正常读取；
2. `image_name` 与图片文件名是否能够正确对应；
3. `data/ISIC_2020/train/` 中的图片是否能够被正常加载；
4. 图像预处理与数据增强流程是否正常；
5. ResNet-18 预训练模型是否能够正常加载；
6. CUDA 是否能够被 PyTorch 正确识别并使用；
7. 训练循环、验证循环、AUC 计算是否能正常运行；
8. 模型权重、训练日志和曲线图是否能够自动保存。

### 正确运行时的预期现象

如果 CUDA 环境配置正确，程序开头应显示类似信息：

```text
使用设备: cuda
混合精度训练 AMP: True
```

如果显示：

```text
使用设备: cpu
```

则说明当前环境没有成功调用 GPU，需要检查 PyTorch CUDA 版本、显卡驱动或虚拟环境配置。

程序正常运行结束后，应在 `outputs/` 目录下生成对应的实验结果，包括：

```text
outputs/
├── checkpoints/
│   └── baseline_resnet18_时间戳/
│       ├── best_model.pth
│       └── last_model.pth
├── figures/
│   └── baseline_resnet18_时间戳/
│       ├── baseline_loss_curve.png
│       └── baseline_auc_curve.png
└── tables/
    └── baseline_resnet18_时间戳/
        └── baseline_metrics.csv
```

### 测试结果记录

本次测试状态：

```text
测试样本数：300
训练轮数：1 epoch
模型：ResNet-18
Batch Size：8
图像尺寸：224×224
num_workers：0
测试目的：检查代码、数据路径、CUDA 环境和结果保存流程是否正常
```

测试完成后，需要记录以下结果：

```text
是否成功读取数据：是
是否成功调用 CUDA：是
是否成功完成训练：是
是否生成 best_model.pth：是
是否生成 baseline_metrics.csv：
是否生成 Loss/AUC 曲线图：是
终端是否出现报错：否
```

### 后续计划

如果 300 张图片的小样本测试能够正常运行，将进一步使用 1000 张图片进行调试测试：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 1 --batch_size 8 --img_size 224 --num_workers 0 --debug_samples 1000
```

若 1000 张图片测试也能正常运行，再使用完整训练集进行 baseline 训练：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 1 --batch_size 8 --img_size 224 --num_workers 0
```

最后根据硬件情况，将正式 baseline 实验扩展到 3～15 个 epoch，用于获得完整的训练曲线和验证 AUC 结果。

---

## 记录 4：Baseline 训练检查与项目文件管理调整

### 当前进度

目前项目已完成以下工作：

1. 完成 `train-metadata.csv` 到 `ground_truth.csv` 的格式转换；
2. 完成 300 张与 1000 张图片的小样本测试；
3. 完成完整数据集 1 epoch 的 baseline 训练测试；
4. 确认训练流程可以正常运行，模型能够完成训练与验证；
5. 完成 PyTorch AMP 相关 `FutureWarning` 的定位与修改；
6. 新增数据集完整性检查脚本，用于检查图片是否缺失、损坏或无法读取。

### FutureWarning 调整

在完整数据 baseline 训练过程中，程序出现如下警告：

```text
FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated.
Please use torch.amp.GradScaler('cuda', args...) instead.
```

该警告不影响训练结果，但说明当前 AMP 混合精度接口写法已经较旧。因此对 `scripts/train_baseline.py` 进行了修改：

```python
from torch.cuda.amp import GradScaler, autocast
```

修改为：

```python
from torch.amp import GradScaler, autocast
```

同时将训练过程中的 `autocast` 和 `GradScaler` 调整为新版接口，避免后续版本兼容问题。

### 图片完整性检查

由于完整数据集包含 33126 张皮肤镜图片，为避免隐藏坏图或文件名不匹配影响后续实验，新增数据完整性检查脚本：

```text
scripts/check_dataset_integrity.py
```

该脚本主要检查：

1. `ground_truth.csv` 是否包含 `image_name` 和 `target`；
2. `target` 是否只有 0 和 1；
3. 是否存在重复图片名；
4. CSV 中记录的图片是否都能在图片文件夹中找到；
5. 每张图片是否能被 PIL 正常打开和读取；
6. 图片文件夹中是否存在未被 CSV 使用的额外图片。

运行命令：

```bash
python scripts/check_dataset_integrity.py
```

完整数据 1 epoch 已经能够顺利跑完，说明训练与验证过程中用到的图片基本都能被正常读取；完整性检查脚本用于进一步确认数据集无隐藏问题。

### 工序 1 当前状态

工序 1：Baseline 训练目前已基本跑通。

已完成：

* 数据读取；
* 数据集划分；
* ResNet-18 baseline 模型构建；
* CUDA 与混合精度训练；
* 小样本测试；
* 完整数据 1 epoch 测试；
* 输出模型权重、训练日志和曲线图；
* FutureWarning 修复；
* 图片完整性检查脚本。

当前工序 1 仅剩：

```text
完整数据 15 epoch baseline 正式实验
```

建议正式运行命令：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

若 Windows 下 `num_workers=4` 不稳定，则改为：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 15 --batch_size 16 --img_size 224 --num_workers 0
```

当前优先级：

```text
先完成工序 1 的 15 epoch baseline 正式实验；
同时建立一个统一的远程仓库（github desktop）；
之后每完成一个工序，就同步上传对应代码、结果和 AI_LOG 记录。
```

## 记录 5：Baseline 实验结果

### 实验设置

本次实验使用 ResNet-18 作为 baseline 模型，加载 ImageNet 预训练权重，并冻结 backbone，仅训练最后的二分类分类头。

主要参数如下：

- model: resnet18
- epochs: 15
- batch_size: 16
- img_size: 224
- optimizer: Adam
- lr: 1e-3
- scheduler: ReduceLROnPlateau
- loss: BCEWithLogitsLoss with pos_weight
- metric: Validation AUC

### 实验结果

训练 15 个 epoch 后，模型取得的最佳验证集 AUC 为：

- Best Valid AUC: 0.83784

最佳模型保存路径为：

```text
outputs/checkpoints/baseline_resnet18_20260626_213923/best_model.pth
```

## 记录 6：工序2 调优实验设计

### 本阶段目标

本阶段在 baseline 模型基础上进行调优实验对照，比较不同优化器和不同学习率调度策略对验证集 AUC 的影响。

### 固定条件

为了保证实验公平，本阶段固定以下条件：

- 数据集划分方式不变；
- 模型使用 ResNet-18；
- 使用 ImageNet 预训练权重；
- 冻结 backbone，仅训练分类头；
- 图像尺寸为 224；
- 评价指标为 Validation AUC；
- 损失函数使用带 `pos_weight` 的 `BCEWithLogitsLoss`。

### 对比内容

优化器对比包括：

1. Adam；
2. AdamW；
3. SGD + momentum。

学习率调度器对比包括：

1. Fixed Learning Rate；
2. CosineAnnealingLR；
3. ReduceLROnPlateau。

### 输出结果

本阶段脚本会自动保存：

- 每组实验的 `metrics.csv`；
- 每组实验的 `config.json`；
- 汇总表格 `tuning_summary.csv`；
- 优化器 AUC 对比图；
- 学习率调度器 AUC 对比图。

本阶段结果将用于判断后续微调策略实验应采用哪种优化器和学习率调度方式。

### 调优实验结果分析

在 baseline 模型基础上，本项目进一步比较了不同优化器和学习率调度策略对模型性能的影响。为保证实验公平，本阶段固定模型结构为 ResNet-18，加载 ImageNet 预训练权重，并冻结 backbone，仅训练最后的二分类头。评价指标采用验证集 AUC。

优化器对比结果显示，Adam 和 AdamW 均取得了约 0.838 的最佳验证 AUC，其中 AdamW 的 Best AUC 为 0.837864，Adam 的 Best AUC 为 0.837860，二者表现几乎一致；SGD+momentum 的 Best AUC 为 0.822258，明显低于 Adam 系列优化器，说明在本任务和当前训练设置下，自适应优化器具有更好的收敛效果。

学习率调度器对比结果显示，Fixed LR、ReduceLROnPlateau 和 CosineAnnealingLR 的 Best AUC 分别为 0.837980、0.837860 和 0.835348。其中 Fixed LR 数值最高，但与 ReduceLROnPlateau 的差距极小。考虑到后续微调实验会解冻更多网络层，训练过程可能更加不稳定，因此后续实验选择 AdamW + ReduceLROnPlateau 作为默认训练策略。

## 记录 7：工序2 调优实验结果

### 实验设置

本阶段在 baseline 基础上进行调优实验对照。实验固定模型为 ResNet-18，加载 ImageNet 预训练权重，冻结 backbone，仅训练最后的分类头。评价指标为 Validation AUC。

本阶段共运行 5 组实验：

1. Adam + ReduceLROnPlateau；
2. AdamW + ReduceLROnPlateau；
3. SGD + momentum + ReduceLROnPlateau；
4. Adam + Fixed LR；
5. Adam + CosineAnnealingLR。

### 实验结果

各组实验的最佳验证集 AUC 如下：

| 实验组 | Optimizer | Scheduler | Best AUC | Best Epoch |
|---|---|---|---:|---:|
| adam_plateau | Adam | ReduceLROnPlateau | 0.837860 | 15 |
| adamw_plateau | AdamW | ReduceLROnPlateau | 0.837864 | 15 |
| sgd_plateau | SGD + momentum | ReduceLROnPlateau | 0.822258 | 15 |
| adam_fixed | Adam | Fixed LR | 0.837980 | 13 |
| adam_cosine | Adam | CosineAnnealingLR | 0.835348 | 13 |

### 结果分析

优化器对比结果显示，Adam 和 AdamW 的表现非常接近，二者 Best AUC 均约为 0.838；SGD+momentum 的表现明显低于 Adam 系列优化器。

学习率调度器对比结果显示，Fixed LR 与 ReduceLROnPlateau 的表现非常接近，其中 Fixed LR 的 Best AUC 略高，但差距很小；CosineAnnealingLR 的表现略低。

综合考虑后续微调实验中训练参数量增加、训练过程更容易波动，本项目后续采用 AdamW + ReduceLROnPlateau 作为默认训练策略。