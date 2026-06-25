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
