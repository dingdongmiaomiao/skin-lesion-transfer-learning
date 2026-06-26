# 仓库结构、运行说明与文件放置步骤

## 一、最终仓库结构

本项目最终仓库建议整理为以下结构：

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
│           ├── ISIC_xxxxxxx.jpg
│           ├── ISIC_xxxxxxx.jpg
│           └── ...
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
│   │   ├── baseline_resnet18_20260626_213923/
│   │   ├── tuning_resnet18_20260626_222832/
│   │   ├── finetune_resnet18_20260627_002934/
│   │   ├── final_eval_resnet18_20260627_012535/
│   │   ├── gradcam_resnet18_20260627_013535/
│   │   └── tsne_resnet18_20260627_014555/
│   └── tables/
│       ├── baseline_resnet18_20260626_213923/
│       ├── tuning_resnet18_20260626_222832/
│       ├── finetune_resnet18_20260627_002934/
│       ├── final_eval_resnet18_20260627_012535/
│       ├── gradcam_resnet18_20260627_013535/
│       └── tsne_resnet18_20260627_014555/
└── report/
    ├── project_report.md
    ├── video_script.md
    ├── repository_guide.md
    └── ppt_assets/
```

---

## 二、各文件和文件夹作用说明

### 1. 根目录文件

| 文件                 | 作用                          |
| ------------------ | --------------------------- |
| `README.md`        | 项目总说明，包括项目简介、实验流程、结果汇总和运行方式 |
| `requirements.txt` | Python 依赖列表                 |
| `.gitignore`       | Git 忽略规则，用于排除数据集、模型权重和缓存文件  |
| `AI_LOG.md`        | AI 工具使用记录和项目开发过程记录          |

---

### 2. `data/` 数据目录

```text
data/
└── ISIC_2020/
    ├── ground_truth.csv
    └── train/
```

该目录用于放置 ISIC 数据集。

说明：

* `ground_truth.csv` 是标签文件；
* `train/` 中存放所有 `.jpg` 图像；
* 由于数据集体积较大，不建议上传到 GitHub / Gitee；
* 但应在 README 中说明数据应如何放置。

`.gitignore` 中应包含：

```gitignore
data/ISIC_2020/train/
data/ISIC_2020/*.jpg
data/ISIC_2020/*.png
```

如果希望保留空文件夹，可以在 `data/ISIC_2020/` 下放一个 `.gitkeep` 文件。

---

### 3. `src/` 核心代码目录

```text
src/
├── __init__.py
├── dataset.py
├── models.py
└── utils.py
```

| 文件            | 作用                                   |
| ------------- | ------------------------------------ |
| `dataset.py`  | 定义 ISIC 数据集类，以及训练集/验证集图像预处理          |
| `models.py`   | 构建 ResNet-18 / EfficientNet-B0 二分类模型 |
| `utils.py`    | 随机种子、目录创建、指标保存、曲线绘制等工具函数             |
| `__init__.py` | 使 `src` 目录可以作为 Python 包导入            |

---

### 4. `scripts/` 实验脚本目录

```text
scripts/
├── train_baseline.py
├── run_tuning_experiments.py
├── run_finetune_experiments.py
├── evaluate_final_model.py
├── generate_gradcam.py
└── visualize_tsne.py
```

| 脚本                            | 作用                      |
| ----------------------------- | ----------------------- |
| `train_baseline.py`           | 训练 baseline 模型          |
| `run_tuning_experiments.py`   | 运行优化器和学习率调度器对比实验        |
| `run_finetune_experiments.py` | 运行迁移学习微调策略对比实验          |
| `evaluate_final_model.py`     | 对最终模型进行 ROC、混淆矩阵和分类指标评估 |
| `generate_gradcam.py`         | 生成 Grad-CAM 可解释性热力图     |
| `visualize_tsne.py`           | 生成 t-SNE 特征可视化图         |

---

### 5. `outputs/` 实验结果目录

```text
outputs/
├── checkpoints/
├── logs/
├── figures/
└── tables/
```

| 文件夹            | 内容                                           |
| -------------- | -------------------------------------------- |
| `checkpoints/` | 保存模型权重 `.pth`，不建议上传仓库                        |
| `logs/`        | 保存运行日志                                       |
| `figures/`     | 保存实验结果图，例如 AUC 曲线、ROC 曲线、混淆矩阵、Grad-CAM、t-SNE |
| `tables/`      | 保存实验结果表格，例如 `metrics.csv`、`summary.csv`      |

建议上传：

```text
outputs/figures/
outputs/tables/
```

不建议上传：

```text
outputs/checkpoints/
*.pth
```

因为模型权重体积较大，不适合放进普通课程仓库。

---

### 6. `report/` 报告与展示材料目录

```text
report/
├── project_report.md
├── video_script.md
├── repository_guide.md
└── ppt_assets/
```

| 文件                    | 作用                 |
| --------------------- | ------------------ |
| `project_report.md`   | 最终实验报告             |
| `video_script.md`     | 项目展示视频讲稿和 PPT 页面规划 |
| `repository_guide.md` | 仓库结构、运行说明和文件放置步骤   |
| `ppt_assets/`         | PPT 中使用的图片素材       |

---

## 三、环境配置步骤

### 1. 克隆仓库

```bash
git clone <your-repository-url>
cd skin-lesion-transfer-learning
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```bash
.venv\Scripts\activate
```

macOS / Linux 激活虚拟环境：

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，建议根据本机 CUDA 版本安装对应的 PyTorch GPU 版本。

---

## 四、数据放置步骤

### 1. 下载数据集

下载 ISIC 2020 / SIIM-ISIC Melanoma Classification 数据集。

### 2. 放置标签文件

将标签文件放到：

```text
data/ISIC_2020/ground_truth.csv
```

该文件至少需要包含：

```text
image_name,target
```

其中：

* `image_name`：图片名，不带 `.jpg` 后缀；
* `target`：标签，`0` 表示良性，`1` 表示恶性。

### 3. 放置图像文件

将所有训练图像放到：

```text
data/ISIC_2020/train/
```

例如：

```text
data/ISIC_2020/train/ISIC_0015719.jpg
data/ISIC_2020/train/ISIC_0052212.jpg
data/ISIC_2020/train/ISIC_0068279.jpg
```

### 4. 检查目录

最终应满足：

```text
data/
└── ISIC_2020/
    ├── ground_truth.csv
    └── train/
        ├── ISIC_xxxxxxx.jpg
        └── ...
```

---

## 五、完整运行流程

下面给出从 baseline 到最终可视化的完整运行命令。

---

### 1. Baseline 训练

运行：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

如果 Windows 下多进程 DataLoader 报错，使用：

```bash
python scripts/train_baseline.py --model resnet18 --epochs 15 --batch_size 16 --img_size 224 --num_workers 0
```

输出结果：

```text
outputs/checkpoints/baseline_resnet18_时间戳/
outputs/figures/baseline_resnet18_时间戳/
outputs/tables/baseline_resnet18_时间戳/
```

主要结果：

```text
baseline_loss_curve.png
baseline_auc_curve.png
baseline_metrics.csv
```

---

### 2. 优化器与学习率调度器对比实验

运行：

```bash
python scripts/run_tuning_experiments.py --experiment_set all --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

如果 Windows 下多进程报错：

```bash
python scripts/run_tuning_experiments.py --experiment_set all --epochs 15 --batch_size 16 --img_size 224 --num_workers 0
```

输出结果：

```text
outputs/figures/tuning_resnet18_时间戳/
outputs/tables/tuning_resnet18_时间戳/
```

主要结果：

```text
optimizer_auc_compare.png
scheduler_auc_compare.png
tuning_summary.csv
```

---

### 3. 微调策略对比实验

运行：

```bash
python scripts/run_finetune_experiments.py --experiment_set all --epochs 15 --batch_size 16 --img_size 224 --num_workers 4
```

如果 Windows 下多进程报错：

```bash
python scripts/run_finetune_experiments.py --experiment_set all --epochs 15 --batch_size 16 --img_size 224 --num_workers 0
```

输出结果：

```text
outputs/checkpoints/finetune_resnet18_时间戳/
outputs/figures/finetune_resnet18_时间戳/
outputs/tables/finetune_resnet18_时间戳/
```

主要结果：

```text
finetune_auc_compare.png
finetune_summary.csv
```

最终模型位于：

```text
outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth
```

注意：该 `.pth` 文件不建议上传仓库。

---

### 4. 最终模型评估

运行：

```bash
python scripts/evaluate_final_model.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 4
```

如果 Windows 下多进程报错：

```bash
python scripts/evaluate_final_model.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 0
```

输出结果：

```text
outputs/figures/final_eval_resnet18_时间戳/
outputs/tables/final_eval_resnet18_时间戳/
```

主要结果：

```text
roc_curve.png
confusion_matrix_threshold_0.5.png
confusion_matrix_youden_threshold.png
final_metrics.csv
final_metrics.json
```

---

### 5. Grad-CAM 可解释性分析

运行：

```bash
python scripts/generate_gradcam.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 4 --samples_per_group 4
```

如果 Windows 下多进程报错：

```bash
python scripts/generate_gradcam.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 0 --samples_per_group 4
```

输出结果：

```text
outputs/figures/gradcam_resnet18_时间戳/
outputs/tables/gradcam_resnet18_时间戳/
```

主要结果：

```text
TP_malignant_correct_summary.png
FN_malignant_missed_summary.png
FP_benign_false_alarm_summary.png
TN_benign_correct_summary.png
gradcam_selected_cases.csv
```

---

### 6. t-SNE 特征可视化

运行：

```bash
python scripts/visualize_tsne.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 4
```

如果 Windows 下多进程报错：

```bash
python scripts/visualize_tsne.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 0
```

如果 t-SNE 运行较慢，可减少样本数：

```bash
python scripts/visualize_tsne.py --checkpoint_path outputs/checkpoints/finetune_resnet18_20260627_002934/full_finetune/best_model.pth --batch_size 16 --num_workers 0 --max_samples 1500 --balanced_per_class 80
```

输出结果：

```text
outputs/figures/tsne_resnet18_时间戳/
outputs/tables/tsne_resnet18_时间戳/
```

主要结果：

```text
tsne_features_sampled.png
tsne_features_balanced.png
```

---

## 六、建议上传到仓库的文件

### 1. 必须上传

```text
README.md
requirements.txt
.gitignore
AI_LOG.md
src/
scripts/
report/
```

### 2. 建议上传的实验结果

```text
outputs/figures/
outputs/tables/
```

特别是以下结果图和表格：

```text
outputs/figures/baseline_resnet18_20260626_213923/baseline_loss_curve.png
outputs/figures/baseline_resnet18_20260626_213923/baseline_auc_curve.png

outputs/figures/tuning_resnet18_20260626_222832/optimizer_auc_compare.png
outputs/figures/tuning_resnet18_20260626_222832/scheduler_auc_compare.png
outputs/tables/tuning_resnet18_20260626_222832/tuning_summary.csv

outputs/figures/finetune_resnet18_20260627_002934/finetune_auc_compare.png
outputs/tables/finetune_resnet18_20260627_002934/finetune_summary.csv

outputs/figures/final_eval_resnet18_20260627_012535/roc_curve.png
outputs/figures/final_eval_resnet18_20260627_012535/confusion_matrix_threshold_0.5.png
outputs/figures/final_eval_resnet18_20260627_012535/confusion_matrix_youden_threshold.png
outputs/tables/final_eval_resnet18_20260627_012535/final_metrics.csv

outputs/figures/gradcam_resnet18_20260627_013535/TP_malignant_correct_summary.png
outputs/figures/gradcam_resnet18_20260627_013535/FN_malignant_missed_summary.png
outputs/figures/gradcam_resnet18_20260627_013535/FP_benign_false_alarm_summary.png
outputs/figures/gradcam_resnet18_20260627_013535/TN_benign_correct_summary.png

outputs/figures/tsne_resnet18_20260627_014555/tsne_features_sampled.png
outputs/figures/tsne_resnet18_20260627_014555/tsne_features_balanced.png
```

---

## 七、不建议上传到仓库的文件

以下文件不建议上传：

```text
data/ISIC_2020/train/
*.jpg
*.png
outputs/checkpoints/
*.pth
*.pt
*.ckpt
__pycache__/
.venv/
```

原因：

1. 原始数据集体积很大；
2. 模型权重文件体积较大；
3. 虚拟环境和缓存文件不属于项目源码；
4. 上传这些文件会导致仓库过大，不利于查看和提交。

---

## 八、`.gitignore` 建议内容

```gitignore
# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
.env/
.venv/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Dataset
data/ISIC_2020/train/
data/ISIC_2020/*.jpg
data/ISIC_2020/*.png

# Model checkpoints
outputs/checkpoints/
*.pth
*.pt
*.ckpt

# Runtime logs
outputs/logs/
*.log

# Temporary files
tmp/
temp/
```

---

## 九、推荐 Commit 顺序

为了让仓库贡献记录更清楚，建议按阶段提交。

### 1. 初始化项目结构

```bash
git add README.md requirements.txt .gitignore AI_LOG.md
git add src scripts report
git commit -m "init project structure and documentation"
git push
```

### 2. Baseline 结果

```bash
git add scripts/train_baseline.py
git add outputs/figures/baseline_resnet18_20260626_213923/
git add outputs/tables/baseline_resnet18_20260626_213923/
git commit -m "add baseline training results"
git push
```

### 3. 调优实验结果

```bash
git add scripts/run_tuning_experiments.py
git add outputs/figures/tuning_resnet18_20260626_222832/
git add outputs/tables/tuning_resnet18_20260626_222832/
git commit -m "add tuning experiment results"
git push
```

### 4. 微调实验结果

```bash
git add scripts/run_finetune_experiments.py
git add outputs/figures/finetune_resnet18_20260627_002934/
git add outputs/tables/finetune_resnet18_20260627_002934/
git commit -m "add fine-tuning strategy results"
git push
```

### 5. 最终模型评估结果

```bash
git add scripts/evaluate_final_model.py
git add outputs/figures/final_eval_resnet18_20260627_012535/
git add outputs/tables/final_eval_resnet18_20260627_012535/
git commit -m "add final model evaluation results"
git push
```

### 6. Grad-CAM 与 t-SNE

```bash
git add scripts/generate_gradcam.py scripts/visualize_tsne.py
git add outputs/figures/gradcam_resnet18_20260627_013535/
git add outputs/tables/gradcam_resnet18_20260627_013535/
git add outputs/figures/tsne_resnet18_20260627_014555/
git add outputs/tables/tsne_resnet18_20260627_014555/
git commit -m "add interpretability and feature visualization"
git push
```

### 7. 最终报告和展示材料

```bash
git add README.md AI_LOG.md
git add report/project_report.md
git add report/video_script.md
git add report/repository_guide.md
git commit -m "add final report and presentation materials"
git push
```

---

## 十、最终提交前检查清单

提交仓库前，建议逐项检查。

### 1. 基础文件

```text
[ ] README.md 已完成
[ ] requirements.txt 已完成
[ ] .gitignore 已完成
[ ] AI_LOG.md 已完成
```

### 2. 代码文件

```text
[ ] src/dataset.py
[ ] src/models.py
[ ] src/utils.py
[ ] scripts/train_baseline.py
[ ] scripts/run_tuning_experiments.py
[ ] scripts/run_finetune_experiments.py
[ ] scripts/evaluate_final_model.py
[ ] scripts/generate_gradcam.py
[ ] scripts/visualize_tsne.py
```

### 3. 实验结果

```text
[ ] baseline Loss / AUC 曲线
[ ] optimizer 对比图
[ ] scheduler 对比图
[ ] finetune 对比图
[ ] ROC 曲线
[ ] 混淆矩阵
[ ] Grad-CAM summary 图
[ ] t-SNE 图
[ ] summary.csv / metrics.csv 表格
```

### 4. 报告与展示

```text
[ ] report/project_report.md
[ ] report/video_script.md
[ ] report/repository_guide.md
[ ] PPT 或展示视频已准备
[ ] README 中视频链接已补充
```

### 5. 不应上传的内容

```text
[ ] 没有上传原始图像数据集
[ ] 没有上传 .pth 模型权重
[ ] 没有上传 .venv 虚拟环境
[ ] 没有上传 __pycache__ 缓存目录
```

---

## 十一、推荐最终仓库首页展示顺序

README 首页建议按照以下顺序展示：

```text
1. 项目简介
2. 数据集说明
3. 环境配置
4. 仓库结构
5. 实验流程
6. 主要结果表格
7. 关键结果图
8. 运行命令
9. 团队分工
10. AI 工具使用说明
11. 注意事项
```

这样老师打开仓库后，可以快速看到项目完成度、技术路线、实验结果和成员贡献。

---

## 十二、最终说明

本项目最终提交时，仓库应重点体现以下内容：

1. 有完整代码，而不是只有实验结果；
2. 有清晰运行方式，而不是只能在某一台电脑上运行；
3. 有实验图表和表格，而不是只给口头描述；
4. 有 AI_LOG，能够体现项目实施过程；
5. 有 README 和报告，能够说明实验设计、结果和不足；
6. 有明确分工，能够体现两名成员均参与项目。

本项目不上传原始数据集和模型权重，但需要在 README 中明确说明数据集放置路径和模型训练方式。这样既能保证仓库清晰，也能满足课程对开源仓库和项目展示的要求。
