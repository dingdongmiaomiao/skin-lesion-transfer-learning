import argparse
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Check ISIC dataset integrity")

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
        help="图片文件夹路径",
    )

    parser.add_argument(
        "--output_report",
        type=str,
        default="outputs/tables/dataset_integrity_report.csv",
        help="输出检查报告路径",
    )

    return parser.parse_args()


def check_image_readable(image_path):
    """
    检查图片是否可以被正常打开和读取。
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        # verify 后需要重新打开一次，确认能真正解码成 RGB
        with Image.open(image_path) as img:
            img.convert("RGB").load()

        return True, ""

    except Exception as e:
        return False, str(e)


def main():
    args = parse_args()

    print("=" * 80)
    print("开始检查数据集完整性")
    print("CSV 路径:", args.csv_path)
    print("图片目录:", args.image_dir)
    print("=" * 80)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {args.csv_path}")

    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"找不到图片目录: {args.image_dir}")

    df = pd.read_csv(args.csv_path)

    print("CSV 形状:", df.shape)
    print("CSV 列名:", df.columns.tolist())

    required_cols = {"image_name", "target"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    df["image_name"] = df["image_name"].astype(str)
    df["target"] = df["target"].astype(int)

    print("\n标签分布:")
    print(df["target"].value_counts())

    invalid_targets = df[~df["target"].isin([0, 1])]
    if len(invalid_targets) > 0:
        print("\n警告：存在非法 target，示例:")
        print(invalid_targets.head())

    duplicated = df[df["image_name"].duplicated()]
    if len(duplicated) > 0:
        print("\n警告：存在重复 image_name，示例:")
        print(duplicated.head())
    else:
        print("\n未发现重复 image_name")

    issues = []

    csv_image_names = set(df["image_name"].tolist())

    print("\n开始检查 CSV 中引用的图片是否存在且可读取...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        image_name = row["image_name"]
        image_path = os.path.join(args.image_dir, image_name + ".jpg")

        if not os.path.exists(image_path):
            issues.append(
                {
                    "image_name": image_name,
                    "issue_type": "missing_file",
                    "message": f"找不到文件: {image_path}",
                }
            )
            continue

        readable, msg = check_image_readable(image_path)

        if not readable:
            issues.append(
                {
                    "image_name": image_name,
                    "issue_type": "unreadable_image",
                    "message": msg,
                }
            )

    print("\n开始检查图片文件夹中的额外图片...")

    folder_image_names = set()

    for filename in os.listdir(args.image_dir):
        lower = filename.lower()
        if lower.endswith((".jpg", ".jpeg", ".png")):
            name_without_ext = os.path.splitext(filename)[0]
            folder_image_names.add(name_without_ext)

    extra_images = folder_image_names - csv_image_names
    missing_in_folder = csv_image_names - folder_image_names

    print("\n检查结果汇总:")
    print(f"CSV 样本数: {len(csv_image_names)}")
    print(f"图片文件夹中图片数: {len(folder_image_names)}")
    print(f"CSV 中有但文件夹缺失的图片数: {len(missing_in_folder)}")
    print(f"文件夹中有但 CSV 未使用的额外图片数: {len(extra_images)}")
    print(f"缺失/损坏问题总数: {len(issues)}")

    if len(extra_images) > 0:
        print("\n额外图片示例:")
        print(list(extra_images)[:10])

    if len(missing_in_folder) > 0:
        print("\nCSV 中有但文件夹缺失的图片示例:")
        print(list(missing_in_folder)[:10])

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    if len(issues) > 0:
        report_df = pd.DataFrame(issues)
        report_df.to_csv(args.output_report, index=False, encoding="utf-8-sig")

        print("\n发现问题，已保存报告:")
        print(args.output_report)
        print(report_df.head())
    else:
        report_df = pd.DataFrame(columns=["image_name", "issue_type", "message"])
        report_df.to_csv(args.output_report, index=False, encoding="utf-8-sig")

        print("\n未发现缺失或损坏图片")
        print("空报告已保存:")
        print(args.output_report)

    print("=" * 80)
    print("数据集完整性检查完成")
    print("=" * 80)


if __name__ == "__main__":
    main()