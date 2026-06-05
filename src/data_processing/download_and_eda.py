"""
下载两个中文情感数据集并合并进行 EDA。

数据源：
    1. lansinuote/ChnSentiCorp          (~12k 条，二分类)
    2. wangbulehouhouhou/chinese-sentiment  (~40k 条，六分类映射为二分类)

输出：
    - data/raw_chnsenticorp/*.csv
    - data/raw_chinese_sentiment/*.csv
    - data/eda/eda_report.txt
    - data/eda/length_distribution.png
    - data/eda/length_distribution_merged.png

用法：
    python -m src.data_processing.download_and_eda
"""

import os
from datasets import load_dataset
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 适配 Windows 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 六分类 → 二分类映射（chinese-sentiment 数据集）
# 正样本：joy(1), love(2), surprise(5)
# 负样本：sadness(0), anger(3), fear(4)
POS_LABELS = {1, 2, 5}
NEG_LABELS = {0, 3, 4}


def _map_chinese_sentiment_label(label: int) -> int | None:
    """将 chinese-sentiment 的 6 分类映射为二分类。"""
    if label in POS_LABELS:
        return 1
    elif label in NEG_LABELS:
        return 0
    return None


def _download_chnsenticorp(output_dir: str = "data/raw_chnsenticorp") -> pd.DataFrame:
    """下载 ChnSentiCorp 并保存。"""
    print("[1/2] 正在下载 lansinuote/ChnSentiCorp ...")
    ds = load_dataset("lansinuote/ChnSentiCorp")
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    split_map = {"train": "train", "validation": "val", "test": "test"}
    for split_hf, split_name in split_map.items():
        df = ds[split_hf].to_pandas()
        df = df[["text", "label"]].copy()
        df["source"] = "chnsenticorp"
        df["split"] = split_name
        save_path = os.path.join(output_dir, f"{split_name}.csv")
        df[["text", "label"]].to_csv(save_path, index=False, encoding="utf-8")
        frames.append(df)
        pos = int(df["label"].sum())
        neg = len(df) - pos
        print(f"  {split_name}.csv: {len(df)} 条 (pos={pos}, neg={neg})")

    return pd.concat(frames, ignore_index=True)


def _download_chinese_sentiment(output_dir: str = "data/raw_chinese_sentiment") -> pd.DataFrame:
    """下载 chinese-sentiment 并保存，同时进行二分类映射。"""
    print("[2/2] 正在下载 wangbulehouhouhou/chinese-sentiment ...")
    ds = load_dataset("wangbulehouhouhou/chinese-sentiment", verification_mode="no_checks")
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    split_map = {"train": "train", "validation": "val", "test": "test"}
    for split_hf, split_name in split_map.items():
        df = ds[split_hf].to_pandas()
        # 映射为二分类
        df["label"] = df["label"].apply(_map_chinese_sentiment_label)
        # 丢弃无法映射的标签（理论上不存在）
        before = len(df)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        after = len(df)
        if before != after:
            print(f"  {split_name}: 过滤未知标签 {before} -> {after}")

        df = df[["text", "label"]].copy()
        df["source"] = "chinese_sentiment"
        df["split"] = split_name
        save_path = os.path.join(output_dir, f"{split_name}.csv")
        df[["text", "label"]].to_csv(save_path, index=False, encoding="utf-8")
        frames.append(df)
        pos = int(df["label"].sum())
        neg = len(df) - pos
        print(f"  {split_name}.csv: {len(df)} 条 (pos={pos}, neg={neg})")

    return pd.concat(frames, ignore_index=True)


def _eda_single(df: pd.DataFrame, name: str, output_dir: str):
    """对单个数据集做 EDA 统计。"""
    lines = [f"\n【{name} 数据集统计】"]
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        total = len(sub)
        pos = int(sub["label"].sum())
        neg = total - pos
        lines.append(
            f"  {split:8s}: 总计={total:6d} | 正样本={pos:5d} | 负样本={neg:5d} | 正样本比例={pos/total:.2%}"
        )
    # 文本长度
    df["text_len"] = df["text"].astype(str).apply(len)
    lines.append(f"\n  文本长度统计:")
    lines.append(str(df["text_len"].describe()))
    return "\n".join(lines)


def _plot_length(df: pd.DataFrame, title: str, save_path: str):
    """绘制文本长度分布直方图。"""
    df = df.copy()
    df["text_len"] = df["text"].astype(str).apply(len)

    plt.figure(figsize=(10, 5))
    plt.hist(df["text_len"], bins=60, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("文本长度（字符数）")
    plt.ylabel("样本数量")

    mean_len = df["text_len"].mean()
    median_len = df["text_len"].median()
    plt.axvline(mean_len, color="red", linestyle="--", linewidth=2, label=f"均值={mean_len:.1f}")
    plt.axvline(median_len, color="green", linestyle="--", linewidth=2, label=f"中位数={median_len:.1f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  图表已保存: {save_path}")
    plt.close()


def eda(df_chn: pd.DataFrame, df_big: pd.DataFrame, output_dir: str = "data/eda"):
    """合并 EDA：分别统计 + 合并统计。"""
    os.makedirs(output_dir, exist_ok=True)

    report_lines = ["=" * 60, "ChnSentiCorp + chinese-sentiment 合并 EDA 报告", "=" * 60]

    # 单个数据集统计
    report_lines.append(_eda_single(df_chn, "ChnSentiCorp", output_dir))
    report_lines.append(_eda_single(df_big, "chinese-sentiment", output_dir))

    # 合并统计
    df_merged = pd.concat([df_chn, df_big], ignore_index=True)
    report_lines.append("\n【合并后总统计】")
    for split in ["train", "val", "test"]:
        sub = df_merged[df_merged["split"] == split]
        total = len(sub)
        pos = int(sub["label"].sum())
        neg = total - pos
        report_lines.append(
            f"  {split:8s}: 总计={total:6d} | 正样本={pos:5d} | 负样本={neg:5d} | 正样本比例={pos/total:.2%}"
        )
    report_lines.append(f"  {'Total':8s}: {len(df_merged)} 条")

    df_merged["text_len"] = df_merged["text"].astype(str).apply(len)
    report_lines.append(f"\n  合并后文本长度统计:")
    report_lines.append(str(df_merged["text_len"].describe()))

    # 保存报告
    report_path = os.path.join(output_dir, "eda_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nEDA 报告已保存: {report_path}")
    print("\n".join(report_lines))

    # 绘制分布图
    _plot_length(df_chn, "ChnSentiCorp 训练集文本长度分布", os.path.join(output_dir, "length_distribution_chn.png"))
    _plot_length(df_big, "chinese-sentiment 训练集文本长度分布", os.path.join(output_dir, "length_distribution_big.png"))
    _plot_length(df_merged, "合并后训练集文本长度分布", os.path.join(output_dir, "length_distribution_merged.png"))


def main():
    df_chn = _download_chnsenticorp()
    df_big = _download_chinese_sentiment()
    eda(df_chn, df_big)
    print("\n[OK] 数据下载与 EDA 完成！")
    print("   ChnSentiCorp:        ~12,000 条")
    print("   chinese-sentiment:   ~40,000 条")
    print("   合并总计:            ~52,000 条")


if __name__ == "__main__":
    main()
