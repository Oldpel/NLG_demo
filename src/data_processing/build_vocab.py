"""
基于训练集构建词表。

输入：data/train.csv（由 preprocess.py 生成）
输出：data/vocab.json

词表格式：
    {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "词": 4, ...}

用法：
    python -m src.data_processing.build_vocab
"""

import os
import sys
import json

import jieba
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DataProcessor


def build_vocab(
    train_csv: str = "data/train.csv",
    output_path: str = "data/vocab.json",
    max_size: int = 30000,
):
    """
    基于清洗后的训练集构建 Jieba 词表。

    Args:
        train_csv: 训练集 CSV 路径。
        output_path: 词表保存路径。
        max_size: 词表上限（不含特殊标记）。
    """
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"未找到 {train_csv}，请先运行 preprocess.py")

    df = pd.read_csv(train_csv)
    processor = DataProcessor()

    counter = Counter()
    print(f"正在统计训练集词频: {train_csv} ({len(df)} 条)")

    for idx, text in enumerate(df["text"], 1):
        clean = processor.clean_text(str(text), remove_stopwords=False)
        words = jieba.lcut(clean)
        # 过滤空字符串
        words = [w for w in words if w.strip()]
        counter.update(words)
        if idx % 1000 == 0:
            print(f"  已处理 {idx}/{len(df)} 条")

    # 保留高频词，不超过 max_size
    most_common = counter.most_common(max_size)

    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for idx, (word, freq) in enumerate(most_common, start=len(vocab)):
        vocab[word] = idx

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"\n词表构建完成:")
    print(f"  词表大小: {len(vocab)} (上限={max_size})")
    print(f"  覆盖词数: {len(counter)}")
    print(f"  保存路径: {output_path}")
    print(f"  Top 10 高频词: {most_common[:10]}")


if __name__ == "__main__":
    build_vocab()
