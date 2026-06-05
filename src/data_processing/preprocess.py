"""
数据清洗与划分管线（支持多数据源合并）。

输入：
    - data/raw_chnsenticorp/*.csv
    - data/raw_chinese_sentiment/*.csv

输出：
    - data/train.csv, val.csv, test.csv（必须包含 text, label 两列）

清洗步骤：
    1. regex 去除 HTML 标签
    2. regex 去除 URL
    3. OpenCC 繁简转换
    4. 去停用词（可选，BiLSTM 路径建议启用）
    5. 过滤空文本
    6. 合并多个数据源的同 split 数据

用法：
    python -m src.data_processing.preprocess
"""

import os
import re
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DataProcessor


_DEFAULT_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
    "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "与", "及", "等", "或", "但", "而", "如果", "因为", "所以", "虽然", "被", "让", "把", "给",
    "向", "从", "以", "为", "将", "对", "已经", "可以", "进行", "通过", "根据", "作为", "需要",
    "使用", "时候", "现在", "开始", "一下", "一些", "一样", "一直", "一般", "不过", "不能",
    "不会", "不得", "不同", "不仅", "专门", "主要", "之", "么", "乎", "于", "吧", "只",
    "呢", "啊", "哦", "嗯", "唉", "哟", "吗", "嘛", "哇", "呀", "哪", "啥", "啦", "咯", "嘿",
    "哼", "哎", "咳", "哎哟", "得了", "也罢", "也就是说", "具体", "相关", "表示", "认为",
    "成为", "具有", "其中", "包括", "由于", "因此", "随着", "，", "。", "、", "；", "：",
    "？", "！", "“", "”", "‘", "’", "（", "）", "《", "》", "「", "」", "『", "』", "【",
    "】", "—", "…", "～",
}


def _fetch_stopwords(save_path: str = "data/stopwords.txt") -> str:
    """尝试下载标准中文停用词表，失败则使用内置停用词。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        return save_path

    url = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
    try:
        print(f"正在下载停用词表: {url}")
        urllib.request.urlretrieve(url, save_path)
        print(f"停用词表已保存: {save_path}")
    except Exception as e:
        print(f"下载停用词表失败 ({e})，使用内置默认停用词。")
        with open(save_path, "w", encoding="utf-8") as f:
            for w in sorted(_DEFAULT_STOPWORDS):
                f.write(w + "\n")
        print(f"内置停用词已保存: {save_path}")
    return save_path


def desensitize_text(text: str) -> str:
    """脱敏处理：清洗并替换隐私特征。"""
    text = str(text)
    # 1. 模糊化 11 位手机号码
    text = re.sub(r"1[3-9]\d{9}", "[PHONE]", text)
    # 2. 模糊化邮箱地址
    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", text
    )
    # 3. 模糊化 18 位身份证号
    text = re.sub(
        r"[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dX]",
        "[ID_CARD]",
        text,
    )
    return text


def _clean_single_csv(
    input_path: str,
    processor: DataProcessor,
    remove_stopwords: bool = True,
) -> pd.DataFrame:
    """清洗单个 CSV 文件，返回 DataFrame。"""
    df = pd.read_csv(input_path)
    # [Security] 确保只保留需要的列
    df = df[["text", "label"]].copy()

    print(f"  清洗 {os.path.basename(input_path)} ({len(df)} 条) ...")

    # [Security] 在业务清洗前强制执行隐私脱敏
    df["text"] = df["text"].astype(str).apply(desensitize_text)

    df["text"] = df["text"].apply(
        lambda x: processor.clean_text(x, remove_stopwords=remove_stopwords)
    )

    # 过滤空文本
    before = len(df)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"    过滤空文本: {before} -> {after} 条")

    return df


def preprocess(
    input_dirs: list[str] | None = None,
    output_dir: str = "data",
    remove_stopwords: bool = True,
):
    """
    对多个数据源的原始 CSV 进行清洗并合并输出标准化数据文件。

    Args:
        input_dirs: 原始数据目录列表。默认为两个数据源。
        output_dir: 输出目录。
        remove_stopwords: 是否去除停用词（BiLSTM 路径建议启用）。
    """
    if input_dirs is None:
        input_dirs = [
            "data/raw_chnsenticorp",
            "data/raw_chinese_sentiment",
        ]

    os.makedirs(output_dir, exist_ok=True)

    # 准备停用词
    stopwords_path = None
    if remove_stopwords:
        stopwords_path = _fetch_stopwords()

    processor = DataProcessor(stopwords_path=stopwords_path)

    for split in ["train", "val", "test"]:
        frames = []
        for input_dir in input_dirs:
            input_path = os.path.join(input_dir, f"{split}.csv")
            if not os.path.exists(input_path):
                print(f"  [跳过] 未找到 {input_path}")
                continue
            df = _clean_single_csv(input_path, processor, remove_stopwords)
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                f"未找到任何 {split}.csv 数据源，请先运行 download_and_eda.py"
            )

        # 合并多个数据源的同 split 数据
        merged = pd.concat(frames, ignore_index=True)
        print(f"  => {split}.csv 合并后: {len(merged)} 条")

        output_path = os.path.join(output_dir, f"{split}.csv")
        merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"  已保存: {output_path}")


if __name__ == "__main__":
    preprocess()
