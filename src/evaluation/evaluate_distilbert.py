"""
DistilBERT 独立评估脚本。

功能：
    - 加载最佳 checkpoint 目录
    - 在测试集上评估
    - 输出 Accuracy, Precision, Recall, F1 到 JSON（格式同 BiLSTM）

用法：
    cd NLG
    python -m src.evaluation.evaluate_distilbert --device cpu
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.dataset import SentimentDataset


def get_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(args):
    device = get_device(preference=args.device)
    print(f"[INFO] 使用设备: {device}")

    # 加载模型和 tokenizer
    ckpt_dir = os.path.join(PROJECT_ROOT, args.checkpoint)
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint 目录不存在: {ckpt_dir}")

    print(f"[INFO] 加载模型: {ckpt_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model.to(device)

    # 读取训练元信息
    meta_path = os.path.join(ckpt_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[INFO] 训练元信息: Epoch {meta.get('epoch', '?')} | Val Acc={meta.get('val_acc', '?'):.4f}")

    # 测试集
    # 从 checkpoint 目录推断模型名（优先用训练时相同的 tokenizer）
    pretrained_model = getattr(args, "pretrained_model", ckpt_dir)
    test_csv = os.path.join(PROJECT_ROOT, args.test_csv)
    test_ds = SentimentDataset(
        csv_path=test_csv,
        model_type="distilbert",
        max_len=args.max_len,
        pretrained_model_name=pretrained_model,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"[INFO] 测试集样本数: {len(test_ds)}")

    # 推理
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    result = {
        "model": "DistilBERT",
        "checkpoint": args.checkpoint,
        "dataset": args.test_csv,
        "num_samples": len(all_labels),
        "metrics": {
            "accuracy": round(acc, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
        },
    }

    # 输出到控制台
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    for k, v in result["metrics"].items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 50)

    # 保存到 JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, args.output_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果已保存: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DistilBERT 评估脚本")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/distilbert_best")
    parser.add_argument("--output_dir", type=str, default="outputs/distilbert")
    parser.add_argument("--output_name", type=str, default="evaluate_result.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-multilingual-cased",
                        help="用于初始化 Tokenizer 的预训练模型名")
    parser.add_argument("--device", type=str, default=None, help="计算设备（cpu/cuda/mps），默认自动检测")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
