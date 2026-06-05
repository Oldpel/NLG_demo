"""
BiLSTM-Attention 独立评估脚本。

功能：
    - 加载最佳 checkpoint
    - 在测试集上评估
    - 输出 Accuracy, Precision, Recall, F1 到 JSON

用法：
    cd NLG
    python -m src.evaluation.evaluate_bilstm

    # 指定 checkpoint
    python -m src.evaluation.evaluate_bilstm --checkpoint checkpoints/bilstm_best.pth
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.dataset import SentimentDataset
from src.models.bilstm.model import BiLSTMAttentionClassifier


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
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    lengths = (input_ids != 0).sum(dim=1)
    return {"input_ids": input_ids, "labels": labels, "lengths": lengths}


def evaluate(args):
    device = get_device(preference=args.device)
    print(f"[INFO] 使用设备: {device}")

    # 加载词表
    vocab_path = os.path.join(PROJECT_ROOT, args.vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # 测试集
    test_csv = os.path.join(PROJECT_ROOT, args.test_csv)
    test_ds = SentimentDataset(
        csv_path=test_csv,
        model_type="bilstm",
        max_len=args.max_len,
        vocab_path=vocab_path,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"[INFO] 测试集样本数: {len(test_ds)}")

    # 构建模型
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # 加载 checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] 加载 checkpoint: {ckpt_path} (Epoch {checkpoint.get('epoch', '?')})")

    # 推理
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(x, lengths)
            preds = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(y.cpu().tolist())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    result = {
        "model": "BiLSTM-Attention",
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
    parser = argparse.ArgumentParser(description="BiLSTM-Attention 评估脚本")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bilstm_best.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/bilstm")
    parser.add_argument("--output_name", type=str, default="evaluate_result.json")
    parser.add_argument("--device", type=str, default=None, help="计算设备（cpu/cuda/mps），默认自动检测")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
