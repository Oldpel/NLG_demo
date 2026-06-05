"""
BiLSTM-Attention 训练脚本。

功能：
    - 加载 data_processing 生成的词表与数据
    - 支持 CPU / CUDA / MPS 自适应
    - Early Stopping + 最佳模型保存
    - 训练曲线（Loss / Accuracy）保存为 PNG

用法：
    cd NLG
    python -m src.training.train_bilstm

    # 使用预训练词向量
    python -m src.training.train_bilstm --pretrained_embed path/to/tencent.vec
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 路径处理：确保能 import src 下的模块
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.dataset import SentimentDataset
from src.models.bilstm.model import BiLSTMAttentionClassifier


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_device(preference: str | None = None) -> torch.device:
    """自动选择最优可用设备。若 preference 指定则优先使用。"""
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch):
    """
    将 list[dict] 合并为 dict[tensor]，并计算真实长度。
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    lengths = (input_ids != 0).sum(dim=1)  # PAD=0
    return {"input_ids": input_ids, "labels": labels, "lengths": lengths}


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """返回 (loss, accuracy)"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, save_path: str):
    """绘制并保存 Loss / Accuracy 曲线。"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-s", label="Val Acc", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] 训练曲线已保存: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train(args):
    # 设备
    device = get_device(preference=args.device)
    print(f"[INFO] 使用设备: {device}")

    # 目录（基于项目根目录解析，避免在不同子目录运行时路径漂移）
    checkpoint_dir = (
        args.checkpoint_dir if os.path.isabs(args.checkpoint_dir) else os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    )
    output_dir = (
        args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_ROOT, args.output_dir)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 加载词表
    vocab_path = os.path.join(PROJECT_ROOT, args.vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[INFO] 词表大小: {len(vocab)}")

    # 数据集
    train_csv = os.path.join(PROJECT_ROOT, args.train_csv)
    val_csv = os.path.join(PROJECT_ROOT, args.val_csv)

    train_ds = SentimentDataset(
        csv_path=train_csv,
        model_type="bilstm",
        max_len=args.max_len,
        vocab_path=vocab_path,
    )
    val_ds = SentimentDataset(
        csv_path=val_csv,
        model_type="bilstm",
        max_len=args.max_len,
        vocab_path=vocab_path,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"[INFO] 训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    # 模型
    pretrained_path = args.pretrained_embed
    if pretrained_path:
        pretrained_path = os.path.join(PROJECT_ROOT, pretrained_path)

    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pretrained_embed_path=pretrained_path if (pretrained_path and os.path.exists(pretrained_path)) else None,
        vocab=vocab,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 可训练参数: {total_params:,}")

    # 优化器 & 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 训练状态
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -1.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()

            # 梯度裁剪，防止 LSTM 爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        val_loss, val_acc = evaluate(model, val_loader, device)

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # LR 调度
        scheduler.step(val_acc)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"Time={epoch_time:.1f}s"
        )

        # ---- Checkpoint & Early Stopping ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, args.checkpoint_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  [SAVE] 验证集 Acc 提升，保存 checkpoint -> {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  [PAT] 早停计数: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\n[INFO] 早停触发，最佳验证集 Acc={best_val_acc:.4f}")
            break

    # 保存训练曲线
    plot_path = os.path.join(output_dir, "training_curves.png")
    plot_curves(history, plot_path)

    print("\n" + "=" * 60)
    print("训练完成")
    print(f"最佳验证集 Acc: {best_val_acc:.4f}")
    print(f"模型保存路径: {os.path.join(checkpoint_dir, args.checkpoint_name)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BiLSTM-Attention 训练脚本")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="训练集 CSV")
    parser.add_argument("--val_csv", type=str, default="data/val.csv", help="验证集 CSV")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json", help="词表路径")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="checkpoint 目录")
    parser.add_argument("--checkpoint_name", type=str, default="bilstm_best.pth", help="checkpoint 文件名")
    parser.add_argument("--output_dir", type=str, default="outputs/bilstm", help="输出目录（训练曲线等）")
    parser.add_argument("--pretrained_embed", type=str, default=None, help="预训练词向量路径（可选）")
    parser.add_argument("--device", type=str, default=None, help="计算设备（cpu/cuda/mps），默认自动检测")

    # 模型超参
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=128)

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
