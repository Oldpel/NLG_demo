"""
DistilBERT 微调训练脚本。

功能：
    - 基于 transformers 加载 DistilBERT + 分类头
    - AdamW + Linear Warmup + Cosine Annealing
    - Early Stopping + 最佳模型保存（save_pretrained 完整目录）
    - 显存监控（兼容 CPU/GPU）
    - 训练曲线保存

用法：
    cd NLG
    python -m src.training.train_distilbert --device cpu

    # 使用 GPU
    python -m src.training.train_distilbert --device cuda --batch_size 32
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.dataset import SentimentDataset
from src.models.distilbert.model import create_model_and_tokenizer


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

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


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """返回 (loss, accuracy)"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if outputs.loss is not None else criterion(outputs.logits, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

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


def print_memory_stats(device: torch.device):
    """打印显存占用（GPU 环境下有效）。"""
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"[MEM] 显存占用: {alloc:.1f} MB / 预留: {reserved:.1f} MB")


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train(args):
    device = get_device(preference=args.device)
    print(f"[INFO] 使用设备: {device}")

    # 目录（基于项目根目录）
    checkpoint_dir = (
        args.checkpoint_dir if os.path.isabs(args.checkpoint_dir) else os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    )
    output_dir = (
        args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_ROOT, args.output_dir)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 模型 & Tokenizer
    print(f"[INFO] 加载模型: {args.pretrained_model}")
    model, tokenizer = create_model_and_tokenizer(
        num_labels=2,
        pretrained_model_name=args.pretrained_model,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 可训练参数: {total_params:,}")

    # 数据集
    train_csv = os.path.join(PROJECT_ROOT, args.train_csv)
    val_csv = os.path.join(PROJECT_ROOT, args.val_csv)

    train_ds = SentimentDataset(
        csv_path=train_csv,
        model_type="distilbert",
        max_len=args.max_len,
        pretrained_model_name=args.pretrained_model,
    )
    val_ds = SentimentDataset(
        csv_path=val_csv,
        model_type="distilbert",
        max_len=args.max_len,
        pretrained_model_name=args.pretrained_model,
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

    # 优化器：AdamW + Layer-wise LR Decay（可选）
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    # Scheduler: Linear Warmup + Cosine Annealing
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"[INFO] 总步数: {total_steps} | Warmup 步数: {warmup_steps}")

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)
            preds = outputs.logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += input_ids.size(0)

            # 更新进度条后缀，显示实时 loss 与 acc
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_correct / train_total:.4f}",
            })

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        val_loss, val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"LR={current_lr:.2e} | Time={epoch_time:.1f}s"
        )
        print_memory_stats(device)

        # ---- Checkpoint & Early Stopping ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存完整模型（含 config.json + pytorch_model.bin）
            save_path = os.path.join(checkpoint_dir, args.checkpoint_name)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            # 额外保存训练元信息
            meta = {
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "args": vars(args),
            }
            with open(os.path.join(save_path, "training_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  [SAVE] 验证集 Acc 提升，保存模型 -> {save_path}")
        else:
            patience_counter += 1
            print(f"  [PAT] 早停计数: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\n[INFO] 早停触发，最佳验证集 Acc={best_val_acc:.4f}")
            break

    # 保存训练曲线
    plot_path = os.path.join(output_dir, "distilbert_training_curves.png")
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
    parser = argparse.ArgumentParser(description="DistilBERT 微调训练脚本")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/val.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="distilbert_best")
    parser.add_argument("--output_dir", type=str, default="outputs/distilbert")
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--device", type=str, default=None, help="计算设备（cpu/cuda/mps），默认自动检测")

    # 模型超参
    parser.add_argument("--max_len", type=int, default=128)

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 步数占总步数比例")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
