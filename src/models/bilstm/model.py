"""
BiLSTM + SelfAttention + MLP 情感分类模型。

输出规范（M4 直接调用）:
    - BiLSTMAttentionClassifier: PyTorch nn.Module
    - predict(text, model, vocab, device): dict

用法:
    from src.models.bilstm.model import BiLSTMAttentionClassifier, predict
"""

import os
import re
import json
from typing import Dict

import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention 模块
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    基于可学习 context vector 的 Attention 机制。

    参考: Yang et al. "Hierarchical Attention Networks for Document Classification"
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim*2)
            mask:        (batch, seq_len)  bool，True 表示有效位置

        Returns:
            context: (batch, hidden_dim*2)
            weights: (batch, seq_len)
        """
        energy = self.projection(lstm_output).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        weights = F.softmax(energy, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden*2)
        return context, weights


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_embed_path: str | None = None,
        vocab: dict | None = None,
    ):
        """
        Args:
            vocab_size: 词表大小
            embed_dim: 词向量维度
            hidden_dim: BiLSTM 隐藏层维度
            num_classes: 类别数（二分类=2）
            num_layers: LSTM 层数
            dropout: Dropout 概率
            pretrained_embed_path: 预训练词向量路径（Tencent/FastText 文本格式），None 则随机初始化
            vocab: 当前词表 dict{word: idx}，用于对齐预训练词向量
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 加载预训练词向量
        if pretrained_embed_path and os.path.exists(pretrained_embed_path) and vocab:
            self._load_pretrained_embeddings(pretrained_embed_path, vocab, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _load_pretrained_embeddings(self, path: str, vocab: dict, embed_dim: int):
        """加载 Tencent/FastText 格式的文本词向量，并与当前 vocab 对齐。"""
        print(f"[INFO] 正在加载预训练词向量: {path}")
        pretrained = {}
        first_line = True
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if first_line:
                    first_line = False
                    continue  # 跳过首行（词数 维度）
                parts = line.strip().split()
                if len(parts) < embed_dim + 1:
                    continue
                word = parts[0]
                vec = torch.tensor([float(x) for x in parts[1 : embed_dim + 1]], dtype=torch.float)
                pretrained[word] = vec

        # 对齐
        embedding_matrix = self.embedding.weight.data
        hit = 0
        for word, idx in vocab.items():
            if word in pretrained:
                embedding_matrix[idx] = pretrained[word]
                hit += 1

        print(f"[INFO] 预训练词向量对齐完成: {hit}/{len(vocab)} 词命中 ({hit / len(vocab):.1%})")

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        Args:
            input_ids: (batch, seq_len)
            lengths:   (batch,) 每个样本的真实长度，用于构造 mask

        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)

        # 构造 mask
        if lengths is not None:
            batch_size, seq_len = input_ids.size()
            mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = input_ids != 0  # 默认 PAD=0

        context, attn_weights = self.attention(lstm_out, mask)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits


# ---------------------------------------------------------------------------
# 推理接口（M4 直接调用）
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """简易清洗，与 M1 的 DataProcessor 保持一致。"""
    text = str(text)
    if len(text) > 5000:
        text = text[:5000]
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    return text


def predict(
    text: str,
    model: nn.Module,
    tokenizer: dict,  # 对 BiLSTM 路径，tokenizer 就是 vocab dict
    device: str = "cpu",
    max_len: int = 128,
) -> dict:
    """
    单条文本推理接口。

    Args:
        text: 原始输入文本
        model: BiLSTMAttentionClassifier 实例
        tokenizer: 词表 dict {word: idx}
        device: 计算设备
        max_len: 最大序列长度

    Returns:
        {"label": "正向积极评论" | "负向消极评论", "score": float, "prob": float}
    """
    model.eval()
    clean = _clean_text(text)
    words = jieba.lcut(clean)

    vocab = tokenizer
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)

    input_ids = [vocab.get(w, unk_id) for w in words][:max_len]
    length = len(input_ids)
    input_ids += [pad_id] * (max_len - len(input_ids))

    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = F.softmax(logits, dim=1)  # (1, 2)

    prob = probs[0][1].item()  # 正类概率
    pred = logits.argmax(dim=1).item()

    label = "正向积极评论" if pred == 1 else "负向消极评论"
    score = prob if pred == 1 else (1 - prob)

    return {
        "label": label,
        "score": round(score, 4),
        "prob": round(prob, 4),
    }
