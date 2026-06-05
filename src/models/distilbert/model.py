"""
DistilBERT 情感分类模型。

输出规范（M4 直接调用）:
    - create_model_and_tokenizer(num_labels=2)
    - predict(text, model, tokenizer, device): dict

用法:
    from src.models.distilbert.model import create_model_and_tokenizer, predict
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def create_model_and_tokenizer(num_labels: int = 2, pretrained_model_name: str = "distilbert-base-multilingual-cased"):
    """
    加载预训练 DistilBERT 与对应 Tokenizer，并添加分类头。

    Args:
        num_labels: 分类数（二分类=2）
        pretrained_model_name: HuggingFace 模型名

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
    )
    return model, tokenizer


def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    device: str = "cpu",
    max_len: int = 128,
) -> dict:
    """
    单条文本推理接口（与 M2 BiLSTM 完全一致）。

    Args:
        text: 原始输入文本
        model: DistilBERT ForSequenceClassification 实例
        tokenizer: AutoTokenizer 实例
        device: 计算设备
        max_len: 最大序列长度

    Returns:
        {"label": "正向积极评论" | "负向消极评论", "score": float, "prob": float}
    """
    model.eval()

    inputs = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, num_labels)
        probs = F.softmax(logits, dim=1)

    prob = probs[0][1].item()  # 正类概率
    pred = logits.argmax(dim=1).item()

    label = "正向积极评论" if pred == 1 else "负向消极评论"
    score = prob if pred == 1 else (1 - prob)

    return {
        "label": label,
        "score": round(score, 4),
        "prob": round(prob, 4),
    }
