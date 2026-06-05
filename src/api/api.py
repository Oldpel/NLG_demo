"""
模型工厂与统一推理接口（M4 核心胶水层）。

功能：
    - 自动根据配置加载 BiLSTM 或 DistilBERT 模型权重
    - 暴露统一的 predict / predict_batch 接口
    - 输入安全过滤（集成 security 模块）
    - 端到端延迟统计
    - 支持模型热切换

用法：
    from src.api.api import SentimentAnalyzer

    analyzer = SentimentAnalyzer(model_type="distilbert")
    result = analyzer.predict("这部电影太棒了！")
    print(result)
    # {'label': '正向积极评论', 'score': 0.9856, 'prob': 0.9856,
    #  'latency_ms': 42.3, 'model_type': 'distilbert'}
"""

import os
import sys
import json
import time
from typing import List, Dict

import torch

# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.dataset import DataProcessor
from src.models.bilstm.model import BiLSTMAttentionClassifier, predict as bilstm_predict
from src.models.distilbert.model import predict as distilbert_predict
from src.api.utils.security import sanitize_input, InputValidationError


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    统一情感分析器。

    自动根据 model_type 加载对应的模型权重和预处理工具，
    对外暴露一致的 predict() / predict_batch() 签名。
    """

    def __init__(
        self,
        model_type: str = "distilbert",
        model_path: str | None = None,
        device: str | None = None,
    ):
        """
        Args:
            model_type: "bilstm" 或 "distilbert"（不区分大小写）。
            model_path: 模型权重路径。None 则使用默认路径。
            device: 计算设备（"cpu"/"cuda"/"mps"）。None 则自动检测。
        """
        self.model_type = model_type.lower().strip()
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None  # BiLSTM 下是 vocab dict；DistilBERT 下是 AutoTokenizer
        self.processor = DataProcessor()  # 统一清洗工具

        self._load_model(model_path)

    # -----------------------------------------------------------------------
    # 设备解析
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_device(preference: str | None) -> torch.device:
        if preference:
            return torch.device(preference)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # -----------------------------------------------------------------------
    # 模型加载
    # -----------------------------------------------------------------------

    def _load_model(self, model_path: str | None):
        if self.model_type == "bilstm":
            self._load_bilstm(model_path)
        elif self.model_type == "distilbert":
            self._load_distilbert(model_path)
        else:
            raise ValueError(
                f"不支持的模型类型: {self.model_type!r}，"
                f'请使用 "bilstm" 或 "distilbert"'
            )

    def _load_bilstm(self, model_path: str | None):
        """加载 BiLSTM checkpoint（.pth 格式）。"""
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "checkpoints", "bilstm_best.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"BiLSTM checkpoint 不存在: {model_path}\n"
                f"请先运行训练脚本: python -m src.training.train_bilstm"
            )

        # 加载词表
        vocab_path = os.path.join(PROJECT_ROOT, "data", "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"词表不存在: {vocab_path}\n"
                f"请先运行: python -m src.data_processing.build_vocab"
            )

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.tokenizer = vocab

        # 构建模型
        model = BiLSTMAttentionClassifier(
            vocab_size=len(vocab),
            embed_dim=300,
            hidden_dim=128,
            num_classes=2,
            num_layers=2,
            dropout=0.5,
        )

        # 加载权重
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        print(f"[INFO] BiLSTM 模型已加载 | 路径: {model_path} | 设备: {self.device}")

    def _load_distilbert(self, model_path: str | None):
        """加载 DistilBERT checkpoint（save_pretrained 目录）。"""
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "checkpoints", "distilbert_best")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DistilBERT checkpoint 不存在: {model_path}\n"
                f"请先运行训练脚本: python -m src.training.train_distilbert"
            )

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        print(f"[INFO] DistilBERT 模型已加载 | 路径: {model_path} | 设备: {self.device}")

    # -----------------------------------------------------------------------
    # 推理接口
    # -----------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, float | str]:
        """
        单条文本情感分析。

        Args:
            text: 原始输入文本（会自动经过安全过滤和清洗）。

        Returns:
            {
                "label": "正向积极评论" | "负向消极评论",
                "score": float,          # 置信度（0~1）
                "prob": float,           # 正类概率（0~1）
                "latency_ms": float,     # 端到端耗时（毫秒）
                "model_type": str,       # 使用的模型类型
            }
        """
        start = time.perf_counter()

        # [Security] 输入安全过滤
        try:
            safe_text = sanitize_input(text)
        except InputValidationError as e:
            return {
                "label": "输入无效",
                "score": 0.0,
                "prob": 0.5,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "model_type": self.model_type,
                "error": e.reason,
            }

        # 统一清洗（OpenCC 繁简转换、去 HTML/URL）
        clean_text = self.processor.clean_text(safe_text)

        # 路由到对应模型
        if self.model_type == "bilstm":
            result = bilstm_predict(
                clean_text, self.model, self.tokenizer, str(self.device)
            )
        else:
            result = distilbert_predict(
                clean_text, self.model, self.tokenizer, str(self.device)
            )

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        result["model_type"] = self.model_type
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float | str]]:
        """
        批量文本情感分析。

        Args:
            texts: 原始文本列表。

        Returns:
            与输入顺序一一对应的结果列表。
            被安全过滤拦截的条目会在结果中标记 error 字段，不会中断整体流程。
        """
        results = []
        for i, text in enumerate(texts):
            result = self.predict(text)
            result["index"] = i
            results.append(result)
        return results

    # -----------------------------------------------------------------------
    # 模型切换
    # -----------------------------------------------------------------------

    def switch_model(self, model_type: str, model_path: str | None = None):
        """
        运行时切换模型（无需重新实例化 SentimentAnalyzer）。

        Args:
            model_type: "bilstm" 或 "distilbert"。
            model_path: 新的模型路径。None 则使用默认路径。
        """
        self.model_type = model_type.lower().strip()
        self.model = None
        self.tokenizer = None
        self._load_model(model_path)


# ---------------------------------------------------------------------------
# CLI 快速测试
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SentimentAnalyzer 快速测试")
    parser.add_argument("--model", type=str, default="distilbert", choices=["bilstm", "distilbert"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text", type=str, default="这部电影真的太精彩了，强烈推荐！")
    args = parser.parse_args()

    analyzer = SentimentAnalyzer(model_type=args.model, device=args.device)
    result = analyzer.predict(args.text)

    print("\n" + "=" * 50)
    print("推理结果")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k:15s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
