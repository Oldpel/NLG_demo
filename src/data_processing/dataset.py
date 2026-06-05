import os
import re
import json
import torch
import pandas as pd
import jieba
# 需确保环境中已安装 opencc: pip install opencc
from opencc import OpenCC
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DataProcessor:
    """数据预处理工具类，解耦清洗逻辑以便在 EDA 和词表构建中复用"""

    def __init__(self, stopwords_path: str | None = None):
        self.cc = OpenCC('t2s')
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())

    def clean_text(self, text: str, remove_stopwords: bool = False) -> str:
        # [Security] 注意：如果系统直接暴露给公网，恶意的超长 HTML 标签或连续特殊字符可能导致正则表达式拒绝服务攻击 (ReDoS)，耗尽 CPU 资源。建议在正则前对 len(text) 做硬性截断（例如 max 5000 字符）。
        text = str(text)
        # 安全截断，防止超长文本导致正则 ReDoS
        if len(text) > 5000:
            text = text[:5000]
        text = re.sub(r'<.*?>', '', text)       # 去除HTML
        text = re.sub(r'http\S+', '', text)     # 去除URL
        text = self.cc.convert(text)            # 繁简转换
        if remove_stopwords and self.stopwords:
            words = jieba.lcut(text)
            words = [w for w in words if w.strip() and w.strip() not in self.stopwords]
            text = ''.join(words)
        return text

class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, model_type: str = "bilstm", max_len: int = 128,
                 vocab_path: str = "data/vocab.json", pretrained_model_name: str = "distilbert-base-multilingual-cased"):
        """
        model_type: "bilstm" 或 "distilbert"
        """
        # [Security] 如果 csv_path 或 vocab_path 是由上游接口动态传入的参数，存在路径穿越 (Path Traversal) 风险。必须在实例化前确保传入的是可信的受限相对路径（如锁定在 ./data/ 目录下）。
        self.data = pd.read_csv(csv_path)
        self.model_type = model_type
        self.max_len = max_len
        self.processor = DataProcessor()

        # 根据模型类型预加载必要的依赖，避免在 __getitem__ 中重复加载
        if self.model_type == "distilbert":
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        elif self.model_type == "bilstm":
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        raw_text = row['text']
        label = int(row['label'])

        # 清洗数据
        clean_text = self.processor.clean_text(raw_text)

        # 路由到对应的处理管线
        if self.model_type == "distilbert":
            encoded = self.tokenizer(
                clean_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                "input_ids": encoded['input_ids'].squeeze(0),
                "attention_mask": encoded['attention_mask'].squeeze(0),
                "label": label
            }
        else:
            # BiLSTM 路径的常规分词与填充
            words = jieba.lcut(clean_text)
            unk_id = self.vocab.get("<UNK>", 0)
            pad_id = self.vocab.get("<PAD>", 0)

            input_ids = [self.vocab.get(w, unk_id) for w in words][:self.max_len]
            input_ids += [pad_id] * (self.max_len - len(input_ids))

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "label": label
            }
