# NLG_demo — 中文文本情感分析系统

基于 **BiLSTM-Attention** 与 **DistilBERT** 的双路线中文情感分析系统，覆盖数据工程、模型训练、评估对比到 Gradio 可视化交互的全生命周期。

---

## 1. 系统架构

```
NLG_demo/
├── configs/                 # 统一配置中心
├── src/
│   ├── data_processing/     # M1 数据工程：下载、EDA、清洗、脱敏、词表
│   ├── models/              # M2/M3 模型定义
│   │   ├── bilstm/          #   BiLSTM + SelfAttention + MLP
│   │   └── distilbert/      #   DistilBERT 微调
│   ├── training/            # M2/M3 训练脚本
│   ├── evaluation/          # M2/M3 评估脚本
│   └── api/                 # M4 系统集成：模型工厂 + Gradio UI
├── tests/                   # 单元测试与边界测试
├── main_pipeline.py         # 自动化总闸：一键运行全链路
├── data/                    # 数据目录（受 .gitignore 保护）
├── checkpoints/             # 模型权重（受 .gitignore 保护）
└── outputs/                 # 训练曲线与评估报告
```

### 数据流

```
HuggingFace 数据集
        │
        ▼
src.data_processing.download_and_eda  ──▶ data/raw_*
        │
        ▼
src.data_processing.preprocess        ──▶ data/train.csv / val.csv / test.csv
        │                                              （含隐私脱敏）
        ▼
src.data_processing.build_vocab       ──▶ data/vocab.json
        │
        ├──▶ src.training.train_bilstm     ──▶ checkpoints/bilstm_best.pth
        │                                         outputs/bilstm/training_curves.png
        │
        └──▶ src.training.train_distilbert ──▶ checkpoints/distilbert_best/
                                                  outputs/distilbert/distilbert_training_curves.png
        │
        ├──▶ src.evaluation.evaluate_bilstm     ──▶ outputs/bilstm/evaluate_result.json
        │
        └──▶ src.evaluation.evaluate_distilbert ──▶ outputs/distilbert/evaluate_result.json
        │
        ▼
src.api.app  ──▶ Gradio Web UI（单文本分析 + CSV 批量分析）
```

---

## 2. 环境安装

### 2.1 创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 2.2 安装依赖

```bash
pip install torch transformers datasets pandas jieba opencc scikit-learn matplotlib tqdm gradio pyyaml
```

> **注意**：DistilBERT 训练需要 `transformers>=4.30`，`torch>=2.0` 兼容性最佳。

---

## 3. 快速开始

### 3.1 一键全链路运行

```bash
python main_pipeline.py --all
```

这条命令会按顺序执行：

1. 下载 HuggingFace 数据集并执行 EDA
2. 数据清洗 + 隐私脱敏
3. 构建词表
4. 训练 BiLSTM
5. 评估 BiLSTM
6. 训练 DistilBERT
7. 评估 DistilBERT

> 完整运行需要数小时（取决于硬件）。如果已有数据或模型权重，可只运行指定阶段。

### 3.2 只运行数据准备

```bash
python main_pipeline.py --download --preprocess --vocab
```

### 3.3 只运行 BiLSTM 路径

```bash
python main_pipeline.py --train-bilstm --eval-bilstm
```

### 3.4 只运行 DistilBERT 路径

```bash
python main_pipeline.py --train-distilbert --eval-distilbert
```

### 3.5 自定义配置

```bash
python main_pipeline.py --config configs/my_config.yaml --all
```

---

## 4. 分阶段详细使用

### 4.1 数据工程（M1）

#### 下载数据集并执行 EDA

```bash
python -m src.data_processing.download_and_eda
```

输出：

- `data/raw_chnsenticorp/*.csv`
- `data/raw_chinese_sentiment/*.csv`
- `data/eda/eda_report.txt`
- `data/eda/length_distribution_*.png`

#### 数据清洗与隐私脱敏

```bash
python -m src.data_processing.preprocess
```

输出：

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

清洗流程：

1. 正则去除 HTML 标签
2. 正则去除 URL
3. OpenCC 繁简转换
4. **隐私脱敏**：手机号 → `[PHONE]`，邮箱 → `[EMAIL]`，身份证 → `[ID_CARD]`
5. 去停用词（可选，BiLSTM 路径默认启用）
6. 过滤空文本

#### 构建词表

```bash
python -m src.data_processing.build_vocab
```

输出：

- `data/vocab.json`

### 4.2 模型训练（M2 / M3）

#### BiLSTM 训练

```bash
python -m src.training.train_bilstm
```

常用参数：

```bash
python -m src.training.train_bilstm \
    --device cuda \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-3 \
    --pretrained_embed path/to/tencent.vec
```

输出：

- `checkpoints/bilstm_best.pth`
- `outputs/bilstm/training_curves.png`

#### DistilBERT 训练

```bash
python -m src.training.train_distilbert
```

常用参数：

```bash
python -m src.training.train_distilbert \
    --device cuda \
    --epochs 10 \
    --batch_size 32 \
    --lr 2e-5
```

输出：

- `checkpoints/distilbert_best/`
- `outputs/distilbert/distilbert_training_curves.png`

### 4.3 模型评估（M2 / M3）

```bash
python -m src.evaluation.evaluate_bilstm
python -m src.evaluation.evaluate_distilbert
```

输出 JSON 格式（一致）：

```json
{
  "model": "BiLSTM-Attention",
  "checkpoint": "checkpoints/bilstm_best.pth",
  "dataset": "data/test.csv",
  "num_samples": 5200,
  "metrics": {
    "accuracy": 0.9231,
    "precision": 0.9187,
    "recall": 0.9294,
    "f1_score": 0.9240
  }
}
```

---

## 5. 配置中心

所有路径、超参、输出目录统一收敛到 [`configs/default_config.yaml`](configs/default_config.yaml)。

### 关键配置项

```yaml
data:
  raw_dir: "data/raw"
  train_csv: "data/train.csv"
  val_csv: "data/val.csv"
  test_csv: "data/test.csv"
  vocab_path: "data/vocab.json"

models:
  bilstm:
    embed_dim: 300
    hidden_dim: 128
    num_layers: 2
    dropout: 0.5
    max_len: 128

  distilbert:
    pretrained_model: "distilbert-base-multilingual-cased"
    max_len: 128

training:
  bilstm:
    batch_size: 64
    lr: 0.001
    epochs: 30
    patience: 5

  distilbert:
    batch_size: 32
    lr: 0.00002
    weight_decay: 0.01
    warmup_ratio: 0.1
    epochs: 10
    patience: 3

output:
  checkpoint_dir: "checkpoints"
  bilstm_output_dir: "outputs/bilstm"
  distilbert_output_dir: "outputs/distilbert"
```

> **安全提醒**：配置文件中严禁明文写入 API Token、数据库密码、云服务密钥等敏感信息。

---

## 6. API 与 Gradio 服务（M4）

### 6.1 编程式 API

```python
from src.api.api import SentimentAnalyzer

# 加载 DistilBERT
analyzer = SentimentAnalyzer(model_type="distilbert")

# 单条推理
result = analyzer.predict("这部电影真的太精彩了，强烈推荐！")
print(result)
# {
#   'label': '正向积极评论',
#   'score': 0.9856,
#   'prob': 0.9856,
#   'latency_ms': 42.3,
#   'model_type': 'distilbert'
# }

# 批量推理
results = analyzer.predict_batch(["很好", "很差", "一般般"])

# 运行时切换模型
analyzer.switch_model("bilstm")
```

### 6.2 启动 Gradio Web UI

```bash
pip install gradio
python -m src.api.app
```

打开浏览器访问：`http://localhost:7860`

界面功能：

- **单文本分析**：输入中文评论 → 实时情感标签 + 置信度分布 + 推理耗时
- **批量分析**：上传 CSV（需含 `text` 列）→ 批量推理 → 下载结果
- **模型切换**：Dropdown 实时切换 BiLSTM / DistilBERT

自定义端口：

```bash
python -m src.api.app --port 8080
```

生成公网共享链接：

```bash
python -m src.api.app --share
```

---

## 7. 测试

### 7.1 安全过滤与边界测试

```bash
python -m tests.test_api
```

覆盖场景：

- 正常中文文本、中英文混合、数字代码
- 空字符串、纯空白、纯标点
- 超长文本截断（> 2000 字符）
- XSS 脚本注入拦截（`<script>`、`javascript:`、`on事件=` 等）
- 控制字符过滤
- 批量推理中的部分无效输入

### 7.2 使用 pytest

```bash
python -m pytest tests/test_api.py -v
```

---

## 8. 安全设计

| 层级 | 措施 |
|------|------|
| **代码提交** | `.gitignore` 拦截 `data/`、`checkpoints/`、环境文件、缓存 |
| **数据脱敏** | 手机号 → `[PHONE]`，邮箱 → `[EMAIL]`，身份证 → `[ID_CARD]` |
| **输入过滤** | 长度截断、控制字符去除、XSS 注入检测、空/纯标点拦截 |
| **路径安全** | 禁止路径穿越，`dataset.py` 中路径由配置中心统一管控 |
| **凭据管理** | 配置文件中不写死任何密钥，全部通过环境变量注入 |

---

## 9. 项目分工

参考 [`实训课题任务分工与协作规范.md`](实训课题任务分工与协作规范.md)。

| 模块 | 目录 | 核心交付物 |
|------|------|-----------|
| **M1 数据工程** | `src/data_processing/` | 清洗数据、词表、EDA 报告 |
| **M2 传统模型** | `src/models/bilstm/` + `src/training/` + `src/evaluation/` | BiLSTM checkpoint、训练曲线、评估 JSON |
| **M3 预训练模型** | `src/models/distilbert/` + `src/training/` + `src/evaluation/` | DistilBERT checkpoint、训练曲线、评估 JSON |
| **M4 系统集成** | `src/api/` + `tests/` | Gradio UI、统一 API、测试报告 |

---

## 10. 许可证

本项目仅用于学术研究与教学实训。模型权重与数据集请遵守各自原始许可证。

---

## 11. 常见问题

### Q1: 运行 `main_pipeline.py` 时提示缺少 `pyyaml`

```bash
pip install pyyaml
```

### Q2: DistilBERT 训练时显存溢出

在 `configs/default_config.yaml` 中减小 `batch_size`：

```yaml
training:
  distilbert:
    batch_size: 16   # 或 8
```

### Q3: Gradio 启动失败

```bash
pip install gradio
# 若端口被占用
python -m src.api.app --port 8080
```

### Q4: 模型权重不存在导致 API 报错

请先完成对应模型的训练：

```bash
python main_pipeline.py --train-bilstm
# 或
python main_pipeline.py --train-distilbert
```

---

*项目基于 Python 3.10+、PyTorch 2.0+、Transformers 4.30+ 构建。*
