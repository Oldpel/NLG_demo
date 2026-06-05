#!/usr/bin/env python3
"""
NLG_demo 自动化流水线总闸（车间主任）。

功能：
    读取 configs/default_config.yaml，按顺序调度各功能模块：
    1. 数据下载与 EDA
    2. 数据清洗与隐私脱敏
    3. 词表构建
    4. BiLSTM 训练
    5. BiLSTM 评估
    6. DistilBERT 训练
    7. DistilBERT 评估

用法：
    # 一键全链路运行
    python main_pipeline.py --all

    # 仅运行数据准备阶段
    python main_pipeline.py --download --preprocess --vocab

    # 仅运行 BiLSTM 训练和评估
    python main_pipeline.py --train-bilstm --eval-bilstm

    # 仅运行 DistilBERT 训练和评估
    python main_pipeline.py --train-distilbert --eval-distilbert

    # 自定义配置文件
    python main_pipeline.py --config configs/my_config.yaml --all
"""

import os
import sys
import argparse
import subprocess

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def _load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    try:
        import yaml
    except ImportError:
        print("[ERROR] 请先安装 PyYAML: pip install pyyaml")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_module(module_path: str, extra_args: list[str] | None = None):
    """
    以子进程方式运行指定 Python 模块。
    使用 sys.executable 确保调用当前 Python 解释器。
    """
    cmd = [sys.executable, "-m", module_path]
    if extra_args:
        cmd.extend(extra_args)

    print("\n" + "=" * 60)
    print(f"[RUN] {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] 模块执行失败: {module_path} (exit={result.returncode})")
        sys.exit(result.returncode)


def stage_download_and_eda():
    """阶段 1：下载数据集并执行 EDA。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 1: 数据下载与 EDA")
    print("▓" * 60)
    _run_module("src.data_processing.download_and_eda")


def stage_preprocess():
    """阶段 2：数据清洗与隐私脱敏。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 2: 数据清洗与隐私脱敏")
    print("▓" * 60)
    _run_module("src.data_processing.preprocess")


def stage_build_vocab():
    """阶段 3：构建词表。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 3: 构建词表")
    print("▓" * 60)
    _run_module("src.data_processing.build_vocab")


def stage_train_bilstm(config: dict):
    """阶段 4：BiLSTM 训练。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 4: BiLSTM 模型训练")
    print("▓" * 60)
    mcfg = config["models"]["bilstm"]
    tcfg = config["training"]["bilstm"]
    out = config["output"]
    data = config["data"]

    extra = [
        "--train_csv", data["train_csv"],
        "--val_csv", data["val_csv"],
        "--vocab_path", data["vocab_path"],
        "--checkpoint_dir", out["checkpoint_dir"],
        "--checkpoint_name", "bilstm_best.pth",
        "--output_dir", out["bilstm_output_dir"],
        "--embed_dim", str(mcfg["embed_dim"]),
        "--hidden_dim", str(mcfg["hidden_dim"]),
        "--num_layers", str(mcfg["num_layers"]),
        "--dropout", str(mcfg["dropout"]),
        "--max_len", str(mcfg["max_len"]),
        "--batch_size", str(tcfg["batch_size"]),
        "--lr", str(tcfg["lr"]),
        "--epochs", str(tcfg["epochs"]),
        "--patience", str(tcfg["patience"]),
    ]
    if mcfg.get("pretrained_embed"):
        extra.extend(["--pretrained_embed", mcfg["pretrained_embed"]])
    if tcfg.get("device"):
        extra.extend(["--device", tcfg["device"]])

    _run_module("src.training.train_bilstm", extra)


def stage_eval_bilstm(config: dict):
    """阶段 5：BiLSTM 评估。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 5: BiLSTM 模型评估")
    print("▓" * 60)
    mcfg = config["models"]["bilstm"]
    out = config["output"]
    data = config["data"]

    extra = [
        "--test_csv", data["test_csv"],
        "--vocab_path", data["vocab_path"],
        "--checkpoint", os.path.join(out["checkpoint_dir"], "bilstm_best.pth"),
        "--output_dir", out["bilstm_output_dir"],
        "--output_name", out["evaluate_result"],
        "--embed_dim", str(mcfg["embed_dim"]),
        "--hidden_dim", str(mcfg["hidden_dim"]),
        "--num_layers", str(mcfg["num_layers"]),
        "--dropout", str(mcfg["dropout"]),
        "--max_len", str(mcfg["max_len"]),
    ]
    _run_module("src.evaluation.evaluate_bilstm", extra)


def stage_train_distilbert(config: dict):
    """阶段 6：DistilBERT 训练。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 6: DistilBERT 模型训练")
    print("▓" * 60)
    mcfg = config["models"]["distilbert"]
    tcfg = config["training"]["distilbert"]
    out = config["output"]
    data = config["data"]

    extra = [
        "--train_csv", data["train_csv"],
        "--val_csv", data["val_csv"],
        "--checkpoint_dir", out["checkpoint_dir"],
        "--checkpoint_name", "distilbert_best",
        "--output_dir", out["distilbert_output_dir"],
        "--pretrained_model", mcfg["pretrained_model"],
        "--max_len", str(mcfg["max_len"]),
        "--batch_size", str(tcfg["batch_size"]),
        "--lr", str(tcfg["lr"]),
        "--weight_decay", str(tcfg["weight_decay"]),
        "--warmup_ratio", str(tcfg["warmup_ratio"]),
        "--epochs", str(tcfg["epochs"]),
        "--patience", str(tcfg["patience"]),
    ]
    if tcfg.get("device"):
        extra.extend(["--device", tcfg["device"]])

    _run_module("src.training.train_distilbert", extra)


def stage_eval_distilbert(config: dict):
    """阶段 7：DistilBERT 评估。"""
    print("\n" + "▓" * 60)
    print("▓ 阶段 7: DistilBERT 模型评估")
    print("▓" * 60)
    mcfg = config["models"]["distilbert"]
    out = config["output"]
    data = config["data"]

    extra = [
        "--test_csv", data["test_csv"],
        "--checkpoint", os.path.join(out["checkpoint_dir"], "distilbert_best"),
        "--output_dir", out["distilbert_output_dir"],
        "--output_name", out["evaluate_result"],
        "--max_len", str(mcfg["max_len"]),
        "--pretrained_model", mcfg["pretrained_model"],
    ]
    _run_module("src.evaluation.evaluate_distilbert", extra)


def main():
    parser = argparse.ArgumentParser(
        description="NLG_demo 自动化流水线总闸",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main_pipeline.py --all                    # 全链路运行
  python main_pipeline.py --download --preprocess  # 仅数据准备
  python main_pipeline.py --train-bilstm --eval-bilstm
  python main_pipeline.py --config configs/custom.yaml --all
        """,
    )
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--all", action="store_true",
                        help="运行完整流水线（下载 → 清洗 → 词表 → 训练 → 评估）")
    parser.add_argument("--download", action="store_true",
                        help="阶段 1: 下载数据集并 EDA")
    parser.add_argument("--preprocess", action="store_true",
                        help="阶段 2: 数据清洗与脱敏")
    parser.add_argument("--vocab", action="store_true",
                        help="阶段 3: 构建词表")
    parser.add_argument("--train-bilstm", action="store_true",
                        help="阶段 4: BiLSTM 训练")
    parser.add_argument("--eval-bilstm", action="store_true",
                        help="阶段 5: BiLSTM 评估")
    parser.add_argument("--train-distilbert", action="store_true",
                        help="阶段 6: DistilBERT 训练")
    parser.add_argument("--eval-distilbert", action="store_true",
                        help="阶段 7: DistilBERT 评估")

    args = parser.parse_args()

    # 如果没有指定任何阶段，打印帮助信息
    if not any([
        args.all, args.download, args.preprocess, args.vocab,
        args.train_bilstm, args.eval_bilstm,
        args.train_distilbert, args.eval_distilbert,
    ]):
        parser.print_help()
        sys.exit(0)

    # 加载配置
    config = _load_config(args.config)
    print(f"[INFO] 已加载配置: {args.config}")

    # 阶段调度
    if args.all or args.download:
        stage_download_and_eda()
    if args.all or args.preprocess:
        stage_preprocess()
    if args.all or args.vocab:
        stage_build_vocab()
    if args.all or args.train_bilstm:
        stage_train_bilstm(config)
    if args.all or args.eval_bilstm:
        stage_eval_bilstm(config)
    if args.all or args.train_distilbert:
        stage_train_distilbert(config)
    if args.all or args.eval_distilbert:
        stage_eval_distilbert(config)

    print("\n" + "▓" * 60)
    print("▓ [OK] 流水线执行完毕！")
    print("▓" * 60)


if __name__ == "__main__":
    main()
