"""
Gradio 可视化交互界面（M4 前端）。

功能：
    - 面板 A：单文本输入 → 实时情感标签 + 置信度 + 延迟显示
    - 面板 B：CSV 拖拽上传 → 批量推理 → 结果下载
    - 模型切换：Dropdown 选择 bilstm / distilbert（运行时热切换）
    - 安全防护：集成 sanitize_input 对前端输入做过滤

用法：
    # 安装 Gradio（若未安装）
    pip install gradio

    # 启动服务
    python -m src.api.app

    # 指定端口
    python -m src.api.app --port 8080
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    import gradio as gr
except ImportError:
    print("[ERROR] Gradio 未安装。请运行: pip install gradio")
    print("       若仅需无 UI 的 API 服务，请使用 src.api.api 模块。")
    sys.exit(1)

from src.api.api import SentimentAnalyzer
from src.api.utils.security import InputValidationError


# ---------------------------------------------------------------------------
# 全局分析器实例（延迟加载，支持模型热切换）
# ---------------------------------------------------------------------------

_analyzers: dict[str, SentimentAnalyzer] = {}


def _get_analyzer(model_type: str) -> SentimentAnalyzer:
    """获取或创建指定类型的分析器，支持缓存复用。"""
    global _analyzers
    model_type = model_type.lower().strip()

    if model_type not in _analyzers:
        try:
            _analyzers[model_type] = SentimentAnalyzer(model_type=model_type)
        except FileNotFoundError as e:
            raise gr.Error(f"模型加载失败: {e}")

    return _analyzers[model_type]


# ---------------------------------------------------------------------------
# 单文本推理
# ---------------------------------------------------------------------------

def predict_single(text: str, model_type: str):
    """
    单条文本推理，返回 Gradio 友好的输出格式。

    Returns:
        (label_text, prob_bar, latency_text, detail_text)
    """
    try:
        analyzer = _get_analyzer(model_type)
    except gr.Error:
        return "模型未加载", {("错误", 1.0)}, "N/A", "请先训练模型"

    result = analyzer.predict(text)

    # 处理安全过滤拦截的情况
    if result.get("error"):
        return (
            f"⚠️ 输入被拦截: {result['error']}",
            {("无效", 1.0)},
            f"{result.get('latency_ms', 0):.1f} ms",
            str(result),
        )

    label = result["label"]
    prob = result["prob"]
    score = result["score"]
    latency = result.get("latency_ms", 0)

    # 构建 Label 组件需要的概率分布
    if label == "正向积极评论":
        prob_bar = {"正向积极评论": prob, "负向消极评论": 1 - prob}
        confidence_str = f"正向置信度: {prob:.2%}"
    else:
        prob_bar = {"正向积极评论": 1 - score, "负向消极评论": score}
        confidence_str = f"负向置信度: {score:.2%}"

    detail = (
        f"情感标签: {label}\n"
        f"正类概率: {prob:.4f}\n"
        f"置信度:   {score:.4f}\n"
        f"推理耗时: {latency:.1f} ms\n"
        f"模型:     {result['model_type']}"
    )

    return label, prob_bar, f"{latency:.1f} ms", detail


# ---------------------------------------------------------------------------
# 批量推理
# ---------------------------------------------------------------------------

def predict_batch(file, model_type: str):
    """
    CSV 批量推理。

    Args:
        file: Gradio File 对象（上传的 CSV）。
        model_type: 模型类型。

    Returns:
        (result_dataframe, info_text, download_path)
    """
    if file is None:
        return None, "请先上传 CSV 文件", None

    try:
        analyzer = _get_analyzer(model_type)
    except gr.Error:
        return None, "模型未加载，请先训练模型", None

    import pandas as pd

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return None, f"CSV 读取失败: {e}", None

    if "text" not in df.columns:
        return None, "CSV 中未找到 'text' 列，请确保列名为 'text'", None

    texts = df["text"].astype(str).tolist()
    results = analyzer.predict_batch(texts)

    # 构建结果 DataFrame
    output_df = pd.DataFrame({
        "text": df["text"],
        "label": [r["label"] for r in results],
        "prob": [round(r["prob"], 4) for r in results],
        "score": [round(r["score"], 4) for r in results],
        "latency_ms": [round(r.get("latency_ms", 0), 2) for r in results],
    })

    # 统计信息
    valid_count = sum(1 for r in results if not r.get("error"))
    error_count = len(results) - valid_count
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / max(len(results), 1)
    pos_count = sum(1 for r in results if r["label"] == "正向积极评论")
    neg_count = valid_count - pos_count

    info = (
        f"总计: {len(results)} 条 | "
        f"成功: {valid_count} 条 | "
        f"拦截: {error_count} 条\n"
        f"正向: {pos_count} 条 | "
        f"负向: {neg_count} 条 | "
        f"平均延迟: {avg_latency:.1f} ms"
    )

    # 保存结果 CSV
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "batch_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "batch_result.csv")
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_df, info, output_path


# ---------------------------------------------------------------------------
# Gradio 界面构建
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="中文情感分析系统",
        theme=gr.themes.Soft(),
        css="""
        .footer { text-align: center; color: #888; margin-top: 20px; }
        """,
    ) as demo:
        gr.Markdown("""
        # 🎭 中文文本情感分析系统
        基于 **BiLSTM-Attention** 与 **DistilBERT** 双模型的情感分析服务
        """)

        # -------------------------------------------------------------------
        # 全局控制：模型选择
        # -------------------------------------------------------------------
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["distilbert", "bilstm"],
                value="distilbert",
                label="🤖 选择模型",
                info="DistilBERT（精度高）或 BiLSTM（轻量快）",
            )

        # -------------------------------------------------------------------
        # Tab 1: 单文本分析
        # -------------------------------------------------------------------
        with gr.Tab("✏️ 单文本分析"):
            with gr.Row():
                input_text = gr.Textbox(
                    label="输入评论",
                    placeholder="请输入一段中文评论，例如：这家餐厅的服务态度太差了，再也不会来！",
                    lines=4,
                    max_lines=8,
                )

            predict_btn = gr.Button("🔍 分析情感", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    output_label = gr.Textbox(
                        label="情感标签",
                        interactive=False,
                    )
                with gr.Column(scale=2):
                    output_prob = gr.Label(
                        label="置信度分布",
                        num_top_classes=2,
                    )
                with gr.Column(scale=1):
                    output_latency = gr.Textbox(
                        label="推理耗时",
                        interactive=False,
                    )

            with gr.Row():
                output_detail = gr.Textbox(
                    label="详细信息",
                    interactive=False,
                    lines=5,
                )

            predict_btn.click(
                fn=predict_single,
                inputs=[input_text, model_dropdown],
                outputs=[output_label, output_prob, output_latency, output_detail],
            )

            # 示例按钮
            gr.Examples(
                examples=[
                    ["这部电影真的太精彩了，强烈推荐！", "distilbert"],
                    ["服务态度极差，等了一个小时还没上菜，非常失望。", "distilbert"],
                    ["一般般吧，没什么特别的感受。", "bilstm"],
                    ["包装很精美，物流也很快，五星好评！", "distilbert"],
                ],
                inputs=[input_text, model_dropdown],
                label="📋 快速示例",
            )

        # -------------------------------------------------------------------
        # Tab 2: 批量分析
        # -------------------------------------------------------------------
        with gr.Tab("📁 批量分析"):
            with gr.Row():
                file_input = gr.File(
                    label="上传 CSV 文件",
                    file_types=[".csv"],
                )

            batch_btn = gr.Button("🚀 批量推理", variant="primary")

            with gr.Row():
                output_table = gr.Dataframe(
                    label="推理结果预览",
                    interactive=False,
                )

            with gr.Row():
                output_info = gr.Textbox(
                    label="统计信息",
                    interactive=False,
                    lines=2,
                )
                output_download = gr.File(
                    label="下载完整结果",
                    interactive=False,
                )

            batch_btn.click(
                fn=predict_batch,
                inputs=[file_input, model_dropdown],
                outputs=[output_table, output_info, output_download],
            )

            gr.Markdown("""
            **CSV 格式要求**：
            - 文件必须为 `.csv` 格式
            - 必须包含一列名为 `text` 的列
            - 每行一条待分析文本
            """)

        # -------------------------------------------------------------------
        # Footer
        # -------------------------------------------------------------------
        gr.Markdown("""
        ---
        <div class="footer">
        NLG_demo | 基于轻量化预训练模型的中文情感分析系统 | 实训课题
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="中文情感分析 Gradio 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成公网共享链接（Gradio Tunnel）")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
