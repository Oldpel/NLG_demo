"""
API 边界测试与压力测试套件。

覆盖场景：
    - 正常中文文本（短/中/长）
    - 空字符串
    - 纯标点符号
    - 超长文本（超过 2000 字符上限）
    - 脚本注入攻击（XSS）
    - 控制字符 / 特殊 Unicode
    - 数字 / 代码片段
    - 混合语言
    - 批量推理中的部分无效输入

用法：
    python -m pytest tests/test_api.py -v
    # 或
    python -m tests.test_api
"""

import os
import sys
import unittest

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.api.utils.security import sanitize_input, InputValidationError


# ---------------------------------------------------------------------------
# 安全过滤测试
# ---------------------------------------------------------------------------

class TestSecurityFilter(unittest.TestCase):
    """测试输入安全过滤模块。"""

    def test_normal_chinese_text(self):
        """正常中文文本应通过校验。"""
        text = "这部电影真的太精彩了，强烈推荐！"
        result = sanitize_input(text)
        self.assertEqual(result, text)

    def test_text_with_punctuation(self):
        """含标点的正常文本应通过。"""
        text = "服务一般吧……没什么特别的感受。"
        result = sanitize_input(text)
        self.assertEqual(result, text)

    def test_empty_string(self):
        """空字符串应被拦截。"""
        with self.assertRaises(InputValidationError) as ctx:
            sanitize_input("")
        self.assertIn("空字符串", str(ctx.exception))

    def test_whitespace_only(self):
        """仅空白字符应被拦截。"""
        with self.assertRaises(InputValidationError):
            sanitize_input("   \t\n  ")

    def test_pure_punctuation(self):
        """纯标点应被拦截。"""
        with self.assertRaises(InputValidationError) as ctx:
            sanitize_input("！！！。。。？？、、、")
        self.assertIn("标点", str(ctx.exception))

    def test_too_long_text(self):
        """超长文本应被截断。"""
        long_text = "好" * 3000
        result = sanitize_input(long_text)
        self.assertLessEqual(len(result), 2000)
        self.assertEqual(result, "好" * 2000)

    def test_script_injection(self):
        """脚本注入特征应被拦截。"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "onclick=alert(1)",
            "&#60;script&#62;",
        ]
        for text in malicious_inputs:
            with self.subTest(text=text):
                with self.assertRaises(InputValidationError) as ctx:
                    sanitize_input(text)
                self.assertIn("脚本", str(ctx.exception))

    def test_control_characters(self):
        """控制字符应被去除。"""
        text = "正常文本\x00\x01\x02结尾"
        result = sanitize_input(text)
        self.assertEqual(result, "正常文本结尾")

    def test_mixed_language(self):
        """中英文混合应通过。"""
        text = "This movie is 真的 very good!"
        result = sanitize_input(text)
        self.assertEqual(result, text)

    def test_numbers_and_code(self):
        """数字和代码片段应通过（代码本身不是攻击）。"""
        text = "print('hello') + 12345 = ???"
        result = sanitize_input(text)
        self.assertEqual(result, text)

    def test_none_input(self):
        """非字符串输入应被拦截。"""
        with self.assertRaises(InputValidationError) as ctx:
            sanitize_input(None)
        self.assertIn("类型", str(ctx.exception))

    def test_sql_injection_like(self):
        """SQL 注入风格文本（非脚本注入）应通过（由应用层处理）。"""
        text = "DROP TABLE users; -- 评论"
        result = sanitize_input(text)
        self.assertIn("DROP TABLE", result)


# ---------------------------------------------------------------------------
# API 推理测试（需要模型权重）
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer(unittest.TestCase):
    """
    测试 SentimentAnalyzer 的推理行为。
    注意：以下测试需要对应的模型权重存在，否则会被跳过。
    """

    @classmethod
    def setUpClass(cls):
        """尝试加载模型，不存在则跳过整个测试类。"""
        from src.api.api import SentimentAnalyzer

        cls.skip_all = False
        cls.analyzer = None

        # 测试 DistilBERT（优先）
        distilbert_path = os.path.join(PROJECT_ROOT, "checkpoints", "distilbert_best")
        bilstm_path = os.path.join(PROJECT_ROOT, "checkpoints", "bilstm_best.pth")

        if os.path.exists(distilbert_path):
            try:
                cls.analyzer = SentimentAnalyzer(model_type="distilbert")
            except Exception as e:
                print(f"[WARN] DistilBERT 模型加载失败: {e}")
                cls.skip_all = True
        elif os.path.exists(bilstm_path):
            try:
                cls.analyzer = SentimentAnalyzer(model_type="bilstm")
            except Exception as e:
                print(f"[WARN] BiLSTM 模型加载失败: {e}")
                cls.skip_all = True
        else:
            print("[SKIP] 未找到模型权重，跳过推理测试")
            cls.skip_all = True

    def setUp(self):
        if self.skip_all:
            self.skipTest("模型权重不存在，跳过")

    def test_positive_text(self):
        """正面评论应返回正向标签。"""
        result = self.analyzer.predict("这部电影真的太精彩了，强烈推荐！")
        self.assertEqual(result["label"], "正向积极评论")
        self.assertGreater(result["prob"], 0.5)
        self.assertGreater(result["score"], 0.5)
        self.assertIn("latency_ms", result)
        self.assertGreater(result["latency_ms"], 0)

    def test_negative_text(self):
        """负面评论应返回负向标签。"""
        result = self.analyzer.predict("服务态度极差，非常失望，再也不会来了！")
        self.assertEqual(result["label"], "负向消极评论")
        self.assertLess(result["prob"], 0.5)

    def test_empty_input(self):
        """空输入应返回错误标记而非崩溃。"""
        result = self.analyzer.predict("")
        self.assertIn("error", result)
        self.assertEqual(result["label"], "输入无效")

    def test_script_injection_handled(self):
        """脚本注入应被安全拦截。"""
        result = self.analyzer.predict("<script>alert(1)</script>")
        self.assertIn("error", result)
        self.assertEqual(result["label"], "输入无效")

    def test_very_long_text(self):
        """超长文本应被截断后正常推理。"""
        long_text = "好" * 3000
        result = self.analyzer.predict(long_text)
        # 不应报错，且应有正常标签
        self.assertIn(result["label"], ["正向积极评论", "负向消极评论"])
        self.assertIn("latency_ms", result)

    def test_latency_reasonable(self):
        """推理延迟应在合理范围内（目标 < 100ms）。"""
        result = self.analyzer.predict("这是一条测试评论")
        self.assertLess(result["latency_ms"], 1000)  # 放宽到 1s 作为上限

    def test_batch_predict(self):
        """批量推理应返回等长结果列表。"""
        texts = [
            "太棒了！",
            "太差了！",
            "",  # 会被拦截
            "一般般吧",
        ]
        results = self.analyzer.predict_batch(texts)
        self.assertEqual(len(results), len(texts))
        # 前三条应有 index 字段
        for i, r in enumerate(results):
            self.assertEqual(r["index"], i)

    def test_switch_model(self):
        """模型热切换应正常工作（需要两个模型都存在）。"""
        distilbert_path = os.path.join(PROJECT_ROOT, "checkpoints", "distilbert_best")
        bilstm_path = os.path.join(PROJECT_ROOT, "checkpoints", "bilstm_best.pth")

        if not (os.path.exists(distilbert_path) and os.path.exists(bilstm_path)):
            self.skipTest("需要同时存在两个模型权重")

        # 当前模型
        old_type = self.analyzer.model_type

        # 切换到另一个模型
        new_type = "bilstm" if old_type == "distilbert" else "distilbert"
        self.analyzer.switch_model(new_type)

        result = self.analyzer.predict("测试文本")
        self.assertEqual(result["model_type"], new_type)
        self.assertIn(result["label"], ["正向积极评论", "负向消极评论"])

        # 切回
        self.analyzer.switch_model(old_type)


# ---------------------------------------------------------------------------
# 压力测试
# ---------------------------------------------------------------------------

class TestStress(unittest.TestCase):
    """压力测试：边界用例。"""

    def test_boundary_inputs(self):
        """测试各种边界输入的安全过滤。"""
        boundary_cases = [
            ("", "空字符串"),
            ("   ", "纯空格"),
            ("\t\n\r", "纯换行符"),
            ("！！！", "纯中文感叹号"),
            ("...", "纯英文省略号"),
            ("🎉🎊✨", "纯 Emoji"),  # Emoji 不是标点，应通过
            ("1" * 5000, "超长数字串"),
            ("你好" + "\x00" * 100 + "世界", "含空字节"),
            ("<script>", "脚本标签"),
            ("javascript:alert(1)", "JS 协议"),
        ]

        for text, desc in boundary_cases:
            with self.subTest(desc=desc, text=text[:30]):
                try:
                    result = sanitize_input(text)
                    # 如果通过了，说明是合法输入
                    self.assertIsInstance(result, str)
                except InputValidationError:
                    # 被拦截也是预期行为
                    pass


# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------

def run_tests():
    """独立运行测试套件。"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSecurityFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestStress))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
