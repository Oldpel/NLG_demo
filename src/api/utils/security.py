"""
输入安全过滤模块。

职责：
    - 长度截断（防止超长文本导致正则 ReDoS / OOM）
    - 危险字符过滤（HTML 标签、脚本注入、控制字符）
    - 空字符串 / 纯标点检测
    - 统一清洗（与 data_processing 保持一致）

使用位置：
    - api.py 的 predict / predict_batch 入口处调用
    - app.py 的前端输入层调用
"""

import re
from typing import Final

# 硬性长度上限（字符数），超过则截断
MAX_INPUT_LENGTH: Final[int] = 2000

# 危险控制字符（C0 控制字符 U+0000-U+001F，C1 控制字符 U+007F-U+009F）
_CONTROL_CHAR_PATTERN: Final[re.Pattern] = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

# 脚本注入特征（简单启发式）
_SCRIPT_PATTERN: Final[re.Pattern] = re.compile(
    r"<\s*script|javascript:|on\w+\s*=|&#\d+;|&#x[0-9a-fA-F]+;",
    re.IGNORECASE,
)

# 纯标点检测（如果一个字符串去掉标点后为空，则判定为无效）
_PUNCTUATION_PATTERN: Final[re.Pattern] = re.compile(
    r"[^\w一-鿿]+", re.UNICODE
)


class InputValidationError(ValueError):
    """输入校验失败异常，携带 human-readable 的拒绝原因。"""

    def __init__(self, reason: str, sanitized_preview: str = ""):
        self.reason = reason
        self.sanitized_preview = sanitized_preview
        super().__init__(f"[Security] 输入被拦截: {reason}")


def sanitize_input(text: str) -> str:
    """
    对前端输入执行安全清洗。

    处理流程：
        1. 去除首尾空白
        2. 去除控制字符
        3. 检测脚本注入特征
        4. 硬性长度截断
        5. 检测空字符串 / 纯标点

    Returns:
        清洗后的安全文本。

    Raises:
        InputValidationError: 当输入包含危险特征或完全无效时。
    """
    if not isinstance(text, str):
        raise InputValidationError(
            f"输入类型必须为 str，收到 {type(text).__name__}"
        )

    # 1. 去除首尾空白
    text = text.strip()

    # 2. 去除控制字符
    text = _CONTROL_CHAR_PATTERN.sub("", text)

    # 3. 检测脚本注入
    if _SCRIPT_PATTERN.search(text):
        preview = text[:50].replace("\n", " ")
        raise InputValidationError(
            "检测到疑似脚本注入特征（如 <script>, javascript: 等）",
            sanitized_preview=preview,
        )

    # 4. 硬性截断
    original_len = len(text)
    if original_len > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    # 5. 检测空或纯标点
    if not text:
        raise InputValidationError("输入为空字符串")

    # 去除标点后检查是否还有实质内容
    content_remain = _PUNCTUATION_PATTERN.sub("", text).strip()
    if not content_remain:
        raise InputValidationError(
            "输入仅包含标点符号，无实质文本内容",
            sanitized_preview=text[:50],
        )

    return text


def sanitize_inputs(texts: list[str]) -> list[str]:
    """批量安全清洗，返回所有通过校验的文本列表。"""
    results = []
    for i, text in enumerate(texts):
        try:
            results.append(sanitize_input(text))
        except InputValidationError as e:
            # 记录但不中断，跳过无效条目
            print(f"[WARN] 第 {i + 1} 条输入被拦截: {e.reason}")
    return results
