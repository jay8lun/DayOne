"""上下文预算：估算 token 占用。

不引入 tiktoken 之类的额外依赖，用可解释的启发式：
- 中日韩字符：1 字符 ≈ 1 token（主流 tokenizer 的中文压缩率大致如此）
- 英文/数字：4 字符 ≈ 1 token
- 其余符号：按 0.5 计

误差在 ±15% 以内，对于「预算分配」这个用途足够——
我们只需要知道「快超了」，不需要精确到个位。真要精确时替换 `estimate_tokens` 一处即可。
"""

from __future__ import annotations

import math


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # 基本汉字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0x3000 <= code <= 0x303F   # 中文标点
        or 0xFF00 <= code <= 0xFFEF   # 全角字符
    )


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    ascii_chars = sum(1 for ch in text if ch.isascii() and not ch.isspace())
    other = len(text) - cjk - ascii_chars
    return int(math.ceil(cjk * 1.0 + ascii_chars / 4.0 + other * 0.5))


def truncate_to_tokens(text: str, limit: int, suffix: str = "…") -> str:
    """按估算 token 截断，尽量在句读处断开。"""
    if limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    head = text[:lo]
    for sep in ("。", "；", "\n", "，", " "):
        idx = head.rfind(sep)
        if idx > lo * 0.6:
            head = head[: idx + 1]
            break
    return head + suffix
