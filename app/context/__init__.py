"""上下文管理：预算分层、旧历史压缩、记忆注入。

对应 docs/演进设计.md 第 3 节（P2）。核心约束：**参考资料永不压缩**——
压缩 chunk 会让引用回验的原文重叠度下降，grounding 指标假性下跌。
"""

from __future__ import annotations

from .budget import estimate_tokens, truncate_to_tokens
from .builder import BuiltContext, ContextBuilder, Layer, get_builder

__all__ = ["BuiltContext", "ContextBuilder", "Layer", "estimate_tokens", "get_builder", "truncate_to_tokens"]
