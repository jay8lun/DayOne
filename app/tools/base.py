"""工具的定义与结果类型。

两条硬约束（对应 docs/演进设计.md §4.4）：

1. **只读与写操作必须分开声明**。`read_only=False` 的工具不允许被自动执行——
   必须先返回「待确认的操作」给用户，确认后才落库，且带幂等键。
2. **结果必须能裁剪**。内部接口经常返回几十上百行 JSON，原样塞进上下文会瞬间撑爆窗口，
   所以 `ToolResult.to_text()` 负责把结构化结果压成一段可控长度的说明。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

Handler = Callable[..., Awaitable[Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    read_only: bool = True
    timeout_s: float = 8.0

    def schema(self) -> dict[str, Any]:
        """OpenAI function calling 的 schema 形状。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    name: str
    data: Any = None
    error: str = ""
    elapsed_ms: int = 0
    args: dict[str, Any] = field(default_factory=dict)

    def to_text(self, max_chars: int = 600) -> str:
        """压成一段给模型看的文本：超长截断，并明确告知被截断了。"""
        if not self.ok:
            return f"工具 {self.name} 调用失败：{self.error or '未知错误'}"
        if isinstance(self.data, str):
            text = self.data
        else:
            try:
                text = json.dumps(self.data, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                text = str(self.data)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n…（结果过长，已截断，原始 {len(text)} 字符）"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "args": self.args,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "data": self.data,
        }
