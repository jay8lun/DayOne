"""工具调用（P3）：注册表 + 执行器 + 路由。

设计取舍见 docs/演进设计.md §4：
- 工具这层和 `app/adapters/` 同构——主流程只认名字与 schema，换内部系统只换实现；
- 只读工具可以自动执行，写操作必须先「提议—确认」；
- 工具结果必须裁剪，否则一次查询就能把上下文撑爆。
"""

from __future__ import annotations

from .base import ToolResult, ToolSpec
from .builtin import register_builtin_tools
from .registry import ToolRegistry, registry, tool_schemas
from .router import answer_with_tool, looks_actionable, route_tool

__all__ = [
    "ToolResult", "ToolSpec", "ToolRegistry", "answer_with_tool", "looks_actionable",
    "register_builtin_tools", "registry", "route_tool", "tool_schemas",
]
