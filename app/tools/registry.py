"""工具注册表 + 执行器。

注册表这层的意义和 `app/adapters/` 一样：**主流程只认识名字和 schema**，
换公司、换内部系统只换工具实现，不改编排代码。

执行器负责四件事，缺一件就上不了生产：
- 超时（内部接口卡住不能把问答一起拖死）
- 异常兜底（工具炸了要变成一句可读的话，而不是 500）
- 只读校验（写操作默认不自动执行）
- 审计（谁在何时调了什么、参数是什么、返回是否成功）
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ..core.logging import get_logger
from .base import ToolResult, ToolSpec

log = get_logger(__name__)


class ToolNotAllowed(Exception):
    """越权或未注册的工具。"""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            log.warning("工具重名注册，后者覆盖前者: %s", spec.name)
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, *, read_only_only: bool = True) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values() if t.read_only or not read_only_only]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "read_only": t.read_only, "timeout_s": t.timeout_s,
             "description": t.description}
            for t in self._tools.values()
        ]

    async def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        owner_id: str | None = None,
        allow_write: bool = False,
    ) -> ToolResult:
        spec = self._tools.get(name)
        args = dict(args or {})
        if spec is None:
            return ToolResult(ok=False, name=name, error="工具未注册", args=args)
        if not spec.read_only and not allow_write:
            # 写操作必须在用户确认后才执行，这里直接拒绝自动调用
            return ToolResult(ok=False, name=name,
                              error="这是写操作，需要用户确认后才能执行", args=args)

        # 身份统一在这里注入：工具自己拿不到调用者身份，也绝不接受调用方（模型）传入。
        # 之前漏了这一步，写工具落库时 owner_id 是空的——单据挂在“查无此人”名下，
        # 而接口返回看起来一切正常。这类错必须在执行器这一层堵死。
        if spec.needs_owner:
            if not owner_id:
                return ToolResult(ok=False, name=name, args=args,
                                  error="缺少员工身份，无法执行该工具")
            args["owner_id"] = owner_id

        started = time.perf_counter()
        try:
            data = await asyncio.wait_for(spec.handler(**args), timeout=spec.timeout_s)
            result = ToolResult(
                ok=True, name=name, data=data, args=args,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                ok=False, name=name, args=args,
                error=f"超时（{spec.timeout_s}s）",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except TypeError as exc:
            # 参数不匹配：模型编了不存在的参数名，属于常见幻觉
            result = ToolResult(ok=False, name=name, args=args,
                                error=f"参数不合法：{exc}",
                                elapsed_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:  # noqa: BLE001
            log.warning("工具 %s 执行异常: %s", name, exc)
            result = ToolResult(ok=False, name=name, args=args, error=str(exc),
                                elapsed_ms=int((time.perf_counter() - started) * 1000))

        await self._audit(owner_id, result)
        return result

    async def _audit(self, owner_id: str | None, result: ToolResult) -> None:
        """尽力而为地写审计。记忆库不可用不影响工具执行。"""
        try:
            from ..memory import get_memory
            from ..memory import clock as memory_clock

            store = get_memory().store
            await store.audit(
                memory_clock.iso(memory_clock.now()) or "",
                owner_id,
                "tool.call",
                {
                    "tool": result.name,
                    "args": result.args,
                    "ok": result.ok,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error[:200],
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("工具审计写入失败（忽略）: %s", exc)


registry = ToolRegistry()


def tool_schemas(*, read_only_only: bool = True) -> str:
    """给日志/调试用的紧凑视图。"""
    return json.dumps(registry.describe(), ensure_ascii=False, indent=2)
