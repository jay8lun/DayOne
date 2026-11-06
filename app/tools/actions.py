"""确认式写操作：提议 → 确认 → 执行。

## 为什么写操作不能直接执行

模型把「帮我提个权限申请」翻译成一次工具调用是很容易的，但**翻译对不等于该执行**：
- 参数可能是编的（系统名写成「VPN系统」而不是「VPN」）；
- 用户可能只是问「怎么提」；
- 一次误执行会真的开单、真的找人审批，撤不回来。

所以写操作分三步：`propose()` 只生成**待确认操作**（一句人话 + 参数），
用户在界面上点确认，才 `confirm()` 真正执行。中间这层不会自动越过。

## 三个不显眼但要紧的约束

1. **幂等键就是 action_id**。双击确认、前端重试、消息重投，都只会执行一次；
   再来的请求返回同一份结果并标记 `replayed=True`。
2. **归属校验在服务端**。A 不能确认 B 的待办；报错信息也不透露细节。
3. **过期即作废**。待确认操作有 TTL（默认 30 分钟），过期后确认会被拒，
   逼用户重新发起——避免「昨天点的确认，今天才生效」这种事故。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..config import settings
from ..core.logging import get_logger
from ..memory import clock
from .base import ToolSpec
from .registry import registry
from .store import ActionStore

log = get_logger(__name__)


class ActionError(Exception):
    """可预期的业务错误：给用户看的一句话，不是 500。"""

    def __init__(self, message: str, code: str = "action_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ActionService:
    def __init__(self, store: ActionStore, *, audit: bool = True) -> None:
        self.store = store
        self.audit = audit

    # ---------- 提议 ----------
    async def propose(
        self,
        owner_id: str | None,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        client_key: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        if not owner_id:
            raise ActionError("没有拿到你的员工身份，无法发起操作", "no_owner")
        spec = registry.get(tool)
        if spec is None:
            raise ActionError(f"工具 {tool} 未注册", "unknown_tool")
        if spec.read_only:
            raise ActionError("这是只读查询，直接查就行，不需要确认", "not_writable")

        clean = self._validate(spec, args or {})
        if client_key:
            existing = await self.store.find_pending_by_client_key(owner_id, client_key)
            if existing and existing["status"] == "pending":
                # 连点两次「帮我提交」不该生成两条待办
                return {**existing, "deduped": True}

        action = await self.store.create_action(
            owner_id=owner_id,
            tool=tool,
            args=clean,
            summary=summary or self._summary(spec, clean),
            ttl_minutes=settings.tool_confirm_ttl_minutes,
            client_key=client_key,
        )
        await self._audit(owner_id, "action.propose",
                          {"action_id": action["id"], "tool": tool, "args": clean})
        return {**action, "deduped": False}

    def _validate(self, spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        """按 schema 校验参数：丢掉模型编的字段，缺必填就报错。

        参数幻觉是工具调用最常见的问题（工号写成 E12345、系统名拍脑袋）。
        这里做的是**结构性**检查——必填、类型、长度；语义对不对只能靠工具自己查。
        """
        props: dict[str, Any] = (spec.parameters or {}).get("properties", {}) or {}
        required: list[str] = (spec.parameters or {}).get("required", []) or []
        unknown = [k for k in args if k not in props]
        if unknown:
            log.info("忽略模型传入的未知参数: %s", unknown)

        clean: dict[str, Any] = {}
        for key, value in args.items():
            if key not in props:
                continue
            if isinstance(value, str):
                value = value.strip()[:500]
            clean[key] = value

        missing = [k for k in required if not str(clean.get(k, "")).strip()]
        if missing:
            labels = "、".join(props.get(k, {}).get("description", k).split("，")[0] for k in missing)
            raise ActionError(f"还缺必要信息：{labels}（{', '.join(missing)}）", "missing_args")

        for key, meta in props.items():
            enum = meta.get("enum")
            if enum and key in clean and clean[key] not in enum:
                raise ActionError(f"参数 {key} 只能是 {'/'.join(map(str, enum))}", "bad_enum")

        if spec.validate is not None:
            message = spec.validate(clean)
            if message:
                raise ActionError(message, "invalid_args")
        return clean

    def _summary(self, spec: ToolSpec, args: dict[str, Any]) -> str:
        if spec.describe is not None:
            try:
                return spec.describe(args)
            except Exception as exc:  # noqa: BLE001
                log.warning("工具 %s 的 describe 失败: %s", spec.name, exc)
        pairs = "，".join(f"{k}={v}" for k, v in args.items())
        return f"执行「{spec.name}」：{pairs}"

    # ---------- 确认执行 ----------
    async def confirm(self, owner_id: str | None, action_id: str) -> dict[str, Any]:
        action = await self.store.get_action(action_id)
        if not action:
            raise ActionError("找不到这个待确认操作（可能已过期或链接失效）", "not_found")
        if owner_id and action["owner_id"] != owner_id:
            # 不说是谁的、也不说参数，避免被拿来探测别人在办什么
            raise ActionError("这个操作不属于你，无法确认", "forbidden")

        if action["status"] == "executed":
            return {"ok": True, "replayed": True, "action": action,
                    "result": action.get("result") or {}}
        if action["status"] == "cancelled":
            raise ActionError("这个操作已经取消了，请重新发起", "cancelled")
        if action["status"] == "expired":
            raise ActionError("这个操作已过期，请重新发起", "expired")

        expires = clock.parse_time(action["expires_at"])
        if expires and clock.now() > expires:
            await self.store.update_action(action_id, status="expired")
            await self._audit(action["owner_id"], "action.expired",
                              {"action_id": action_id, "tool": action["tool"]})
            raise ActionError(
                f"这个操作已过期（超过 {settings.tool_confirm_ttl_minutes} 分钟），请重新发起",
                "expired",
            )

        spec = registry.get(action["tool"])
        if spec is None or spec.read_only:
            raise ActionError("该操作不可执行", "not_writable")

        # allow_write=True 只在这里出现：全系统其它地方都不允许执行写工具
        result = await registry.execute(
            action["tool"], action["args"], owner_id=action["owner_id"], allow_write=True
        )
        payload = result.to_dict()
        await self.store.update_action(
            action_id, status="executed", executed_at=clock.iso(clock.now()) or "",
            result=payload,
        )
        await self._audit(action["owner_id"], "action.confirm",
                          {"action_id": action_id, "tool": action["tool"],
                           "ok": result.ok, "error": result.error[:200]})
        refreshed = await self.store.get_action(action_id) or action
        return {"ok": result.ok, "replayed": False, "action": refreshed, "result": payload}

    # ---------- 取消 ----------
    async def cancel(self, owner_id: str | None, action_id: str) -> dict[str, Any]:
        action = await self.store.get_action(action_id)
        if not action:
            raise ActionError("找不到这个待确认操作", "not_found")
        if owner_id and action["owner_id"] != owner_id:
            raise ActionError("这个操作不属于你", "forbidden")
        if action["status"] == "executed":
            raise ActionError("这个操作已经执行了，无法取消", "already_executed")
        if action["status"] != "cancelled":
            await self.store.update_action(action_id, status="cancelled")
            await self._audit(action["owner_id"], "action.cancel",
                              {"action_id": action_id, "tool": action["tool"]})
        return {"ok": True, "action": await self.store.get_action(action_id)}

    async def pending(self, owner_id: str) -> list[dict[str, Any]]:
        rows = await self.store.list_actions(owner_id, status="pending")
        now = clock.now()
        alive = []
        for row in rows:
            expires = clock.parse_time(row["expires_at"])
            if expires and now > expires:
                await self.store.update_action(row["id"], status="expired")
                continue
            alive.append(row)
        return alive

    async def _audit(self, owner_id: str | None, action: str, detail: dict[str, Any]) -> None:
        if not self.audit:
            return
        try:
            from ..memory import get_memory

            await get_memory().store.audit(clock.iso(clock.now()) or "", owner_id, action, detail)
        except Exception as exc:  # noqa: BLE001
            log.debug("动作审计写入失败（忽略）: %s", exc)


# ---------- 单例 ----------
_store: ActionStore | None = None
_service: ActionService | None = None
_lock = asyncio.Lock()


async def get_action_store() -> ActionStore:
    global _store
    async with _lock:
        if _store is None:
            _store = ActionStore(settings.tool_actions_db_path)
            await _store.init()
        return _store


async def get_actions() -> ActionService:
    global _service
    if _service is None:
        _service = ActionService(await get_action_store())
    return _service


def reset_actions(store: ActionStore | None = None, *, audit: bool = True) -> ActionService:
    """测试/探针用：换一套库。探针传 `audit=False`，避免往真实审计表里写噪声。"""
    global _store, _service
    _store = store
    _service = ActionService(store, audit=audit) if store is not None else None
    return _service  # type: ignore[return-value]
