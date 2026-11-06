"""确认式写操作的行为测试。

五条是这个功能的命门，缺一条就可能在真实系统里开重复单或替别人办事：
1. 没确认不能执行（只读栈里也不可能绕过）；
2. 重复确认只执行一次（幂等）；
3. 连点两次「帮我提交」只生成一条待办（client_key 去重）；
4. 别人的待办不能确认（归属校验）；
5. 过期即作废（TTL）。
"""

from __future__ import annotations

import asyncio

import pytest

from app.tools.actions import ActionError, ActionService, reset_actions
from app.tools.registry import registry
from app.tools.store import ActionStore


@pytest.fixture()
def svc(tmp_path):
    store = ActionStore(tmp_path / "actions.db")
    asyncio.run(store.init())
    service = reset_actions(store, audit=False)
    yield service
    reset_actions(None)
    store.close()


def _run(coro):
    return asyncio.run(coro)


def test_propose_then_confirm_creates_ticket(svc: ActionService):
    async def case():
        action = await svc.propose("u1", "submit_permission_request",
                                   {"system": "VPN", "reason": "需要外网调试"})
        assert action["status"] == "pending"
        # 提议阶段绝不能有副作用
        assert await svc.store.count_tickets("u1") == 0
        assert "VPN" in action["summary"]

        result = await svc.confirm("u1", action["id"])
        assert result["ok"] is True
        assert result["replayed"] is False
        assert result["result"]["data"]["ticket_no"].startswith("REQ-")
        assert await svc.store.count_tickets("u1") == 1

    _run(case())


def test_confirm_twice_is_idempotent(svc: ActionService):
    async def case():
        action = await svc.propose("u1", "submit_permission_request",
                                   {"system": "GitLab", "reason": "要建仓库"})
        first = await svc.confirm("u1", action["id"])
        second = await svc.confirm("u1", action["id"])

        assert first["replayed"] is False
        assert second["replayed"] is True          # 第二次是重放，不是又办了一次
        assert second["result"] == first["result"]
        assert await svc.store.count_tickets("u1") == 1

    _run(case())


def test_propose_dedup_by_client_key(svc: ActionService):
    async def case():
        a = await svc.propose("u1", "submit_permission_request",
                              {"system": "VPN", "reason": "外网调试"}, client_key="msg-001")
        b = await svc.propose("u1", "submit_permission_request",
                              {"system": "VPN", "reason": "外网调试"}, client_key="msg-001")
        assert a["id"] == b["id"]
        assert b["deduped"] is True
        assert len(await svc.store.list_actions("u1", status="pending")) == 1

    _run(case())


def test_missing_required_arg_rejected(svc: ActionService):
    async def case():
        with pytest.raises(ActionError) as exc:
            await svc.propose("u1", "submit_permission_request", {"system": "VPN"})
        assert exc.value.code == "missing_args"
        assert "理由" in exc.value.message
        assert await svc.store.list_actions("u1", status="pending") == []

    _run(case())


def test_read_tool_cannot_be_proposed(svc: ActionService):
    async def case():
        with pytest.raises(ActionError) as exc:
            await svc.propose("u1", "lookup_colleague", {"name": "王琳"})
        assert exc.value.code == "not_writable"

    _run(case())


def test_foreign_action_cannot_be_confirmed(svc: ActionService):
    async def case():
        action = await svc.propose("u1", "submit_permission_request",
                                   {"system": "VPN", "reason": "调试"})
        with pytest.raises(ActionError) as exc:
            await svc.confirm("u2", action["id"])
        assert exc.value.code == "forbidden"
        assert await svc.store.count_tickets("u1") == 0
        # 原主人的待办不能被别人的失败请求改状态
        still = await svc.store.get_action(action["id"])
        assert still["status"] == "pending"

    _run(case())


def test_expired_action_rejected(svc: ActionService):
    async def case():
        action = await svc.propose("u1", "submit_permission_request",
                                   {"system": "VPN", "reason": "调试"})
        # 把过期时间调到过去，模拟「昨天点的确认」
        await svc.store.update_action(action["id"], expires_at="2020-01-01T00:00:00+00:00")
        with pytest.raises(ActionError) as exc:
            await svc.confirm("u1", action["id"])
        assert exc.value.code == "expired"
        assert await svc.store.count_tickets("u1") == 0

    _run(case())


def test_cancel_then_confirm_rejected(svc: ActionService):
    async def case():
        action = await svc.propose("u1", "submit_permission_request",
                                   {"system": "VPN", "reason": "调试"})
        await svc.cancel("u1", action["id"])
        with pytest.raises(ActionError) as exc:
            await svc.confirm("u1", action["id"])
        assert exc.value.code == "cancelled"

    _run(case())


def test_unknown_params_are_dropped(svc: ActionService):
    async def case():
        action = await svc.propose(
            "u1", "submit_permission_request",
            {"system": "VPN", "reason": "调试", "owner_id": "u2", "hack": "x"},
        )
        # 模型编的参数不能进待确认操作，更不能进工单
        assert action["args"] == {"system": "VPN", "reason": "调试"}

    _run(case())


def test_executor_refuses_write_without_allow(svc: ActionService):
    async def case():
        result = await registry.execute(
            "submit_permission_request", {"system": "VPN", "reason": "调试"}, owner_id="u1"
        )
        assert result.ok is False
        assert "确认" in result.error
        assert await svc.store.count_tickets("u1") == 0

    _run(case())


def test_api_propose_confirm_roundtrip(tmp_path):
    """接口层闭环：提议 → 确认 ×2 → 查工单 → 越权被拒。

    这条是补漏用的：之前 `get_actions()` 忘了 await，单元测试全绿，
    但走 HTTP 直接 500（`'coroutine' object has no attribute 'propose'`）。
    """
    from fastapi.testclient import TestClient

    from app.main import app

    store = ActionStore(tmp_path / "api-actions.db")
    asyncio.run(store.init())
    reset_actions(store, audit=False)
    try:
        with TestClient(app) as client:
            r = client.post("/api/tools/propose", json={
                "owner_id": "emp-api", "tool": "submit_permission_request",
                "args": {"system": "VPN", "reason": "宿舍调试"},
                "client_key": "api-msg-1",
            })
            assert r.status_code == 200, r.text
            action_id = r.json()["action"]["id"]

            c1 = client.post("/api/tools/confirm", json={"owner_id": "emp-api", "action_id": action_id})
            assert c1.status_code == 200, c1.text
            first = c1.json()
            assert first["replayed"] is False
            assert first["result"]["data"]["ticket_no"].startswith("REQ-")

            c2 = client.post("/api/tools/confirm", json={"owner_id": "emp-api", "action_id": action_id})
            assert c2.json()["replayed"] is True

            tickets = client.get("/api/tools/tickets?owner_id=emp-api").json()
            assert tickets["count"] == 1

            forbidden = client.post("/api/tools/confirm", json={"owner_id": "emp-other", "action_id": action_id})
            assert forbidden.status_code == 403
            assert forbidden.json()["detail"]["code"] == "forbidden"

            tools = client.get("/api/tools").json()
            assert tools["write_enabled"] is True
            assert any(t["name"] == "submit_permission_request" and not t["read_only"]
                       for t in tools["tools"])
    finally:
        reset_actions(None)
        store.close()
