"""编排端到端评测的测试。

这组测试盯三件事：
1. 数据集本身合法（剧本期望与「不该编排」互斥、必填字段在）；
2. 判定逻辑不宽松——「剧本对了但步骤串错」「答案对但多说了已提交」都要判失败；
3. 评测隔离真的生效，否则第二次跑会拿上一次的档案，指标随运行次数漂移。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.eval.dataset import load_dataset
from app.eval.runner import Evaluator, render_orchestration_markdown
from app.memory.store import MemoryStore
from app.tools.store import ActionStore

DATASET = "app/eval/datasets/orchestration_e2e.jsonl"


def _case(case_id: str):
    return next(c for c in load_dataset(DATASET) if c.id == case_id)


# ---------------- 数据集 ----------------


def test_dataset_loads_and_has_both_kinds():
    cases = load_dataset(DATASET)
    assert len(cases) >= 10
    assert any(c.expect_recipe for c in cases), "缺少「该编排」的用例"
    assert any(c.expect_no_recipe for c in cases), "缺少「不该编排」的用例"
    # 每条至少有一个可判定项，否则它在报告里只是凑数
    for c in cases:
        assert c.has_orchestration_expectation, f"{c.id} 没有任何期望，评了等于没评"


def test_dataset_rejects_conflicting_expectation(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id": "x", "question": "q", "expect_recipe": "mentor_contact", "expect_no_recipe": true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="互斥"):
        load_dataset(path)


# ---------------- 判定逻辑 ----------------


class _Sessions:
    def reset(self, _sid: str) -> None:
        pass


class _FakeService:
    """只回放一个 payload，用来测判定，不碰真实检索与模型。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.sessions = _Sessions()
        self.llm = SimpleNamespace(provider="stub", model="stub")
        self.retriever = SimpleNamespace(embedder=SimpleNamespace(tag="stub"))

    async def ask(self, question: str, session_id: str | None = None, owner_id: str | None = None):
        return self.payload


def _payload(**over) -> dict:
    base = {
        "answer": "你的导师是 王琳，分机 8007。",
        "answer_plain": "你的导师是 王琳，分机 8007。",
        "plan": [
            {"step": "读取入职档案", "kind": "memory", "ok": True, "detail": "mentor=王琳"},
            {"step": "查询组织目录", "kind": "tool", "ok": True, "detail": "王琳"},
        ],
        "recipe": "mentor_contact",
        "requires_confirmation": False,
        "sources": [],
        "refused": False,
    }
    base.update(over)
    return base


def _run(case_id: str, payload: dict) -> dict:
    async def case():
        evaluator = Evaluator(service=_FakeService(payload))
        # 判定逻辑与档案无关：这里把造数据和清理短路掉
        evaluator._reset_owner = _noop  # type: ignore[assignment]
        evaluator._apply_setup = _noop  # type: ignore[assignment]
        return await evaluator._run_orchestration_case(_case(case_id), top_k=5)

    return asyncio.run(case())


async def _noop(*args, **kwargs):
    return None


def test_good_chain_passes():
    row = _run("o01", _payload())
    assert row["recipe_ok"] and row["steps_ok"] and row["plan_ok"] and row["content_ok"]
    assert row["e2e_ok"] is True


def test_right_recipe_but_wrong_step_order_fails():
    """剧本对了、答案也对，但步骤顺序反了 —— 必须判失败。"""
    row = _run(
        "o01",
        _payload(
            plan=[
                {"step": "查询组织目录", "kind": "tool", "ok": True, "detail": "王琳"},
                {"step": "读取入职档案", "kind": "memory", "ok": True, "detail": "mentor=王琳"},
            ]
        ),
    )
    assert row["recipe_ok"] is True
    assert row["steps_ok"] is False
    assert row["e2e_ok"] is False


def test_banned_phrase_fails_even_if_everything_else_passes():
    """o06 说「已提交成功」是最危险的错：用户以为真的开单了。"""
    payload = _payload(
        recipe="first_week_env",
        plan=[
            {"step": "检索知识库", "kind": "doc", "ok": True, "detail": ""},
            {"step": "从文档提取系统清单", "kind": "doc", "ok": True, "detail": "GitLab"},
            {"step": "生成待确认申请", "kind": "propose", "ok": True, "detail": "开通 GitLab"},
        ],
        requires_confirmation=True,
        answer="已提交成功，请等待审批。",
        answer_plain="已提交成功，请等待审批。",
    )
    row = _run("o06", payload)
    assert row["content_ok"] is False
    assert "已提交成功" in row["banned"]
    assert row["e2e_ok"] is False


def test_route_case_fails_when_orchestration_hijacks_it():
    """o09 是规则问题：被编排接走就是错，哪怕答案看起来对。"""
    hijacked = _payload(recipe="first_week_env", answer="申请权限需要主管审批。",
                        answer_plain="申请权限需要主管审批。")
    row = _run("o09", hijacked)
    assert row["recipe_ok"] is False
    assert row["e2e_ok"] is False

    ok = _payload(recipe=None, plan=[], answer="申请路径：OA → 权限申请。",
                  answer_plain="申请路径：OA → 权限申请。")
    row = _run("o09", ok)
    assert row["recipe_ok"] is True
    assert row["e2e_ok"] is True


def test_missing_step_is_not_counted_as_failure():
    """o03 档案里没导师：plan 只有一步，执行本身是成功的。"""
    row = _run(
        "o03",
        _payload(
            recipe="mentor_contact",
            plan=[{"step": "读取入职档案", "kind": "memory", "ok": True, "detail": "无记录"}],
            answer="你的档案里没有记录导师，建议直接问 HR。",
            answer_plain="你的档案里没有记录导师，建议直接问 HR。",
        ),
    )
    assert row["e2e_ok"] is True


def test_misroute_rate_counts_route_cases():
    rows = [
        {"expect_no_recipe": True, "recipe": "mentor_contact", "type": "route"},
        {"expect_no_recipe": True, "recipe": None, "type": "route"},
    ]
    agg = Evaluator._aggregate_orchestration(rows)  # type: ignore[arg-type]
    assert agg["route_cases"] == 2
    assert agg["misroute_rate"] == 0.5


# ---------------- 隔离 ----------------


def test_memory_purge_only_touches_target_owner(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")

    async def case():
        await store.init()
        await store.audit("2026-11-06T10:00:00", "eval-a", "remember.created", {})
        await store.audit("2026-11-06T10:00:00", "real-user", "remember.created", {})
        deleted = await store.purge_owner("eval-a")
        assert deleted >= 1
        assert await store.audits("eval-a") == []
        assert len(await store.audits("real-user")) == 1

    try:
        asyncio.run(case())
    finally:
        store.close()


def test_action_purge_clears_pending_and_tickets(tmp_path):
    store = ActionStore(tmp_path / "actions.db")

    async def case():
        await store.init()
        await store.create_ticket(owner_id="eval-a", kind="permission_request", payload={"system": "VPN"})
        await store.create_ticket(owner_id="real-user", kind="permission_request", payload={"system": "VPN"})
        assert await store.count_tickets("eval-a") == 1
        await store.purge_owner("eval-a")
        assert await store.count_tickets("eval-a") == 0
        assert await store.count_tickets("real-user") == 1

    try:
        asyncio.run(case())
    finally:
        store.close()


# ---------------- 报告 ----------------


def test_report_renders_every_row():
    async def case():
        evaluator = Evaluator(service=_FakeService(_payload()))
        evaluator._reset_owner = _noop  # type: ignore[assignment]
        evaluator._apply_setup = _noop  # type: ignore[assignment]
        cases = load_dataset(DATASET)[:3]
        rows = [await evaluator._run_orchestration_case(c, 5) for c in cases]
        return {
            "dataset": DATASET,
            "cases": len(rows),
            "top_k": 5,
            "started_at": "2026-11-06T10:00:00",
            "duration_s": 0.1,
            "llm": {"provider": "stub", "model": "stub"},
            "embedder": "stub",
            "overall": Evaluator._aggregate_orchestration(rows),
            "by_type": Evaluator._by_type_orchestration(rows),
            "rows": rows,
        }

    md = render_orchestration_markdown(asyncio.run(case()))
    assert "# 编排端到端评估报告" in md
    for case_id in ("o01", "o02", "o03"):
        assert case_id in md
