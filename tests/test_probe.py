"""闭环探针自身的测试。

两个目的：
1. **防退化**：探针全绿是「记忆系统在时间轴上收敛」的基线，破了要立刻知道；
2. **防自欺**：断言必须能失败。这里用 monkeypatch 把「常驻保护」关掉，
   验证对应探针确实会红——否则探针只是摆设。
"""

from __future__ import annotations

import asyncio

from app.config import settings
from app.eval.probe import render_markdown, run_probes, summarize


def test_all_probes_pass():
    reports = asyncio.run(run_probes())
    assert len(reports) >= 10, f"探针数量异常：{[r.id for r in reports]}"

    failures = [
        f"{r.id} / {c.name}: {c.detail}"
        for r in reports
        if not r.passed
        for c in r.failed
    ]
    assert not failures, "探针失败：\n" + "\n".join(failures)

    stats = summarize(reports)
    assert stats["failed"] == 0
    assert stats["checks_passed"] == stats["checks"], stats
    # 每条探针至少要有 2 条断言，否则「通过」没有意义
    assert all(len(r.checks) >= 2 for r in reports), {r.id: len(r.checks) for r in reports}


def test_probe_can_fail(monkeypatch):
    """关掉常驻保护，`pinned_survives_decay` 必须红：证明断言不是空转。"""
    monkeypatch.setattr(settings, "memory_pinned_predicates", "")
    reports = asyncio.run(run_probes(only=["pinned_survives_decay"]))
    assert len(reports) == 1
    assert reports[0].passed is False
    assert any("工号" in c.name for c in reports[0].failed)


def test_report_renders_markdown():
    reports = asyncio.run(run_probes(only=["conflict_latest_wins"]))
    md = render_markdown(reports, {"time": "2026-11-06 21:00:00"})
    assert "# 记忆 / 上下文闭环探针报告" in md
    assert "conflict_latest_wins" in md
    assert "时间轴" in md
    assert "期望 vs 实际" in md
