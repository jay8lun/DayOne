"""评估执行器：跑数据集 -> 出指标 -> 出报告。

用法：
```bash
python scripts/evaluate.py --top-k 5              # 只评检索（快，不花 token）
python scripts/evaluate.py --with-generation      # 连生成一起评
```
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..config import settings
from ..core.logging import get_logger
from ..rag.retriever import HybridRetriever
from ..rag.service import RagService, get_service
from .dataset import EvalCase, load_dataset
from .metrics import (
    keyword_coverage,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

log = get_logger(__name__)


class Evaluator:
    def __init__(self, service: RagService | None = None, retriever: HybridRetriever | None = None) -> None:
        self.service = service or get_service()
        self.retriever = retriever or self.service.retriever

    async def run(
        self,
        dataset_path: str | None = None,
        top_k: int | None = None,
        limit: int | None = None,
        with_generation: bool = False,
    ) -> dict[str, Any]:
        path = dataset_path or settings.eval_dataset
        top_k = top_k or settings.eval_top_k
        cases = load_dataset(path)
        if limit:
            cases = cases[:limit]

        await self.retriever.refresh_index()
        t0 = time.perf_counter()
        rows: list[dict[str, Any]] = []

        for case in cases:
            row = await self._run_case(case, top_k, with_generation)
            rows.append(row)
            log.info(
                "eval %s: recall@%d=%.2f mrr=%.2f%s",
                case.id,
                top_k,
                row["recall_at_k"],
                row["mrr"],
                f" kw={row['keyword_coverage']:.2f}" if with_generation else "",
            )

        report = {
            "dataset": str(path),
            "cases": len(cases),
            "top_k": top_k,
            "with_generation": with_generation,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - t0, 2),
            "llm": {"provider": self.service.llm.provider, "model": self.service.llm.model},
            "embedder": self.retriever.embedder.tag,
            "overall": self._aggregate(rows, with_generation),
            "by_type": self._by_type(rows, with_generation),
            "rows": rows,
        }
        return report

    # ================= 编排端到端 =================
    #
    # 检索指标（Recall/MRR）只证明「找得到」，证明不了「把一件多步的事办对」。
    # 这一段专门跑编排：命中哪个剧本、步骤顺序对不对、上一步的输出有没有
    # 真的变成下一步的入参、该不该产生待确认操作、该不该被编排接走。

    async def run_orchestration(
        self,
        dataset_path: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        path = dataset_path or settings.eval_orchestration_dataset
        top_k = top_k or settings.eval_top_k
        cases = load_dataset(path)
        if limit:
            cases = cases[:limit]

        t0 = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for case in cases:
            row = await self._run_orchestration_case(case, top_k)
            rows.append(row)
            log.info(
                "orch %s: recipe=%s steps=%s e2e=%s",
                case.id,
                row["recipe"],
                "→".join(row["steps"]),
                row["e2e_ok"],
            )

        return {
            "dataset": str(path),
            "cases": len(cases),
            "top_k": top_k,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - t0, 2),
            "llm": {"provider": self.service.llm.provider, "model": self.service.llm.model},
            "embedder": self.retriever.embedder.tag,
            "overall": self._aggregate_orchestration(rows),
            "by_type": self._by_type_orchestration(rows),
            "rows": rows,
        }

    async def _run_orchestration_case(self, case: EvalCase, top_k: int) -> dict[str, Any]:
        owner = f"eval-orch-{case.id}"
        await self._reset_owner(owner)
        await self._apply_setup(case, owner)

        session_id = f"eval-orch-session-{case.id}"
        self.service.sessions.reset(session_id)
        t0 = time.perf_counter()
        payload = await self.service.ask(case.question, session_id, owner)

        plan = payload.get("plan") or []
        kinds = [p.get("kind") for p in plan]
        recipe = payload.get("recipe")
        answer = payload.get("answer_plain") or ""
        sources = payload.get("sources") or []
        retrieved = [
            {"doc_title": s.get("doc_title", ""), "external_id": s.get("external_id", "")}
            for s in sources
        ]

        # ---- 逐项判定：None 表示这条用例没提这个要求，不参与打分 ----
        recipe_ok: bool | None = None
        if case.expect_recipe:
            recipe_ok = recipe == case.expect_recipe
        elif case.expect_no_recipe:
            recipe_ok = recipe is None

        steps_ok: bool | None = None
        if case.expect_steps:
            steps_ok = kinds == case.expect_steps

        steps_all_ok: bool | None = None
        if plan:
            steps_all_ok = all(bool(p.get("ok")) for p in plan)

        confirmation_ok: bool | None = None
        if case.expect_confirmation is not None:
            confirmation_ok = bool(payload.get("requires_confirmation")) == case.expect_confirmation

        content_ok: bool | None = None
        missing: list[str] = []
        banned: list[str] = []
        if case.expect_contains or case.expect_not_contains:
            missing = [s for s in case.expect_contains if s not in answer]
            banned = [s for s in case.expect_not_contains if s in answer]
            content_ok = not missing and not banned

        checks = [
            v for v in (recipe_ok, steps_ok, steps_all_ok, confirmation_ok, content_ok) if v is not None
        ]

        row: dict[str, Any] = {
            "id": case.id,
            "type": case.type,
            "question": case.question,
            "note": case.note,
            "expected_recipe": case.expect_recipe,
            "expect_no_recipe": case.expect_no_recipe,
            "expect_steps": case.expect_steps,
            "recipe": recipe,
            "steps": kinds,
            "plan": plan,
            "plan_ok": steps_all_ok,
            "answer": answer[:400],
            "requires_confirmation": bool(payload.get("requires_confirmation")),
            "recipe_ok": recipe_ok,
            "steps_ok": steps_ok,
            "confirmation_ok": confirmation_ok,
            "content_ok": content_ok,
            "missing": missing,
            "banned": banned,
            "e2e_ok": all(checks) if checks else None,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "expected_docs": case.expected_docs,
            "top_docs": [r["doc_title"] for r in retrieved[:top_k]],
            "recall_at_k": (
                round(recall_at_k(retrieved, case.expected_docs, top_k), 4)
                if case.expected_docs
                else None
            ),
            "refused": bool(payload.get("refused")),
        }

        await self._reset_owner(owner)
        return row

    @staticmethod
    async def _reset_owner(owner: str) -> None:
        """把这条用例的 owner 清空，保证每次运行的起点完全一样。"""
        from ..memory import get_memory
        from ..tools.actions import get_action_store

        await get_memory().store.purge_owner(owner)
        store = await get_action_store()
        await store.purge_owner(owner)

    @staticmethod
    async def _apply_setup(case: EvalCase, owner: str) -> None:
        """按用例声明造档案 / 工单 / 流程状态。"""
        from ..memory import get_memory
        from ..tools.actions import get_action_store

        setup = case.setup or {}
        memory = get_memory()

        for predicate, value in (setup.get("facts") or {}).items():
            await memory.remember(owner, predicate, str(value), source_turn="eval:setup")

        work = setup.get("work")
        if work:
            await memory.set_work_step(
                owner,
                work["item_key"],
                work["step"],
                step_index=work.get("step_index", 0),
                force=True,
            )
            for item in work.get("materials") or []:
                await memory.mark_material(
                    owner, work["item_key"], item["name"], status=item.get("status", "submitted")
                )

        ticket = setup.get("ticket")
        if ticket:
            store = await get_action_store()
            await store.create_ticket(
                owner_id=owner,
                kind=ticket.get("kind", "permission_request"),
                payload={k: v for k, v in ticket.items() if k not in ("kind", "status")},
            )

    @staticmethod
    def _aggregate_orchestration(rows: list[dict[str, Any]]) -> dict[str, Any]:
        def rate(key: str) -> float | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(mean([1.0 if v else 0.0 for v in vals]), 4) if vals else None

        routed = [r for r in rows if r["expect_no_recipe"]]
        misrouted = [r for r in routed if r["recipe"] is not None]
        agg: dict[str, Any] = {
            "cases": len(rows),
            "e2e_success_rate": rate("e2e_ok"),
            "recipe_accuracy": rate("recipe_ok"),
            "step_sequence_accuracy": rate("steps_ok"),
            "step_success_rate": rate("plan_ok"),
            "confirmation_accuracy": rate("confirmation_ok"),
            "content_accuracy": rate("content_ok"),
            "route_cases": len(routed),
            "misroute_rate": round(len(misrouted) / len(routed), 4) if routed else None,
            "avg_latency_ms": int(mean([float(r.get("latency_ms", 0)) for r in rows])),
        }
        recall_rows = [r for r in rows if r.get("recall_at_k") is not None]
        if recall_rows:
            agg["doc_cases"] = len(recall_rows)
            agg["recall_at_k"] = round(mean([r["recall_at_k"] for r in recall_rows]), 4)
        return agg

    @classmethod
    def _by_type_orchestration(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        types = sorted({r["type"] for r in rows})
        return {t: cls._aggregate_orchestration([r for r in rows if r["type"] == t]) for t in types}

    async def _run_case(self, case: EvalCase, top_k: int, with_generation: bool) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": case.id,
            "type": case.type,
            "question": case.question,
            "expected_docs": case.expected_docs,
            "must_contain": case.must_contain,
        }

        if case.history:
            # 多轮问题：先把历史塞进会话，走完整链路（含问题改写）
            session_id = f"eval-{case.id}"
            self.service.sessions.reset(session_id)
            for turn in case.history:
                self.service.sessions.append(
                    session_id, turn.get("role", "user"), turn.get("content", "")
                )
            prep = await self.service.prepare(case.question, session_id)
            retrieved = [
                {
                    "doc_title": c.doc_title,
                    "external_id": c.metadata.get("external_id", ""),
                }
                for c in prep["chunks"]
            ]
            refused = not prep["sufficient"]
            row["resolved_query"] = prep["standalone_query"]
            row["confidence"] = prep["confidence"]
            row["coverage"] = prep["coverage"]
        else:
            result = await self.retriever.search(case.question, top_k=top_k)
            retrieved = [
                {"doc_title": c.doc_title, "external_id": c.metadata.get("external_id", "")}
                for c in result.chunks
            ]
            refused = (
                not result.chunks
                or result.confidence < settings.min_confidence
                or (
                    result.bm25_top < settings.min_bm25_anchor
                    and result.evidence < settings.min_evidence
                )
            )
            row["confidence"] = result.confidence
            row["coverage"] = result.coverage
            row["evidence"] = result.evidence

        row["top_docs"] = [r["doc_title"] for r in retrieved[:top_k]]
        row["recall_at_k"] = round(recall_at_k(retrieved, case.expected_docs, top_k), 4)
        row["mrr"] = round(reciprocal_rank(retrieved, case.expected_docs), 4)
        row["ndcg_at_k"] = round(ndcg_at_k(retrieved, case.expected_docs, top_k), 4)

        if case.type == "refusal":
            row["refusal_correct"] = 1.0 if refused else 0.0
        else:
            row["refusal_correct"] = None

        if with_generation:
            session_id = f"eval-gen-{case.id}"
            payload = await self.service.ask(case.question, session_id)
            answer = payload.get("answer_plain", "")
            row["answer"] = answer[:600]
            row["keyword_coverage"] = round(
                keyword_coverage(payload.get("answer", ""), case.must_contain), 4
            )
            row["grounding"] = payload.get("grounding_score", 0.0)
            row["answer_refused"] = payload.get("refused", False)
            row["latency_ms"] = payload.get("latency_ms", 0)
        return row

    @staticmethod
    def _aggregate(rows: list[dict[str, Any]], with_generation: bool) -> dict[str, Any]:
        # 拒答类样本没有「期望文档」，不能混进检索指标里（否则会把 Recall 拉低成假象）
        retrieval_rows = [r for r in rows if r["expected_docs"]]
        agg = {
            "retrieval_cases": len(retrieval_rows),
            "recall_at_k": mean([r["recall_at_k"] for r in retrieval_rows]),
            "mrr": mean([r["mrr"] for r in retrieval_rows]),
            "ndcg_at_k": mean([r["ndcg_at_k"] for r in retrieval_rows]),
        }
        refusal_rows = [r for r in rows if r.get("refusal_correct") is not None]
        if refusal_rows:
            agg["refusal_accuracy"] = mean([r["refusal_correct"] for r in refusal_rows])
        if with_generation:
            gen_rows = [r for r in retrieval_rows]
            agg["keyword_coverage"] = mean([r.get("keyword_coverage", 0.0) for r in gen_rows])
            agg["grounding"] = mean(
                [r.get("grounding", 0.0) for r in rows if not r.get("answer_refused")]
            )
            agg["avg_latency_ms"] = int(mean([float(r.get("latency_ms", 0)) for r in rows]))
        return agg

    @classmethod
    def _by_type(cls, rows: list[dict[str, Any]], with_generation: bool) -> dict[str, Any]:
        # 拒答类只看「是否拒答」，它的检索指标没有意义，单独放
        types = sorted({r["type"] for r in rows if r["expected_docs"]})
        return {t: cls._aggregate([r for r in rows if r["type"] == t], with_generation) for t in types}


def render_markdown(report: dict[str, Any]) -> str:
    o = report["overall"]
    degraded = report["llm"]["provider"] == "mock" or "hashing" in report["embedder"]
    lines = [
        "# 评估报告",
        "",
        f"- 数据集：`{report['dataset']}`（{report['cases']} 条，top_k={report['top_k']}）",
        f"- 时间：{report['started_at']}，耗时 {report['duration_s']}s",
        f"- LLM：`{report['llm']['provider']} / {report['llm']['model']}`",
        f"- Embedding：`{report['embedder']}`",
    ]
    if degraded:
        lines += [
            "",
            "> ⚠️ 本次评估在**离线兜底模式**下运行（mock 抽取式生成 + hashing 向量）。",
            "> 检索指标（Recall / MRR / nDCG）能反映切块与混合检索的效果，",
            "> 但「关键事实覆盖」只代表基线下限——配上真实大模型与 bge/text-embedding 后会明显更高。",
        ]
    lines += [
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| Recall@{report['top_k']} | {o['recall_at_k']:.3f} |",
        f"| MRR | {o['mrr']:.3f} |",
        f"| nDCG@{report['top_k']} | {o['ndcg_at_k']:.3f} |",
    ]
    if "refusal_accuracy" in o:
        lines.append(f"| 拒答准确率 | {o['refusal_accuracy']:.3f} |")
    if report["with_generation"]:
        lines.append(f"| 关键事实覆盖 | {o.get('keyword_coverage', 0):.3f} |")
        lines.append(f"| 引用有据率 | {o.get('grounding', 0):.3f} |")
        lines.append(f"| 平均延迟 | {o.get('avg_latency_ms', 0)} ms |")

    lines += ["", "## 分类指标", "", "| 类型 | 样本 | Recall | MRR | nDCG |", "| --- | --- | --- | --- | --- |"]
    for t, v in report["by_type"].items():
        lines.append(
            f"| {t} | {v.get('retrieval_cases', 0)} | {v['recall_at_k']:.3f} | "
            f"{v['mrr']:.3f} | {v['ndcg_at_k']:.3f} |"
        )

    lines += ["", "## 失败样例（有期望文档但一条没召回）", ""]
    fails = [r for r in report["rows"] if r["expected_docs"] and r["recall_at_k"] == 0]
    if not fails:
        lines.append("无。")
    else:
        lines += ["| id | 问题 | 期望文档 | 实际 top3 |", "| --- | --- | --- | --- |"]
        for r in fails:
            lines.append(
                f"| {r['id']} | {r['question'][:28]} | {'、'.join(r['expected_docs'])} | "
                f"{'、'.join(r['top_docs'][:3])} |"
            )
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _mark(ok: bool | None) -> str:
    if ok is None:
        return "—"
    return "✅" if ok else "❌"


def render_orchestration_markdown(report: dict[str, Any]) -> str:
    """编排评测报告：一条用例一行，能直接看出错在「选错剧本」还是「步骤串错」。"""
    o = report["overall"]
    lines = [
        "# 编排端到端评估报告",
        "",
        f"- 数据集：`{report['dataset']}`（{report['cases']} 条）",
        f"- 时间：{report['started_at']}，耗时 {report['duration_s']}s",
        f"- LLM：`{report['llm']['provider']} / {report['llm']['model']}`",
        f"- Embedding：`{report['embedder']}`",
        "",
        "> 这一组评的不是「文档找得准不准」，而是「一件多步的事有没有办对」：",
        "> 选中了哪个剧本、步骤顺序对不对、该不该代办、该被文档检索接走的问题有没有被编排抢走。",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| 端到端成功率 | {_pct(o['e2e_success_rate'])} | 下面各项全过才算过 |",
        f"| 剧本选择准确率 | {_pct(o['recipe_accuracy'])} | 命中期望剧本，或该不命中时确实没命中 |",
        f"| 步骤顺序准确率 | {_pct(o['step_sequence_accuracy'])} | plan 的步骤类型与顺序完全一致 |",
        f"| 步骤执行成功率 | {_pct(o['step_success_rate'])} | plan 里每一步都 ok |",
        f"| 确认策略准确率 | {_pct(o['confirmation_accuracy'])} | 该提案的提案、该只答的只答 |",
        f"| 答案内容准确率 | {_pct(o['content_accuracy'])} | 必含关键词全中、禁词不出现 |",
    ]
    if o.get("recall_at_k") is not None:
        lines.append(f"| 文档召回率 | {_pct(o['recall_at_k'])} | {o['doc_cases']} 条涉及文档的用例 |")
    lines += [
        f"| 误编排率 | {_pct(o.get('misroute_rate'))} | {o.get('route_cases', 0)} 条「不该编排」的用例中被接走的比例 |",
        f"| 平均延迟 | {o.get('avg_latency_ms', 0)} ms | 含检索与工具调用 |",
        "",
        "## 分类指标",
        "",
        "| 类型 | 样本 | 端到端 | 剧本 | 步骤顺序 | 内容 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t, v in report["by_type"].items():
        lines.append(
            f"| {t} | {v.get('cases', 0)} | {_pct(v['e2e_success_rate'])} | "
            f"{_pct(v['recipe_accuracy'])} | {_pct(v['step_sequence_accuracy'])} | "
            f"{_pct(v['content_accuracy'])} |"
        )

    lines += [
        "",
        "## 逐条明细",
        "",
        "| id | 问题 | 期望 | 实际剧本 | 步骤 | 剧本 | 顺序 | 执行 | 确认 | 内容 | 结果 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["rows"]:
        expect = r["expected_recipe"] or ("不编排" if r["expect_no_recipe"] else "—")
        steps = "→".join(r["steps"]) or "—"
        lines.append(
            f"| {r['id']} | {r['question'][:22]} | {expect} | {r['recipe'] or '—'} | {steps} | "
            f"{_mark(r['recipe_ok'])} | {_mark(r['steps_ok'])} | {_mark(r['plan_ok'])} | "
            f"{_mark(r['confirmation_ok'])} | {_mark(r['content_ok'])} | {_mark(r['e2e_ok'])} |"
        )

    fails = [r for r in report["rows"] if r["e2e_ok"] is False]
    lines += ["", "## 失败用例", ""]
    if not fails:
        lines.append("无。")
    else:
        for r in fails:
            detail = []
            if r["missing"]:
                detail.append(f"缺少关键词：{'、'.join(r['missing'])}")
            if r["banned"]:
                detail.append(f"出现禁词：{'、'.join(r['banned'])}")
            if r["recipe_ok"] is False:
                detail.append(f"剧本期望 {r['expected_recipe'] or '不编排'}，实际 {r['recipe'] or '未命中'}")
            if r["steps_ok"] is False:
                detail.append(f"步骤期望 {'→'.join(r['expect_steps'])}，实际 {'→'.join(r['steps']) or '无'}")
            lines += [
                f"### {r['id']} · {r['question']}",
                "",
                f"- 失败项：{'；'.join(detail) or '步骤执行未全部成功'}",
                f"- 实际答案：{r['answer'][:180]}",
                "",
            ]
    return "\n".join(lines) + "\n"
