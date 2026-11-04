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
