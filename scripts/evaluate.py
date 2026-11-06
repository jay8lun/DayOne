"""命令行评估：出指标 + 写报告。

```bash
python scripts/evaluate.py                       # 检索指标（快，无 LLM 成本）
python scripts/evaluate.py --with-generation     # 加生成指标（需要真模型）
python scripts/evaluate.py --top-k 3 --limit 10
python scripts/evaluate.py --out docs/评估报告.md
python scripts/evaluate.py --orchestration       # 编排端到端（剧本命中 + 步骤串接 + 该不该代办）
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.eval.runner import (  # noqa: E402
    Evaluator,
    render_markdown,
    render_orchestration_markdown,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评估")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--orchestration", action="store_true", help="跑编排端到端评测集")
    parser.add_argument("--out", default="docs/评估报告.md", help="Markdown 报告输出路径")
    parser.add_argument("--orch-out", default="docs/评估报告_编排.md")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    setup_logging("WARNING")
    evaluator = Evaluator()

    if args.orchestration:
        report = await evaluator.run_orchestration(
            dataset_path=args.dataset, limit=args.limit, top_k=args.top_k
        )
        md_path = settings.resolve(args.orch_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_orchestration_markdown(report), encoding="utf-8")

        json_path = settings.resolve(args.json_out or "data/eval/last_orchestration.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        o = report["overall"]
        print("\n===== 编排端到端 =====")
        print(f"样本 {report['cases']} 条 | 耗时 {report['duration_s']}s")
        print(f"端到端成功率:   {_fmt(o['e2e_success_rate'])}")
        print(f"剧本选择准确率: {_fmt(o['recipe_accuracy'])}")
        print(f"步骤顺序准确率: {_fmt(o['step_sequence_accuracy'])}")
        print(f"内容准确率:     {_fmt(o['content_accuracy'])}")
        print(f"误编排率:       {_fmt(o.get('misroute_rate'))}")
        print(f"\n报告已写入: {md_path}\n")
        return 0

    report = await evaluator.run(
        dataset_path=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        with_generation=args.with_generation,
    )

    md_path = settings.resolve(args.out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    json_path = settings.resolve(args.json_out or "data/eval/last_report.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    o = report["overall"]
    print("\n===== 评估结果 =====")
    print(f"样本 {report['cases']} 条 | top_k={report['top_k']} | 耗时 {report['duration_s']}s")
    print(f"Recall@{report['top_k']}: {o['recall_at_k']:.3f}")
    print(f"MRR:            {o['mrr']:.3f}")
    print(f"nDCG@{report['top_k']}:      {o['ndcg_at_k']:.3f}")
    if "refusal_accuracy" in o:
        print(f"拒答准确率:     {o['refusal_accuracy']:.3f}")
    if report["with_generation"]:
        print(f"关键事实覆盖:   {o.get('keyword_coverage', 0):.3f}")
        print(f"引用有据率:     {o.get('grounding', 0):.3f}")
    print(f"\n报告已写入: {md_path}\n")
    return 0


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
