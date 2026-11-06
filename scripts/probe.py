"""跑闭环探针，产出 Markdown / JSON 报告。

用法：
    python scripts/probe.py                        # 全部探针
    python scripts/probe.py --only noise_cleanup   # 指定一条
    python scripts/probe.py --json data/probe.json # 额外导出 JSON（给 CI/看板用）

退出码：全部通过 0，有失败 1（可直接接进 CI）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import setup_logging  # noqa: E402
from app.eval.probe import CATEGORY_LABEL, render_markdown, run_probes, summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="记忆/上下文闭环探针")
    p.add_argument("--only", default="", help="只跑指定探针，逗号分隔")
    p.add_argument("--owner", default="probe-user", help="探针使用的虚拟员工 id")
    p.add_argument("--out", default="docs/探针报告.md", help="Markdown 报告输出路径（空字符串=不写）")
    p.add_argument("--json", dest="json_out", default="", help="额外导出 JSON 的路径")
    p.add_argument("--quiet", action="store_true", help="只输出结果表")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    setup_logging("ERROR")

    only = [x.strip() for x in args.only.split(",") if x.strip()]
    reports = await run_probes(only=only or None, owner=args.owner)
    if not reports:
        print("没有匹配的探针")
        return 1

    stats = summarize(reports)
    width = max(len(r.id) for r in reports) + 2
    if not args.quiet:
        print("\n闭环探针（隔天 / 换场景 / 该忘的忘干净）\n")
    for r in reports:
        mark = "PASS" if r.passed else "FAIL"
        good = sum(1 for c in r.checks if c.ok)
        print(f"  [{mark}] {r.id:<{width}} {CATEGORY_LABEL.get(r.category, r.category):<6} "
              f"{good}/{len(r.checks)} 断言  {r.elapsed_ms:>4} ms   {r.title}")
        for c in r.failed:
            print(f"         ✗ {c.name} —— {c.detail}")
        if r.error:
            print(f"         ! 执行异常：{r.error.splitlines()[0]}")

    print(f"\n  合计 {stats['passed']}/{stats['probes']} 条探针通过，"
          f"断言 {stats['checks_passed']}/{stats['checks']}，耗时 {stats['elapsed_ms']} ms")
    for cat, v in stats["by_category"].items():
        print(f"    {CATEGORY_LABEL.get(cat, cat):<6} {v['passed']}/{v['total']}")

    meta = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(reports, meta), encoding="utf-8")
        print(f"\n  报告已写入 {out}")
    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(
            json.dumps({"meta": meta, "summary": stats, "probes": [r.to_dict() for r in reports]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  JSON 已写入 {jp}")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
