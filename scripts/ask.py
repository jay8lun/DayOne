"""命令行问答：验证链路是否通，不用打开浏览器。

```bash
python scripts/ask.py "报销要在多少天内提交？"
python scripts/ask.py            # 进入交互模式，支持多轮
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import setup_logging  # noqa: E402
from app.rag.service import get_service  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="问答调试")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args()

    setup_logging("WARNING")
    service = get_service()

    if args.question:
        payload = await service.ask(args.question)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("\n" + payload["answer"] + "\n")
            print("—" * 60)
            for s in payload["citations"]:
                flag = "✓" if s["used"] else " "
                print(f"[{s['n']}]{flag} {s['doc_title']} > {s['heading']}  ({s['score']:.2f})")
            print(f"\nconfidence={payload['confidence']:.2f} grounding={payload['grounding_score']:.2f} "
                  f"latency={payload['latency_ms']}ms")
        return 0

    session_id = service.sessions.new_id()
    print("交互模式（输入 exit 退出，/reset 清空上下文）")
    while True:
        try:
            line = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        if line == "/reset":
            await service.reset(session_id)
            print("（上下文已清空）")
            continue
        payload = await service.ask(line, session_id)
        print("\n助手:" + payload["answer"])
        if payload["citations"]:
            used = [c for c in payload["citations"] if c["used"]]
            print("\n来源: " + "；".join(f"[{c['n']}] {c['doc_title']}" for c in used))
        print(f"(confidence={payload['confidence']:.2f}, latency={payload['latency_ms']}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
