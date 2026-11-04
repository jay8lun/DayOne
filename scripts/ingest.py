"""命令行入库：不启动 Web 也能建库，适合接 CI / 定时任务。

```bash
python scripts/ingest.py                 # 增量
python scripts/ingest.py --force         # 忽略哈希全量重切
python scripts/ingest.py --reset         # 清空重建
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
from app.ingest.pipeline import IngestionPipeline  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="知识库入库")
    parser.add_argument("--force", action="store_true", help="忽略内容哈希，全部重新切块")
    parser.add_argument("--reset", action="store_true", help="先清空向量库")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少篇文档")
    args = parser.parse_args()

    setup_logging("INFO")
    report = await IngestionPipeline().run(force=args.force, reset=args.reset, limit=args.limit)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
