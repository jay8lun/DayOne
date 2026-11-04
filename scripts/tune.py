"""参数扫描小工具：在自有评估集上找一个不那么差的融合参数组合。

```bash
python scripts/tune.py --top-k 3
```

只扫**检索侧**参数（不调用大模型），每个组合几百次检索，几十秒跑完。
注意：别在评估集上过拟合——扫出来的参数要再用一份没参与调参的样本验一遍。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.eval.dataset import load_dataset  # noqa: E402
from app.eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402


async def evaluate(retriever: HybridRetriever, cases, top_k: int) -> dict[str, float]:
    recalls, mrrs, ndcgs = [], [], []
    for case in cases:
        if not case.expected_docs:
            continue
        result = await retriever.search(case.question, top_k=top_k)
        rows = [
            {"doc_title": c.doc_title, "external_id": c.metadata.get("external_id", "")}
            for c in result.chunks
        ]
        recalls.append(recall_at_k(rows, case.expected_docs, top_k))
        mrrs.append(reciprocal_rank(rows, case.expected_docs))
        ndcgs.append(ndcg_at_k(rows, case.expected_docs, top_k))
    return {
        "recall": mean(recalls),
        "mrr": mean(mrrs),
        "ndcg": mean(ndcgs),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="检索参数扫描")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dataset", default=settings.eval_dataset)
    args = parser.parse_args()

    setup_logging("ERROR")
    cases = [c for c in load_dataset(args.dataset) if c.expected_docs]
    print(f"参与调参的样本：{len(cases)} 条（跳过拒答类）")

    grid = list(
        itertools.product(
            [0.5, 0.65, 0.8],       # rrf_weight
            [0.3, 0.85],            # bm25 b
            [3.0, 5.0, 8.0],        # title weight
        )
    )
    results = []
    for rrf_w, b, tw in grid:
        settings.rrf_weight = rrf_w
        settings.bm25_b = b
        settings.bm25_title_weight = tw
        retriever = HybridRetriever()
        await retriever.refresh_index(force=True)
        metrics = await evaluate(retriever, cases, args.top_k)
        results.append((rrf_w, b, tw, metrics))
        print(
            f"rrf_w={rrf_w:<5} b={b:<5} title_w={tw:<4} "
            f"recall@{args.top_k}={metrics['recall']:.3f} mrr={metrics['mrr']:.3f} ndcg={metrics['ndcg']:.3f}"
        )

    best = max(results, key=lambda r: (r[3]["recall"], r[3]["mrr"]))
    print("\n最佳组合：")
    print(f"  RRF_WEIGHT={best[0]}  BM25_B={best[1]}  BM25_TITLE_WEIGHT={best[2]}")
    print(f"  Recall@{args.top_k}={best[3]['recall']:.3f} MRR={best[3]['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
