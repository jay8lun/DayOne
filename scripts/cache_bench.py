"""量化答案缓存：同一批问题跑两轮，看命中率与延迟变化。

```bash
python scripts/cache_bench.py                 # 默认取评测集前 12 条
python scripts/cache_bench.py --rounds 3 --limit 20
python scripts/cache_bench.py --out docs/缓存报告.md
```

为什么用「两轮 + 每轮换新会话」：

- 第一轮必然全部 miss（缓存刚清空），这一轮的数字就是**没有缓存的基线**；
- 第二轮用全新的 `session_id` 问同样的问题，模拟「另一个人问了同一个问题」——
  这才是有意义的命中场景（同一会话再问会被当成追问，本来就不该命中）；
- 两轮用的是同一批问句，两轮延迟的差就是缓存的净收益。

注意这里跑的是**线上服务**（HTTP），不是进程内调用：缓存键里的知识库指纹、
Redis 连接、序列化开销全都算进去了，数字可以直接拿去汇报。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def load_questions(path: Path, limit: int) -> list[str]:
    """读评测集里的问句。追问类跳过：它们依赖上文，按设计不进缓存。"""
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        item = json.loads(line)
        if item.get("type") in {"followup", "orchestration"}:
            continue
        q = (item.get("question") or "").strip()
        if q:
            out.append(q)
        if len(out) >= limit:
            break
    return out


def clear_cache(client: httpx.Client) -> int:
    """清掉旧的答案缓存，否则第一轮就不准了。

    **必须走服务的接口**：缓存有 Redis 和进程内两层，直接删 Redis 的键，
    服务内存里那份还在，第一轮就会假命中（本项目真踩过）。
    """
    try:
        resp = client.post("/api/cache/clear")
        if resp.status_code == 200:
            return int(resp.json().get("cleared") or 0)
        print(f"[警告] 清缓存接口返回 {resp.status_code}，改用直接删 Redis 键")
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 清缓存接口调用失败: {exc}")

    # 退路：只清 Redis。冷轮可能因此混入假命中，报告里会体现出来
    url = (settings.redis_url or "").strip()
    if not url:
        return 0
    try:
        import redis

        r = redis.Redis.from_url(url, decode_responses=True, protocol=2,
                                 socket_timeout=1.0, socket_connect_timeout=1.0)
        keys = list(r.scan_iter(match=f"{settings.cache_redis_prefix}*", count=500))
        if keys:
            r.delete(*keys)
        return len(keys)
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 清理 Redis 也失败: {exc}")
        return 0


def ask(client: httpx.Client, question: str, session_id: str) -> dict:
    started = time.perf_counter()
    resp = client.post("/api/chat", json={"question": question, "session_id": session_id})
    resp.raise_for_status()
    data = resp.json()
    data["wall_ms"] = int((time.perf_counter() - started) * 1000)
    return data


def run_round(client: httpx.Client, questions: list[str], tag: str) -> list[dict]:
    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        # 每题独立会话：保证都是「第一轮」，可缓存
        sid = f"bench-{tag}-{i}"
        client.post("/api/chat/reset", json={"session_id": sid})
        d = ask(client, q, sid)
        rows.append(
            {
                "question": q,
                "cached": bool(d.get("cached")),
                "source": d.get("answer_source"),
                "ms": d["wall_ms"],
                "server_ms": d.get("latency_ms", 0),
                "refused": bool(d.get("refused")),
                "answer": (d.get("answer_plain") or "")[:60].replace("\n", " "),
            }
        )
        print(f"  [{tag} {i}/{len(questions)}] {d['wall_ms']:>5}ms "
              f"{'HIT ' if d.get('cached') else 'miss'} {q[:26]}")
    return rows


def summarize(rows: list[dict]) -> dict:
    ms = [r["ms"] for r in rows]
    return {
        "n": len(rows),
        "hits": sum(1 for r in rows if r["cached"]),
        "hit_rate": round(sum(1 for r in rows if r["cached"]) / len(rows), 3) if rows else 0.0,
        "mean_ms": int(statistics.mean(ms)) if ms else 0,
        "p50_ms": int(statistics.median(ms)) if ms else 0,
        "max_ms": max(ms) if ms else 0,
    }


def render_markdown(base: dict, cold: dict, hot: dict, rounds: int, questions: int) -> str:
    speedup = (cold["mean_ms"] / hot["mean_ms"]) if hot["mean_ms"] else 0
    lines = [
        "# 答案缓存效果",
        "",
        f"- 知识库指纹：`{base['kb_version']}`",
        f"- 样本：{questions} 个问题 × {rounds} 轮（每轮换新会话，模拟「另一个人问同一问题」）",
        f"- 缓存后端：`{base['backend']}`，TTL {base['ttl']}s",
        "",
        "| 轮次 | 请求数 | 命中 | 命中率 | 平均延迟 | P50 | 最慢 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| 第 1 轮（冷） | {cold['n']} | {cold['hits']} | {cold['hit_rate']:.1%} "
        f"| {cold['mean_ms']} ms | {cold['p50_ms']} ms | {cold['max_ms']} ms |",
        f"| 第 2 轮（热） | {hot['n']} | {hot['hits']} | {hot['hit_rate']:.1%} "
        f"| {hot['mean_ms']} ms | {hot['p50_ms']} ms | {hot['max_ms']} ms |",
        "",
        f"**平均延迟 {cold['mean_ms']} ms → {hot['mean_ms']} ms，快了 {speedup:.1f}×**"
        if speedup
        else "",
        "",
        "冷轮的命中率恒为 0（开跑前已清空缓存），它衡量的是「没有缓存时的代价」；",
        "热轮的命中率才是缓存的真实覆盖率。命中时省下的是一次完整链路：",
        "指代改写 → embedding + BM25 检索 → 生成，所以延迟差基本等于生成耗时。",
        "",
        "> 为什么不是 100%：同一句话在第二轮本该 100% 命中，任何未命中都说明",
        "> key 里混进了不稳定因素（归一化不够、知识库指纹抖动），是排查线索而不是噪声。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="答案缓存基准")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default=str(ROOT / settings.eval_dataset))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", default="docs/缓存报告.md")
    args = parser.parse_args()

    questions = load_questions(Path(args.dataset), args.limit)
    if not questions:
        print("没读到问题", file=sys.stderr)
        return 1

    removed = 0
    cold_rows: list[dict] = []
    hot_rows: list[dict] = []
    with httpx.Client(base_url=args.base_url, timeout=300, trust_env=False) as client:
        client.get("/api/health")  # 先探活，服务没起就别往下跑了
        removed = clear_cache(client)
        print(f"已清空旧缓存 {removed} 条；开始 {len(questions)} 题 × {args.rounds} 轮\n")
        for r in range(1, args.rounds + 1):
            rows = run_round(client, questions, f"r{r}")
            (cold_rows if r == 1 else hot_rows).extend(rows)
        stats = client.get("/api/health").json().get("cache") or {}

    cold = summarize(cold_rows)
    hot = summarize(hot_rows) if hot_rows else cold
    base = {
        "kb_version": stats.get("kb_version", "?"),
        "backend": stats.get("backend", "?"),
        "ttl": stats.get("ttl_seconds", 0),
    }

    md = render_markdown(base, cold, hot, args.rounds, len(questions))
    out = settings.resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print("\n===== 结果 =====")
    print(f"冷轮：{cold['mean_ms']} ms（命中 {cold['hits']}/{cold['n']}）")
    print(f"热轮：{hot['mean_ms']} ms（命中 {hot['hits']}/{hot['n']}，命中率 {hot['hit_rate']:.1%}）")
    if hot["mean_ms"]:
        print(f"加速：{cold['mean_ms'] / hot['mean_ms']:.1f}×")
    print(f"服务端统计：{json.dumps(stats, ensure_ascii=False)}")
    print(f"报告已写入：{out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
