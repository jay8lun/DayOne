"""Redis 会话后端自检：部署前跑一遍，确认「会话真的外置了」。

检查四件事，任何一项不过就返回非 0：

1. **连通**：ping 得通，密码对；
2. **读写**：写入一条消息能按 order 读回来；
3. **TTL**：键有过期时间（不然会话会永久堆积）；
4. **多实例**：两个 store 指向同一个 Redis，第二个能读到第一个写的——
   这才是「多副本共享」的真正含义。

```bash
python scripts/check_redis.py                          # 读 .env 的 REDIS_URL
python scripts/check_redis.py --url redis://host:6379/0
python scripts/check_redis.py --fake                   # 没有 Redis 时验证命令语义
```
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.rag.sessions import RedisSessionStore, _mask_url  # noqa: E402


def _client(url: str, fake: bool):
    if fake:
        import fakeredis

        return fakeredis.FakeRedis.from_url(url, decode_responses=True)
    import redis

    return redis.Redis.from_url(
        url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Redis 会话后端自检")
    parser.add_argument("--url", default=settings.redis_url or "redis://127.0.0.1:6379/0")
    parser.add_argument("--fake", action="store_true", help="用 fakeredis 验证命令语义（无需真 Redis）")
    parser.add_argument("--keep", action="store_true", help="保留测试键，便于用 redis-cli 观察")
    args = parser.parse_args()

    print(f"目标: {_mask_url(args.url)}{' (fake)' if args.fake else ''}")
    client = _client(args.url, args.fake)
    session_id = f"selfcheck-{uuid.uuid4().hex[:8]}"
    store = RedisSessionStore(args.url, client=client, max_turns=3, ttl_minutes=30)
    ok = True

    # 1) 连通
    try:
        pong = client.ping()
        print(f"[{'PASS' if pong else 'FAIL'}] 连通性: PING → {pong}")
        ok = ok and bool(pong)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 连通性: {exc}")
        print("\n排查建议：Redis 起了吗（docker compose up -d redis）？端口/密码对吗？")
        return 1

    # 2) 读写
    store.append(session_id, "user", "一线城市住宿标准是多少")
    store.append(session_id, "assistant", "450 元/晚[[1]]")
    history = store.history(session_id)
    read_ok = [h["content"] for h in history] == ["一线城市住宿标准是多少", "450 元/晚[[1]]"]
    print(f"[{'PASS' if read_ok else 'FAIL'}] 读写顺序: {len(history)} 条")
    ok = ok and read_ok

    # 3) TTL
    key = f"{store.prefix}{session_id}"
    ttl = client.ttl(key)
    ttl_ok = 0 < ttl <= 30 * 60
    print(f"[{'PASS' if ttl_ok else 'FAIL'}] TTL: {ttl}s（应为 0 < ttl ≤ 1800）")
    ok = ok and ttl_ok

    # 4) 多实例共享
    other = RedisSessionStore(args.url, client=client, max_turns=3)
    shared = [h["content"] for h in other.history(session_id)]
    share_ok = shared == ["一线城市住宿标准是多少", "450 元/晚[[1]]"]
    print(f"[{'PASS' if share_ok else 'FAIL'}] 多实例共享: 另一个实例读到 {len(shared)} 条")
    ok = ok and share_ok

    if not args.keep:
        store.reset(session_id)
        print(f"[INFO] 已清理测试键 {key}")
    else:
        print(f"[INFO] 保留测试键 {key}（redis-cli LRANGE {key} 0 -1 可查看）")

    print("\n结论:", "Redis 会话后端可用" if ok else "存在问题，见上方 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
