"""统一时间源（Clock）。

为什么单独抽一层：记忆系统里「时间」是核心判据——有效期、衰减、隔天回测都靠它。
如果代码里到处写 `datetime.now()`，测试就无法模拟「三天后」：
要么真等三天，要么改系统时间（危险，且不可并行）。

规则：**所有写入记忆的时间必须走 `clock.now()`**。
测试里用 `use_clock(FixedClock(...))` 就能时间旅行，验证「隔天还记得吗」「该忘的忘了吗」。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Iterator

UTC = timezone.utc


class Clock:
    """系统时钟（默认实现）。"""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def advance(self, **delta: float) -> None:  # pragma: no cover - 系统时钟不能拨
        raise NotImplementedError("系统时钟不支持时间旅行，测试请用 FixedClock")


class FixedClock(Clock):
    """固定时钟：测试 / 回测 / 演示用。

    >>> c = FixedClock("2026-11-06T10:00:00")
    >>> c.advance(days=3)
    >>> c.now().day
    9
    """

    def __init__(self, at: str | datetime | None = None) -> None:
        self._at = parse_time(at) or datetime.now(UTC)

    def now(self) -> datetime:
        return self._at

    def advance(self, **delta: float) -> None:
        self._at = self._at + timedelta(**delta)


_clock: Clock = Clock()


def now() -> datetime:
    return _clock.now()


def today() -> date:
    return _clock.now().date()


@contextmanager
def use_clock(clock: Clock) -> Iterator[Clock]:
    """临时替换时钟（异常也会还原）。"""
    global _clock
    previous = _clock
    _clock = clock
    try:
        yield clock
    finally:
        _clock = previous


def parse_time(value: str | datetime | None, default: datetime | None = None) -> datetime | None:
    """宽松解析时间：支持 ISO8601、"2026-11-06"、"2026-11-06 10:00"、纯日期。

    解析结果一律带时区（无时区按 UTC 处理），避免「本地时间 vs UTC」混用导致有效期算错。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    raw = value.strip().replace("/", "-")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        else:
            return default
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def iso(dt: datetime | None) -> str | None:
    """统一存成 ISO8601（UTC，秒级精度，便于字符串比较与排序）。"""
    if dt is None:
        return None
    dt = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def days_between(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 86400.0
