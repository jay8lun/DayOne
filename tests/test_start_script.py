"""启动脚本的纯逻辑测试（不真的起 Redis）。

重点验证两件容易出错的事：

1. **数据目录必须是 ASCII**：Windows 版 Redis 用 ANSI API 做 chdir，
   项目路径含中文会 `Can't chdir ... Illegal byte sequence`。这条是实测踩出来的。
2. **生命周期归属**：脚本自己拉起的 Redis，退出时要关掉；用户原本开着的不许碰。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from scripts import start as startmod  # noqa: E402


# ---------------- 路径判定 ----------------
def test_ascii_path_rejects_chinese():
    assert startmod._ascii_path(Path("D:/Redis/dayone")) is True
    assert startmod._ascii_path(Path("D:/kai_study/学习日志/DayOne/data")) is False


def test_pick_run_dir_skips_non_ascii(monkeypatch, tmp_path):
    """配置里给了中文目录也不要：宁可换位置，也不要启动就崩。"""
    mono = tmp_path / "中文目录"
    monkeypatch.setattr(settings, "redis_run_dir", str(mono))
    binary = Path("D:/Redis/redis-server.exe")
    picked = startmod._pick_run_dir(binary)
    assert picked is not None
    assert startmod._ascii_path(picked)
    assert picked != mono


def test_pick_run_dir_honors_ascii_config(monkeypatch, tmp_path):
    target = tmp_path / "redis-data"
    monkeypatch.setattr(settings, "redis_run_dir", str(target))
    picked = startmod._pick_run_dir(Path("D:/Redis/redis-server.exe"))
    assert picked == target


# ---------------- 配置生成 ----------------
def test_prepare_conf_appends_overrides(tmp_path):
    base = tmp_path / "redis.windows.conf"
    base.write_text("port 6379\nbind 0.0.0.0\nappendonly no\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    conf = startmod._prepare_conf(base, run_dir, local_only=True)
    text = conf.read_text(encoding="utf-8")

    # 原名项保留（说明是「追加覆盖」而不是重写）
    assert "port 6379" in text
    # 后出现的覆盖项：dir / logfile / AOF / 只监听本机
    assert f'dir "{run_dir.as_posix()}"' in text
    assert f'logfile "{run_dir.as_posix()}/' in text
    assert "appendonly yes" in text
    assert text.rstrip().endswith("bind 127.0.0.1")
    assert conf.exists() and run_dir.exists()


def test_prepare_conf_respects_aof_switch(tmp_path, monkeypatch):
    base = tmp_path / "redis.conf"
    base.write_text("port 6379\n", encoding="utf-8")
    monkeypatch.setattr(settings, "redis_aof", False)

    text = startmod._prepare_conf(base, tmp_path / "run", local_only=False).read_text(
        encoding="utf-8"
    )
    assert "appendonly no" in text
    assert "bind 127.0.0.1" not in text  # 非本地模式不强制绑回环


# ---------------- URL 解析 ----------------
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("redis://127.0.0.1:6379/0", ("127.0.0.1", 6379)),
        ("redis://localhost/1", ("localhost", 6379)),
        ("redis://10.0.0.8:6380", ("10.0.0.8", 6380)),
    ],
)
def test_parse_url(url, expected):
    assert startmod._parse(url) == expected


def test_is_local():
    assert startmod._is_local("127.0.0.1")
    assert startmod._is_local("localhost")
    assert not startmod._is_local("10.0.0.8")


# ---------------- 生命周期 ----------------
class _FakeProc:
    pid = 4321


@pytest.fixture()
def wired(monkeypatch):
    """把真实进程操作全部换掉，只留调用关系。"""
    calls: dict[str, list] = {"start": [], "stop": [], "run": []}

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr(settings, "session_backend", "redis")
    monkeypatch.setattr(settings, "redis_autostart", True)

    def fake_start(url, *, detach=False):
        calls["start"].append(detach)
        return True, "已启动（测试）", _FakeProc()

    def fake_stop(url, *, quiet=False):
        calls["stop"].append(quiet)
        return 0

    def fake_run(args):
        calls["run"].append(args.port)
        return 0

    monkeypatch.setattr(startmod, "_start_redis", fake_start)
    monkeypatch.setattr(startmod, "_stop_redis", fake_stop)
    monkeypatch.setattr(startmod, "_run_server", fake_run)
    return calls


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["start.py", *argv])
    return startmod.main()


def test_owned_redis_is_stopped_on_exit(wired, monkeypatch):
    code = _run_main(monkeypatch, ["--port", "8123"])
    assert code == 0
    assert wired["start"] == [False]  # 默认不 detach
    assert wired["stop"] == [True]  # 退出时安静地关掉
    assert wired["run"] == [8123]


def test_detach_keeps_redis_alive(wired, monkeypatch):
    code = _run_main(monkeypatch, ["--detach-redis"])
    assert code == 0
    assert wired["start"] == [True]
    assert wired["stop"] == []  # 常驻模式不许关


def test_preexisting_redis_is_not_touched(wired, monkeypatch):
    """Redis 本来就在跑（脚本没拉起）→ 退出时不能去关别人的实例。"""
    monkeypatch.setattr(
        startmod, "_start_redis", lambda url, *, detach=False: (True, "已在运行", None)
    )
    code = _run_main(monkeypatch, [])
    assert code == 0
    assert wired["stop"] == []


def test_strict_backend_aborts_when_redis_down(wired, monkeypatch):
    """SESSION_BACKEND=redis 却起不来 → 直接失败，避免「以为在用 Redis」。"""
    monkeypatch.setattr(
        startmod, "_start_redis", lambda url, *, detach=False: (False, "找不到 redis-server", None)
    )
    code = _run_main(monkeypatch, [])
    assert code == 1
    assert wired["run"] == []  # 没起服务


def test_auto_backend_degrades(wired, monkeypatch):
    monkeypatch.setattr(settings, "session_backend", "auto")
    monkeypatch.setattr(
        startmod, "_start_redis", lambda url, *, detach=False: (False, "找不到 redis-server", None)
    )
    code = _run_main(monkeypatch, [])
    assert code == 0
    assert wired["run"] != []  # 照样起服务，回退内存


def test_memory_flag_skips_redis_entirely(wired, monkeypatch):
    code = _run_main(monkeypatch, ["--memory"])
    assert code == 0
    assert wired["start"] == []
    assert wired["stop"] == []
    assert wired["run"] != []
