"""一键启动：按需拉起 Redis，再起 Web 服务。

```bash
python scripts/start.py                  # 常规：Redis 没跑就拉起来，然后起服务
python scripts/start.py --memory         # 不用 Redis，进程内会话（重启丢历史）
python scripts/start.py --redis-only     # 只把 Redis 拉起来（脱离终端常驻）
python scripts/start.py --detach-redis   # Redis 常驻，Web 停了也不退
python scripts/start.py --check          # 只体检，不启动任何进程
python scripts/start.py --stop-redis     # 关掉本机 Redis
python scripts/start.py --reload         # 开发热重载
```

设计取舍：

- **只有这个脚本会拉进程**，`app/` 里不写「应用自己启动 redis」。生产环境（compose / k8s）
  里进程由编排层管，应用去拉进程会在容器里制造孤儿进程、也没法做健康检查。
- **默认与项目同生命周期**：脚本拉起 Redis，`Ctrl+C` / 关窗口时一起停（退出前
  `SHUTDOWN SAVE` 落盘）。项目的 Redis 只服务于「开着项目看效果」这一个场景，
  不需要常驻后台；想常驻再加 `--detach-redis`（Windows 用
  `DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB`，POSIX 用 `start_new_session`）。
- 用户原本就开着的 Redis（脚本没拉起）**不归脚本管**，`Ctrl+C` 不会去关它。
- 拉起用的是**临时生成的配置**（`<redis 安装目录>/dayone/redis.dayone.conf`）：在原配置后面追加
  `dir` / `logfile` / `appendonly yes`。Redis 按顺序解析，后出现的同名项覆盖前面的，
  所以不用改你下载来的原始 conf，重装/升级也不会被覆盖。
- 数据目录必须**纯 ASCII**：Windows 版 Redis 用 ANSI API 做 `chdir`，
  项目路径带中文（`学习日志`）会直接失败 `Illegal byte sequence`。所以默认落在
  可执行文件旁边的 `dayone/`，而不是项目内的 `data/`。
- 拉不起来不硬撑：打清楚原因，`SESSION_BACKEND=auto` 时照样能起服务（自动回退内存，
  只是重启丢历史）；`SESSION_BACKEND=redis` 时直接退出，避免「以为在用 Redis」。
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

# Redis 常见安装位置（Windows：社区移植版解压目录 / Memurai；POSIX：包管理器路径）
_WINDOWS_HINTS = (
    "D:/Redis",
    "C:/Redis",
    "C:/Program Files/Redis",
    "C:/Program Files/Memurai",
    "C:/Program Files (x86)/Redis",
)
_POSIX_HINTS = ("/usr/local/bin", "/usr/bin", "/opt/homebrew/bin")

_LOG_NAME = "dayone-redis.log"
_CONF_NAME = "redis.dayone.conf"


def _ascii_path(p: Path) -> bool:
    """Redis for Windows 用 ANSI API 处理路径，非 ASCII 会让 chdir 失败。"""
    try:
        str(p).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _pick_run_dir(binary: Path) -> Path | None:
    """挑一个 Redis 能用的数据目录（纯 ASCII）。

    优先用可执行文件旁边的 dayone 子目录（跟安装位置一起，卸载就一并清掉），
    其次用户主目录、系统临时目录。项目自己的 data/ 往往带中文，放最后且不推荐。
    """
    cands: list[Path] = []
    if settings.redis_run_dir:
        cands.append(Path(settings.redis_run_dir).expanduser())
    cands += [
        binary.parent / "dayone",
        Path.home() / ".dayone" / "redis",
    ]
    tmp = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
    cands.append(Path(tmp) / "dayone-redis")
    for p in cands:
        if _ascii_path(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #
def _parse(url: str) -> tuple[str, int]:
    u = urlparse(url)
    return (u.hostname or "127.0.0.1"), (u.port or 6379)


def _is_local(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ping(url: str) -> tuple[bool, str]:
    """返回 (是否可用, 说明)。探测用 RESP2：Redis 5.x 没有 HELLO 命令。"""
    try:
        import redis  # 局部导入：没装也允许用内存后端
    except ImportError:
        return False, "redis 包未安装（pip install redis）"
    try:
        client = redis.Redis.from_url(
            url, socket_timeout=1.0, socket_connect_timeout=1.0, protocol=2
        )
        info = client.info("server")
        version = info.get("redis_version") or info.get(b"redis_version") or "?"
        return True, f"Redis {version}"
    except Exception as exc:  # noqa: BLE001 —— 连不上就是连不上，原因照实打
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 找可执行文件 / 配置
# --------------------------------------------------------------------------- #
def _binary_candidates() -> list[Path]:
    out: list[Path] = []
    if settings.redis_server_bin:
        out.append(Path(settings.redis_server_bin).expanduser())
    found = shutil.which("redis-server")
    if found:
        out.append(Path(found))
    exe = "redis-server.exe" if os.name == "nt" else "redis-server"
    hints = _WINDOWS_HINTS if os.name == "nt" else _POSIX_HINTS
    out += [Path(h) / exe for h in hints]
    out += [ROOT / "redis" / exe, ROOT.parent / "Redis" / exe]
    return out


def _find_binary() -> Path | None:
    for p in _binary_candidates():
        if p.is_file():
            return p
    return None


def _find_conf(binary: Path) -> Path | None:
    if settings.redis_server_conf:
        p = Path(settings.redis_server_conf).expanduser()
        if p.is_file():
            return p
    for name in ("redis.windows.conf", "redis.windows-service.conf", "redis.conf"):
        p = binary.parent / name
        if p.is_file():
            return p
    return None


def _prepare_conf(base: Path, run_dir: Path, local_only: bool) -> Path:
    """生成运行用配置：原样搬过来，再在末尾追加覆盖项。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    conf = run_dir / _CONF_NAME
    body = base.read_text(encoding="utf-8", errors="replace")

    posix_dir = run_dir.as_posix()
    overrides = [
        "",
        "# ---- 以下由 scripts/start.py 生成，覆盖上面同名项 ----",
        f'dir "{posix_dir}"',
        f'logfile "{posix_dir}/{_LOG_NAME}"',
        f"appendonly {'yes' if settings.redis_aof else 'no'}",
    ]
    if local_only:
        # 只给本机用：避免照抄来的 conf 里 bind 0.0.0.0 把库暴露到局域网
        overrides.append("bind 127.0.0.1")
    conf.write_text(body + "\n".join(overrides) + "\n", encoding="utf-8")
    return conf


def _spawn(binary: Path, conf: Path, run_dir: Path, *, detach: bool = False) -> subprocess.Popen:
    """启动 redis-server。

    ``detach=False``（默认）：普通子进程，跟着脚本的生命周期走——脚本退出/ Ctrl+C
    时 Redis 一起走，不留后台常驻。这是本项目选的模式：只有开着项目才需要它。

    ``detach=True``：脱离终端（Windows `DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB`，
    POSIX `start_new_session`），脚本退出后仍在跑，下次启动直接复用。
    """
    log_path = run_dir / _LOG_NAME
    log_fh = open(log_path, "ab", buffering=0)  # noqa: SIM115 —— 故意不关：子进程要写
    kwargs: dict = {
        "cwd": str(binary.parent),
        "stdin": subprocess.DEVNULL,
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    cmd = [str(binary), str(conf)]
    if not detach:
        return subprocess.Popen(cmd, **kwargs)

    if os.name == "nt":
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        # 某些终端（VS Code / 任务管理器派生的 shell）把子进程放在 job object 里，
        # 退出时会整组回收；能 breakaway 就顺带申请，失败再退回普通 detached。
        breakaway = 0x01000000
        for flags in (detached | new_group | breakaway, detached | new_group):
            try:
                return subprocess.Popen(cmd, creationflags=flags, **kwargs)
            except OSError:
                if not flags & breakaway:
                    raise
        raise RuntimeError("无法启动 redis-server")  # pragma: no cover
    return subprocess.Popen(cmd, start_new_session=True, **kwargs)


def _wait_ready(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, _ = _ping(url)
        if ok:
            return True
        time.sleep(0.4)
    return False


# --------------------------------------------------------------------------- #
# 动作
# --------------------------------------------------------------------------- #
def _start_redis(url: str, *, detach: bool = False) -> tuple[bool, str, Any]:
    """确保 Redis 可用。返回 (可用, 说明, 本次拉起的进程或 None)。"""
    host, port = _parse(url)
    ok, info = _ping(url)
    if ok:
        return True, f"已在运行（{info}，{host}:{port}）", None

    if not _is_local(host):
        return False, f"{host}:{port} 不可达，且不是本机地址——请检查远端 Redis", None

    binary = _find_binary()
    if binary is None:
        return False, (
            "找不到 redis-server。三条路：① 装一个（https://github.com/tporadowski/redis/releases）；"
            "② 在 .env 里写 REDIS_SERVER_BIN=D:/Redis/redis-server.exe；"
            "③ 用 --memory 走进程内会话"
        ), None

    base = _find_conf(binary)
    run_dir = _pick_run_dir(binary)
    if run_dir is None:
        return False, (
            "找不到纯 ASCII 的数据目录：Windows 版 Redis 不接受中文路径。"
            "请在 .env 里指定 REDIS_RUN_DIR=D:/redis-data"
        ), None
    conf = _prepare_conf(base, run_dir, local_only=True) if base else None
    if conf is None:
        # 没有配置文件也能起，但那样无法固定 dir / AOF
        return False, f"在 {binary.parent} 找不到 conf，请用 REDIS_SERVER_CONF 指定", None

    proc = _spawn(binary, conf, run_dir, detach=detach)
    if not _wait_ready(url):
        return False, f"已启动 PID {proc.pid} 但 20s 内没就绪，看 {run_dir / _LOG_NAME}", proc

    ok, info = _ping(url)
    return True, f"已启动（PID {proc.pid}，{info}，配置 {conf.name}）", proc


def _stop_redis(url: str, *, quiet: bool = False) -> int:
    ok, info = _ping(url)
    if not ok:
        if not quiet:
            print(f"Redis 没在运行（{info}）")
        return 0
    try:
        import redis

        client = redis.Redis.from_url(url, socket_timeout=3, protocol=2)
        client.shutdown(save=True)  # 先落盘再退，别把会话丢了
    except Exception:  # noqa: BLE001 —— SHUTDOWN 会先断连接，抛异常属正常
        pass
    host, port = _parse(url)
    for _ in range(25):
        if not _port_open(host, port):
            print(f"Redis 已停止（{host}:{port}，已落盘）")
            return 0
        time.sleep(0.2)
    print(f"Redis 仍在响应（{host}:{port}），可能由 Windows 服务托管，请用 services.msc 停")
    return 1


def _run_server(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        args.log_level,
    ]
    if args.reload:
        cmd.append("--reload")
    print(f"  Web:      http://{args.host}:{args.port}")
    print("  停止:     Ctrl+C（Web 与本脚本拉起的 Redis 一起停）\n")
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DayOne 一键启动")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--reload", action="store_true", help="开发热重载")
    parser.add_argument("--memory", action="store_true", help="不用 Redis，进程内会话")
    parser.add_argument("--no-redis", action="store_true", help="不自动拉起 Redis，只起服务")
    parser.add_argument("--redis-only", action="store_true", help="只拉起 Redis（脱离终端常驻）")
    parser.add_argument(
        "--detach-redis",
        action="store_true",
        help="拉起的 Redis 脱离终端常驻；默认随项目退出（一起停）",
    )
    parser.add_argument("--stop-redis", action="store_true", help="停止 Redis")
    parser.add_argument("--check", action="store_true", help="只体检，不启动")
    args = parser.parse_args()

    url = (settings.redis_url or "").strip()
    backend = (settings.session_backend or "auto").strip().lower()

    print("\nDayOne 启动器")
    print(f"  配置:     SESSION_BACKEND={backend} REDIS_URL={url or '（空）'}")

    if args.check:
        ok, info = _ping(url) if url else (False, "未配置 REDIS_URL")
        binary = _find_binary()
        conf = _find_conf(binary) if binary else None
        run_dir = _pick_run_dir(binary) if binary else None
        print(f"  Redis:    {'可用' if ok else '不可用'} · {info}")
        print(f"  可执行:   {binary or '未找到'}")
        print(f"  配置:     {conf or '未找到'}")
        print(f"  数据目录: {run_dir or '找不到纯 ASCII 路径（Windows 版 Redis 不支持中文路径）'}")
        return 0 if ok else 1

    if args.stop_redis:
        return _stop_redis(url) if url else 0

    use_redis = not args.memory and backend != "memory" and bool(url)
    redis_ok = False
    # --redis-only 的目的就是「留一个能用的 Redis」，所以它默认脱离终端；
    # 其余情况默认跟着项目走，Ctrl+C 一起停，不留后台孤儿。
    detach = args.detach_redis or args.redis_only
    owned: Any = None

    try:
        if not use_redis:
            print("  会话:     进程内（重启丢历史；要跨重启请配 REDIS_URL）")
        else:
            print(f"  Redis:    {_parse(url)[0]}:{_parse(url)[1]}", end=" ")
            if args.no_redis or not settings.redis_autostart:
                redis_ok, info = _ping(url)
                print(f"· {info}" if redis_ok else f"· 不可用（{info}）")
            else:
                redis_ok, info, owned = _start_redis(url, detach=detach)
                print(f"· {info}")
                if owned is not None and not detach:
                    print("  生命周期: 随本脚本退出（要常驻加 --detach-redis）")

            if not redis_ok:
                if backend == "redis":
                    print("\n[失败] SESSION_BACKEND=redis 但 Redis 不可用，已中止。", file=sys.stderr)
                    print("       改用 --memory，或修好 Redis 后重试。", file=sys.stderr)
                    return 1
                print("  [降级] 自动回退进程内会话——功能正常，重启会丢追问上下文。")

        if args.redis_only:
            return 0 if redis_ok else 1

        return _run_server(args)
    finally:
        # 只关「本次自己拉起来的」：用户原本就开着的 Redis 不归我们管
        if owned is not None and not detach:
            print("  正在停止 Redis ...")
            _stop_redis(url, quiet=True)


if __name__ == "__main__":
    raise SystemExit(main())
