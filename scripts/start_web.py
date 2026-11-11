"""一键启动：探测端口 → 拉起服务 → 等就绪 → 打开浏览器。

用法::

    python scripts/start_web.py            # 自动选端口并打开浏览器
    python scripts/start_web.py --port 8080 --no-browser

设计要点：

* 只依赖标准库，不需要安装任何东西；
* 端口从 7860 起自动顺延，**浏览器地址与实际监听端口始终一致**；
* 已有实例在跑时不再重复起进程，只打开浏览器；
* 关窗或 Ctrl+C 时终止子进程，端口随之释放。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ENTRY = PROJECT_ROOT / "app.py"
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
# 与 app.py 的顺延范围保持一致（7860-7869）
PORT_SPAN = 10
HEALTH_PATH = "/api/health"


# ------------------------------------------------------------------ 纯函数
def probe_port(host: str = DEFAULT_HOST, preferred: int = DEFAULT_PORT,
               span: int = PORT_SPAN) -> Optional[int]:
    """返回第一个能绑定成功的端口；全被占用则返回 None。"""
    for offset in range(span):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    return None


def build_url(host: str, port: int, path: str = "/") -> str:
    """拼访问地址。监听 0.0.0.0 时浏览器要用 127.0.0.1。"""
    display = "127.0.0.1" if host in ("", "0.0.0.0") else host
    return "http://{0}:{1}{2}".format(display, port, path)


def is_healthy(url: str, timeout: float = 1.5,
               opener: Callable = urllib.request.urlopen) -> bool:
    """请求 /api/health，能拿到 ok=true 才算就绪。"""
    try:
        with opener(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return False
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return bool(isinstance(payload, dict) and payload.get("ok"))


def find_healthy_port(host: str, preferred: int, span: int = PORT_SPAN,
                      probe: Callable = is_healthy) -> Optional[int]:
    """在端口范围内找已经在服务的实例。"""
    for offset in range(span):
        port = preferred + offset
        if probe(build_url(host, port, HEALTH_PATH)):
            return port
    return None


def wait_until_ready(host: str, preferred: int, timeout: float,
                     span: int = PORT_SPAN, interval: float = 0.3,
                     finder: Callable = find_healthy_port,
                     sleep: Callable = time.sleep,
                     clock: Callable = time.monotonic) -> Optional[int]:
    """轮询直到服务就绪，返回实际端口；超时返回 None。"""
    deadline = clock() + timeout
    while True:
        port = finder(host, preferred, span)
        if port is not None:
            return port
        if clock() >= deadline:
            return None
        sleep(interval)


def describe_env(path: Path = ENV_FILE) -> Optional[str]:
    """.env 缺失或不含密钥时返回提醒文本，否则返回 None。"""
    if not path.exists():
        return "未找到 .env：页面可以打开，但图像识别与助手无法调用模型。"
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return "无法读取 .env：请检查文件权限。"
    if "API_KEY" not in content:
        return ".env 中没有 API_KEY：页面可以打开，但无法调用模型。"
    return None


def open_browser(url: str, opener: Callable = webbrowser.open) -> bool:
    try:
        opener(url)
        return True
    except Exception:  # noqa: BLE001 - 打开浏览器失败不应中断服务
        return False


# ------------------------------------------------------------------ 主流程
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动猪只表型智能分析助手")
    parser.add_argument("--host", default=os.environ.get("PIG_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PIG_PORT", DEFAULT_PORT)))
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="等待服务就绪的秒数")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--python", default=sys.executable,
                        help="启动 app.py 使用的解释器")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    # 输出被重定向时 Python 默认会缓冲，这里改成行缓冲，保证进度实时可见
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    # 已经有实例在跑：直接开浏览器，避免重复占端口
    running = find_healthy_port(args.host, args.port)
    if running is not None:
        url = build_url(args.host, running)
        print("[已运行] 服务已在 {0} 上响应。".format(url))
        if not args.no_browser and not open_browser(url):
            print("请手动在浏览器打开：{0}".format(url))
        return 0

    port = probe_port(args.host, args.port)
    if port is None:
        print("[启动失败] 端口 {0}-{1} 都被占用，可用 --port 指定其它端口。".format(
            args.port, args.port + PORT_SPAN - 1), file=sys.stderr)
        return 2

    hint = describe_env()
    if hint:
        print("[提示] {0}".format(hint))

    env = dict(os.environ)
    env["PIG_HOST"] = args.host
    env["PIG_PORT"] = str(port)

    print("[启动] 正在启动服务，端口 {0} ……".format(port))
    process = subprocess.Popen([args.python, str(APP_ENTRY)],
                               cwd=str(PROJECT_ROOT), env=env)
    try:
        actual = wait_until_ready(args.host, port, args.timeout)
        if actual is None:
            print("[启动失败] 等待 {0:.0f} 秒仍未就绪，请检查上方输出的报错。".format(
                args.timeout), file=sys.stderr)
            return 3
        url = build_url(args.host, actual)
        print("[就绪] {0}".format(url))
        if not args.no_browser and not open_browser(url):
            print("浏览器未能自动打开，请手动访问：{0}".format(url))
        print("关闭本窗口（或按 Ctrl+C）即可停止服务。")
        process.wait()
        return process.returncode or 0
    except KeyboardInterrupt:
        print("\n[停止] 收到中断信号，正在关闭服务……")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())