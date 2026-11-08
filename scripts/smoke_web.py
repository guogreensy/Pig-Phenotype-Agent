"""阶段 D 端到端验收：真实启动 Web 服务并调用接口。

默认只做**不需要模型**的检查（页面、健康检查、参数校验）。
加 ``--live`` 会真实提一个问题，走完整的"提问 → 工具 → 回答"链路。

    python scripts/smoke_web.py
    python scripts/smoke_web.py --live

服务监听 127.0.0.1 的随机空闲端口，检查完立即关闭，不占用固定端口。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402
from tornado.httpserver import HTTPServer  # noqa: E402
from tornado.netutil import bind_sockets  # noqa: E402

from src.webapp import make_app  # noqa: E402


class _Server(threading.Thread):
    """在后台线程里跑 tornado，端口由系统分配。"""

    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.port: Optional[int] = None
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self) -> None:
        async def main() -> None:
            self._loop = asyncio.get_running_loop()
            # 先显式绑定拿到真实端口，再交给 HTTPServer
            sockets = bind_sockets(0, address="127.0.0.1")
            self.port = sockets[0].getsockname()[1]
            server = HTTPServer(self.app)
            server.add_sockets(sockets)
            self._ready.set()
            await asyncio.Event().wait()

        asyncio.run(main())

    def start_and_wait(self, timeout: float = 10.0) -> str:
        self.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Web 服务未能在 {0} 秒内启动".format(timeout))
        return "http://127.0.0.1:{0}".format(self.port)

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)


def _check(ok: bool, label: str, detail: str = "") -> bool:
    print("[{0}] {1}{2}".format("OK  " if ok else "FAIL", label,
                                "" if not detail else "  " + detail))
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 D 端到端验收")
    parser.add_argument("--live", action="store_true", help="真实提问一次（消耗接口额度）")
    parser.add_argument("--question", default="最近一周哪个栏位的猪采食量偏低？P2-007 要不要重点关注？")
    args = parser.parse_args(argv)

    server = _Server(make_app())
    base = server.start_and_wait()
    print("服务地址：{0}".format(base))
    print("=" * 72)

    failures = 0
    try:
        # ---------------- 页面
        page = requests.get(base + "/", timeout=10)
        html = page.text
        failures += 0 if _check(page.status_code == 200, "页面可访问", "HTTP 200") else 1
        for marker in ("智能助手", "api/agent", "agentQ", "工具调用轨迹"):
            failures += 0 if _check(marker in html, "页面含元素 {0}".format(marker)) else 1

        # ---------------- 健康检查
        health = requests.get(base + "/api/health", timeout=10).json()
        agent = health.get("agent") or {}
        tools = agent.get("tools") or []
        failures += 0 if _check(bool(agent.get("model")), "健康检查含助手模型",
                                str(agent.get("model"))) else 1
        failures += 0 if _check(len(tools) >= 8, "工具清单完整",
                                "{0} 个".format(len(tools))) else 1
        failures += 0 if _check(bool(health.get("has_api_key")), "密钥已配置") else 1

        # ---------------- 参数校验
        empty = requests.post(base + "/api/agent", json={"question": ""}, timeout=10)
        failures += 0 if _check(empty.status_code == 400, "空问题被拒", "HTTP 400") else 1
        bad = requests.post(base + "/api/agent",
                            json={"question": "x", "tools": "bogus"}, timeout=10)
        failures += 0 if _check(bad.status_code == 400, "非法工具组合被拒", "HTTP 400") else 1

        # ---------------- 真实提问
        if args.live:
            print("-" * 72)
            print("提问：{0}".format(args.question))
            response = requests.post(base + "/api/agent",
                                     json={"question": args.question, "tools": "all",
                                           "max_steps": 6},
                                     timeout=240)
            payload = response.json()
            report = payload.get("report") or {}
            answer = report.get("answer") or ""
            failures += 0 if _check(payload.get("ok") is True, "接口返回成功") else 1
            failures += 0 if _check(bool(answer), "回答非空",
                                    "{0} 字".format(len(answer))) else 1
            trace = report.get("tool_trace") or []
            failures += 0 if _check(bool(trace), "发生了工具调用",
                                    "{0} 次".format(len(trace))) else 1
            print("-" * 72)
            print(answer[:600] + ("…" if len(answer) > 600 else ""))
            print("-" * 72)
            metrics = report.get("metrics") or {}
            print("模型 {0} 次 / 工具 {1} 次 / {2} 秒".format(
                metrics.get("model_calls"), metrics.get("tool_calls"),
                metrics.get("elapsed_seconds")))
            for item in trace:
                print("  · {0}({1}) → {2}".format(
                    item.get("tool"), json.dumps(item.get("args"), ensure_ascii=False),
                    "成功" if item.get("ok") else "失败"))
            for citation in report.get("citations") or []:
                print("  [{0}] {1}（{2}）".format(citation.get("index"),
                                                 citation.get("title", "")[:50],
                                                 citation.get("license")))
        else:
            print("-" * 72)
            print("未加 --live，跳过真实提问")
    finally:
        server.stop()

    print("=" * 72)
    if failures:
        print("验收未通过：{0} 处异常".format(failures))
        return 1
    print("验收通过" + ("（含真实提问）" if args.live else "（离线）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())