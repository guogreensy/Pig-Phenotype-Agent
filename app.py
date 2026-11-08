"""程序入口：启动猪只姿态识别 Web 界面。

用法::

    python app.py

需要先配置 API 密钥（项目根目录 .env），否则界面会明确提示无法调用真实模型。
"""

from __future__ import annotations

import sys

import tornado.ioloop

from config import ConfigError, ensure_dirs, get_settings
from src.prompts import PromptError, load_all
from src.webapp import make_app


def main() -> int:
    try:
        settings = get_settings()
    except ConfigError as exc:
        print("[配置错误] {0}".format(exc), file=sys.stderr)
        return 2

    ensure_dirs()

    try:
        variants = load_all()
    except PromptError as exc:
        print("[提示词错误] {0}".format(exc), file=sys.stderr)
        return 2

    app = make_app(settings)
    bound_port = None
    last_error = None
    # 端口被占用时自动顺延，避免用户看到 WinError 10048 卡住
    for offset in range(0, 10):
        candidate = settings.port + offset
        try:
            app.listen(candidate, address=settings.host)
            bound_port = candidate
            break
        except OSError as exc:
            last_error = exc
    if bound_port is None:
        print(
            "[启动失败] 端口 {0}-{1} 都无法监听（{2}）。\n"
            "           可换端口，例如：set PIG_PORT=8080".format(
                settings.port, settings.port + 9, last_error
            ),
            file=sys.stderr,
        )
        return 2
    if bound_port != settings.port:
        print("提示：端口 {0} 被占用，已自动改用 {1}。".format(settings.port, bound_port))

    print("=" * 62)
    print("猪只姿态识别与可见体态描述系统")
    print("-" * 62)
    print("访问地址 : http://{0}:{1}".format(settings.host, bound_port))
    print("服务商   : {0}".format(settings.provider_label))
    print("模型     : {0}".format(settings.model))
    print("密钥     : {0}".format("已配置" if settings.has_api_key else "未配置（无法调用真实模型）"))
    print("提示词   : {0}".format("、".join("{0}({1})".format(v.key, v.short_sha) for v in variants)))
    print("并发/QPS : {0} / {1}".format(settings.concurrency, settings.qps))
    print("-" * 62)
    print("按 Ctrl+C 停止服务。图片会发送至上述服务商进行处理。")
    print("=" * 62)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
