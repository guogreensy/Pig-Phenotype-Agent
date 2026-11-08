"""环境体检：解释器、依赖、端点连通性、提示词与数据就绪情况。

用法::

    python scripts/check_env.py

全部结论来自实测，不做推断。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.prompts import load_all  # noqa: E402
from src.vision_client import check_connectivity  # noqa: E402

REQUIRED = [
    ("requests", "HTTP 调用"),
    ("PIL", "图像处理"),
    ("pandas", "结果表格"),
    ("sklearn", "指标交叉核对"),
    ("matplotlib", "实验图表"),
    ("numpy", "数值计算"),
    ("tornado", "Web 服务"),
]
OPTIONAL = [("gradio", "已由 tornado 替代"), ("pytest", "已由 unittest 替代"),
            ("dotenv", "已自写解析替代")]


def line(char="-", width=64):
    print(char * width)


def check_python() -> bool:
    print("[1] Python 解释器")
    print("    版本: {0}".format(sys.version.split()[0]))
    print("    路径: {0}".format(sys.executable))
    ok = sys.version_info >= (3, 9)
    print("    结论: {0}".format("满足 >=3.9" if ok else "版本过低"))
    return ok


def check_packages() -> bool:
    import importlib

    print("\n[2] 依赖包")
    all_ok = True
    for name, purpose in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "已安装")
            print("    OK   {0:<12} {1:<14} {2}".format(name, str(version), purpose))
        except Exception as exc:
            all_ok = False
            print("    FAIL {0:<12} {1:<14} {2}（{3}）".format(name, "-", purpose, exc))
    for name, note in OPTIONAL:
        try:
            importlib.import_module(name)
            print("    可选 {0:<12} 已安装".format(name))
        except Exception:
            print("    可选 {0:<12} 未安装，{1}".format(name, note))
    return all_ok


def check_config() -> bool:
    print("\n[3] 配置与密钥")
    try:
        settings = config.get_settings()
    except Exception as exc:
        print("    FAIL 配置读取失败：{0}".format(exc))
        return False
    print("    服务商: {0}（{1}）".format(settings.provider, settings.provider_label))
    print("    模型  : {0}".format(settings.model))
    print("    端点  : {0}".format(settings.base_url))
    print("    参数  : temperature={0} max_tokens={1} qps={2} concurrency={3}".format(
        settings.temperature, settings.max_tokens, settings.qps, settings.concurrency))
    if settings.has_api_key:
        print("    密钥  : 已配置（长度 {0}，不显示内容）".format(len(settings.api_key)))
    else:
        print("    密钥  : 未配置 -> 无法调用真实模型。请在 .env 中填写。")
    return True


def check_endpoint() -> None:
    print("\n[4] 端点连通性（注意：401 表示可达但未授权）")
    settings = config.get_settings()
    ok, message = check_connectivity(settings)
    print("    {0} {1}".format("OK  " if ok else "FAIL", message))


def check_prompts() -> bool:
    print("\n[5] 提示词方案")
    try:
        variants = load_all()
    except Exception as exc:
        print("    FAIL {0}".format(exc))
        return False
    for variant in variants:
        print("    {0}  {1:<28} sha256={2}  字符数={3}".format(
            variant.key, variant.title, variant.short_sha, len(variant.text)))
    return True


def check_dataset() -> None:
    print("\n[6] 数据集")
    manifest = config.MANIFEST_PATH
    if not manifest.is_file():
        print("    尚未构建（运行 python scripts/build_dataset.py）")
        return
    import csv

    with manifest.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    kept = [r for r in rows if not r.get("filter_reason")]
    print("    manifest: {0} 行（合格 {1} 行）".format(len(rows), len(kept)))
    for split in ("dev", "test"):
        subset = [r for r in kept if r["split"] == split]
        counts = {}
        for row in subset:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        print("    {0:<5} 共 {1:>3} 条  {2}".format(split, len(subset), json.dumps(
            counts, ensure_ascii=False, sort_keys=True)))
    missing = [r["sample_id"] for r in kept if not (ROOT / r["image_path"]).is_file()]
    print("    缺失图片文件: {0}".format(len(missing)))


def main() -> int:
    print("=" * 64)
    print("环境体检 —— 猪只姿态识别项目")
    print("=" * 64)
    results = [check_python(), check_packages(), check_config()]
    check_endpoint()
    results.append(check_prompts())
    check_dataset()
    line("=")
    ok = all(results)
    print("总结: {0}".format("关键项全部通过" if ok else "存在未通过项，请查看上方 FAIL"))
    print("注意: 无密钥时只能运行单元测试与界面自检，不能完成真实 API 验证。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
