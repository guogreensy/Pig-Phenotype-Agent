"""阶段一：真实 API 验证（唯一会真正联网调用模型的检查脚本）。

要求说明书 9.1 的验收标准是"至少完成一次真实的图片输入和模型响应验证"。
本脚本用测试集中若干真实图片跑通一次调用，把**原始响应、状态码、耗时、
解析与校验结果**原样写入 ``results/verification/``，作为可核查证据。

没有密钥时直接退出，绝不生成伪造的成功记录。

用法::

    python scripts/verify_api.py                 # 默认 8 张
    python scripts/verify_api.py --count 5 --variant C
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from config import MissingAPIKeyError, get_settings  # noqa: E402
from src.experiment_runner import load_samples  # noqa: E402
from src.image_utils import estimate_visual_tokens, preprocess_image  # noqa: E402
from src.output_parser import parse_model_output  # noqa: E402
from src.prompts import load_prompt  # noqa: E402
from src.schema import validate_result  # noqa: E402
from src.vision_client import VisionClient, check_connectivity  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="真实 API 验证（会消耗调用额度）")
    parser.add_argument("--count", type=int, default=8, help="验证图片数量（建议 5-10）")
    parser.add_argument("--variant", default="C", help="使用哪组提示词")
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    args = parser.parse_args(argv)

    try:
        settings = get_settings(require_key=True)
    except MissingAPIKeyError as exc:
        print("[阻塞] {0}".format(exc), file=sys.stderr)
        print("       阶段一验收需要至少一次真实的图片调用。在拿到密钥前，", file=sys.stderr)
        print("       只能完成模拟测试（见 tests/），不能声称已接入模型。", file=sys.stderr)
        return 2

    config.ensure_dirs()
    prompt = load_prompt(args.variant)

    print("=" * 66)
    print("真实 API 验证")
    print("-" * 66)
    print("  服务商 : {0}".format(settings.provider_label))
    print("  模型   : {0}".format(settings.model))
    print("  端点   : {0}".format(settings.base_url))
    print("  方案   : {0}（sha256 {1}）".format(prompt.key, prompt.short_sha))
    print("=" * 66)

    reachable, message = check_connectivity(settings)
    print("端点探测：{0}".format(message))
    if not reachable:
        print("[失败] 端点不可达，终止验证。", file=sys.stderr)
        return 1

    if not config.MANIFEST_PATH.is_file():
        print("[错误] 未找到 manifest，请先构建数据集。", file=sys.stderr)
        return 2
    samples = load_samples(config.MANIFEST_PATH, split=args.split, limit=args.count)
    if not samples:
        print("[错误] 没有可用样本。", file=sys.stderr)
        return 2

    client = VisionClient(settings)
    records = []
    success = 0
    started = _dt.datetime.now()

    for index, sample in enumerate(samples, 1):
        raw = Path(sample.image_path).read_bytes()
        processed = preprocess_image(raw, max_side=settings.image_max_side,
                                     jpeg_quality=settings.jpeg_quality)
        response = client.call(processed.data_url, prompt.text)
        entry = {
            "index": index,
            "sample_id": sample.sample_id,
            "image_path": str(Path(sample.image_path).relative_to(ROOT)),
            "true_label": sample.true_label,
            "image_sent": "{0}x{1}".format(processed.width, processed.height),
            "estimated_visual_tokens": estimate_visual_tokens(processed.width, processed.height),
            "request_success": response.ok,
            "http_status": response.http_status,
            "latency_seconds": round(response.latency_seconds, 3),
            "attempt": response.attempt,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "error": response.error,
            "raw_text": response.raw_text,
        }
        if response.ok:
            success += 1
            parsed = parse_model_output(response.raw_text)
            validation = validate_result(parsed.data) if parsed.data else None
            entry["parse"] = {"json_valid": parsed.json_valid, "method": parsed.method,
                              "error": parsed.error}
            entry["schema"] = None if validation is None else {
                "valid": validation.valid,
                "posture": validation.posture,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }
            status = "OK"
            detail = "{0} -> {1}".format(
                sample.true_label,
                validation.posture if validation else "解析失败",
            )
        else:
            status = "FAIL"
            detail = response.error or ""
        records.append(entry)
        print("  [{0}/{1}] {2:<4} {3:<34} {4:.2f}s  {5}".format(
            index, len(samples), status, sample.sample_id, response.latency_seconds, detail))

    finished = _dt.datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    out_path = config.VERIFICATION_DIR / "verify_{0}_{1}.json".format(args.variant, stamp)
    payload = {
        "kind": "real_api_verification",
        "note": "真实调用记录；raw_text 为服务端原始返回，未做任何改写",
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
        "prompt_variant": prompt.key,
        "prompt_sha256": prompt.sha256,
        "params": {
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "image_max_side": settings.image_max_side,
            "jpeg_quality": settings.jpeg_quality,
        },
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "requested": len(samples),
        "success": success,
        "failed": len(samples) - success,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 66)
    print("成功 {0} / {1}".format(success, len(samples)))
    print("验证记录：{0}".format(out_path.relative_to(ROOT)))
    if success == 0:
        print("[失败] 全部调用失败，未完成阶段一验收。", file=sys.stderr)
        return 1
    print("阶段一验收：已取得真实图片输入与模型响应，可作为\"已接入\"的依据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
