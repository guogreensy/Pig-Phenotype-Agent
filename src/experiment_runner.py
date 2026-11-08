"""批量实验编排：并发、限流、重试、断点续跑、成本闸。

关键约束（对应需求 6.3 / 6.5 / 8.3）：

* 三组提示词共用同一模型、同一预处理、同一解析与校验代码；
* 调用失败**不当场丢弃**，而是写入一条 ``request_success=False`` 的记录；
* 逐条追加写 CSV，进程中断后可按 ``(sample_id, prompt_variant)`` 续跑；
* ``max_calls`` 为硬上限，超过即停止并如实报告"未跑完"；
* 绝不使用任何预设或随机结果顶替模型输出。
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import Settings
from src.image_utils import ImageValidationError, preprocess_image
from src.output_parser import parse_model_output
from src.prompts import PromptVariant, load_prompt
from src.schema import validate_result
from src.vision_client import RateLimiter, VisionClient

RAW_FIELDS = [
    "sample_id",
    "image_path",
    "true_label",
    "prompt_variant",
    "prompt_sha256",
    "model_name",
    "model_params",
    "predicted_label",
    "description",
    "uncertain",
    "uncertainty_reason",
    "json_valid",
    "schema_valid",
    "request_success",
    "http_status",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "error_message",
    "attempt",
    "parse_method",
    "experiment_timestamp",
    "run_id",
    "raw_text",
]


@dataclass
class Sample:
    """一个待评测样本。"""

    sample_id: str
    image_path: Path
    true_label: str
    split: str = ""


@dataclass
class RunSummary:
    run_id: str
    n_planned: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_aborted: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""
    raw_path: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


class ExperimentRunner:
    """执行 "样本 × 提示词方案" 的笛卡尔积评测。"""

    def __init__(
        self,
        settings: Settings,
        samples: Sequence[Sample],
        variants: Sequence[str],
        run_id: Optional[str] = None,
        output_path: Optional[Path] = None,
        concurrency: Optional[int] = None,
        max_calls: Optional[int] = None,
        repeats: int = 1,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_side: Optional[int] = None,
    ):
        self.settings = settings
        self.samples = list(samples)
        self.variants = [v.upper() for v in variants]
        self.run_id = run_id or _dt.datetime.now().strftime("run%Y%m%d_%H%M%S")
        self.output_path = Path(output_path or self._default_path())
        self.concurrency = max(1, int(concurrency or settings.concurrency))
        self.max_calls = int(max_calls or settings.max_calls) * max(1, repeats)
        self.repeats = max(1, int(repeats))
        self.max_side = int(max_side or settings.image_max_side)
        self.progress = progress
        self._limiter = RateLimiter(settings.qps)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._call_counter = 0
        self._counter_lock = threading.Lock()
        self._prompts: Dict[str, PromptVariant] = {}
        self._prepared: Dict[str, Any] = {}

    def _default_path(self) -> Path:
        from config import RAW_DIR

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        return RAW_DIR / "{0}_raw.csv".format(self.run_id)

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        for key in self.variants:
            self._prompts[key] = load_prompt(key)
        for sample in self.samples:
            raw = Path(sample.image_path).read_bytes()
            try:
                self._prepared[sample.sample_id] = preprocess_image(
                    raw, max_side=self.max_side, jpeg_quality=self.settings.jpeg_quality
                )
            except ImageValidationError as exc:
                self._prepared[sample.sample_id] = exc

    def _client(self) -> VisionClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = VisionClient(self.settings, limiter=self._limiter)
            self._local.client = client
        return client

    # ------------------------------------------------------------------- run
    def jobs(self) -> List[Tuple[Sample, str, int]]:
        tasks = []
        for sample in self.samples:
            for variant in self.variants:
                for repeat in range(self.repeats):
                    tasks.append((sample, variant, repeat))
        return tasks

    def run(self) -> RunSummary:
        self.prepare()
        existing = self._existing_keys()
        tasks = self.jobs()
        summary = RunSummary(run_id=self.run_id, n_planned=len(tasks), raw_path=self.output_path)
        summary.n_skipped = sum(1 for s, v, _ in tasks if (s.sample_id, v) in existing)
        pending = [t for t in tasks if (t[0].sample_id, t[1]) not in existing]

        if not pending:
            summary.stop_reason = "所有任务已存在记录，无需重复调用"
            return summary

        started = time.monotonic()
        self._ensure_header()
        done = 0
        aborted = False

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._run_one, sample, variant, repeat): (sample, variant)
                       for sample, variant, repeat in pending}
            for future in as_completed(futures):
                sample, variant = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # 出乎意料的异常也要落盘
                    record = self._error_record(sample, variant, "内部错误：{0}".format(exc))
                self._append(record)
                done += 1
                if str(record.get("request_success", "")).strip().lower() != "true":
                    summary.n_failed += 1
                summary.total_tokens += int(record.get("input_tokens") or 0) + int(
                    record.get("output_tokens") or 0
                )
                if self.progress:
                    self.progress({
                        "run_id": self.run_id,
                        "done": done,
                        "total": len(pending),
                        "failed": summary.n_failed,
                        "sample_id": sample.sample_id,
                        "variant": variant,
                    })
                with self._counter_lock:
                    if self._call_counter >= self.max_calls:
                        aborted = True
                if aborted:
                    for pending_future in futures:
                        pending_future.cancel()
                    break

        summary.n_done = done
        summary.elapsed_seconds = time.monotonic() - started
        if aborted:
            summary.n_aborted = len(pending) - done
            summary.stop_reason = "达到 max_calls={0} 上限，提前停止（已停止部分未执行）".format(
                self.max_calls
            )
        return summary

    # --------------------------------------------------------------- workers
    def _run_one(self, sample: Sample, variant: str, repeat: int) -> Dict[str, Any]:
        with self._counter_lock:
            self._call_counter += 1
            if self._call_counter > self.max_calls:
                return self._error_record(sample, variant,
                                          "超过 max_calls={0} 上限，未发起调用".format(self.max_calls))

        prompt = self._prompts[variant]
        prepared = self._prepared.get(sample.sample_id)
        timestamp = _dt.datetime.now().isoformat(timespec="seconds")

        if isinstance(prepared, Exception):
            return self._base_record(sample, variant, timestamp) | {
                "request_success": "False",
                "error_message": "图片预处理失败：{0}".format(prepared),
                "schema_valid": "False",
                "json_valid": "False",
            }

        response = self._client().call(prepared.data_url, prompt.text)
        record = self._base_record(sample, variant, timestamp)
        record.update({
            "request_success": str(response.ok),
            "http_status": response.http_status or "",
            "latency_seconds": "{0:.3f}".format(response.latency_seconds),
            "input_tokens": response.input_tokens or "",
            "output_tokens": response.output_tokens or "",
            "attempt": response.attempt,
            "raw_text": response.raw_text or "",
        })

        if not response.ok:
            record["error_message"] = response.error or "调用失败（未提供原因）"
            record["json_valid"] = "False"
            record["schema_valid"] = "False"
            return record

        parsed = parse_model_output(response.raw_text)
        record["json_valid"] = str(parsed.json_valid)
        record["parse_method"] = parsed.method
        if parsed.data is None:
            record["schema_valid"] = "False"
            record["error_message"] = parsed.error or "无法解析模型返回"
            return record

        validation = validate_result(parsed.data)
        record["schema_valid"] = str(validation.valid)
        record["predicted_label"] = validation.posture or ""
        record["description"] = (validation.description or "").replace("\n", " ")
        record["uncertain"] = "" if validation.uncertain is None else str(validation.uncertain)
        record["uncertainty_reason"] = validation.uncertainty_reason or ""
        if not validation.valid:
            record["error_message"] = "Schema 校验失败：" + "；".join(validation.errors)
        elif validation.warnings:
            record["error_message"] = "（提示）" + "；".join(validation.warnings)
        return record

    def _base_record(self, sample: Sample, variant: str, timestamp: str) -> Dict[str, Any]:
        prompt = self._prompts[variant]
        return {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "true_label": sample.true_label,
            "prompt_variant": variant,
            "prompt_sha256": prompt.sha256,
            "model_name": self.settings.model,
            "model_params": "temperature={0};max_tokens={1};image_max_side={2}".format(
                self.settings.temperature, self.settings.max_tokens, self.max_side),
            "predicted_label": "",
            "description": "",
            "uncertain": "",
            "uncertainty_reason": "",
            "json_valid": "False",
            "schema_valid": "False",
            "request_success": "False",
            "http_status": "",
            "latency_seconds": "",
            "input_tokens": "",
            "output_tokens": "",
            "error_message": "",
            "attempt": "",
            "parse_method": "",
            "experiment_timestamp": timestamp,
            "run_id": self.run_id,
            "raw_text": "",
        }

    def _error_record(self, sample: Sample, variant: str, message: str) -> Dict[str, Any]:
        record = self._base_record(sample, variant,
                                   _dt.datetime.now().isoformat(timespec="seconds"))
        record["error_message"] = message
        return record

    # -------------------------------------------------------------------- io
    def _ensure_header(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.is_file() or self.output_path.stat().st_size == 0:
            with self.output_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.DictWriter(handle, fieldnames=RAW_FIELDS).writeheader()

    def _append(self, record: Dict[str, Any]) -> None:
        with self._write_lock:
            with self.output_path.open("a", encoding="utf-8-sig", newline="") as handle:
                csv.DictWriter(handle, fieldnames=RAW_FIELDS).writerow(record)
                handle.flush()
                os.fsync(handle.fileno())

    def _existing_keys(self) -> set:
        keys = set()
        if not self.output_path.is_file():
            return keys
        with self.output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("sample_id", ""), (row.get("prompt_variant") or "").upper())
                if key[0]:
                    keys.add(key)
        return keys


def load_samples(manifest_path: Path, split: str = "test", limit: Optional[int] = None,
                 labels: Optional[Sequence[str]] = None, root: Optional[Path] = None) -> List[Sample]:
    """从 manifest.csv 读取样本。"""
    from config import ROOT

    base = Path(root) if root is not None else ROOT
    samples: List[Sample] = []
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("filter_reason"):
                continue
            if split and row.get("split") != split:
                continue
            label = row.get("label") or row.get("true_label") or ""
            if labels and label not in labels:
                continue
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = base / image_path
            if not image_path.is_file():
                continue
            samples.append(Sample(
                sample_id=row["sample_id"],
                image_path=image_path,
                true_label=label,
                split=row.get("split", ""),
            ))
    if limit:
        samples = samples[:limit]
    return samples
