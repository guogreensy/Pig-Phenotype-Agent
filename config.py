"""项目配置与 .env 加载。

本模块不依赖 python-dotenv，自行解析 .env（见 ``load_dotenv``）。
设计原则：
1. API 密钥只从环境变量或本地 .env 读取，绝不写入源码。
2. 密钥缺失时抛出中文异常，界面层可捕获并给出可操作的提示。
3. 所有路径基于项目根目录，避免工作目录不同导致找不到文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
LABELS_PATH = DATA_DIR / "labels.csv"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
VERIFICATION_DIR = RESULTS_DIR / "verification"
ENV_PATH = ROOT / ".env"

#: 支持的服务商。base_url 为 OpenAI 兼容的 chat/completions 完整地址。
PROVIDERS: Dict[str, Dict[str, str]] = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model": "glm-4.6v-flash",
        "key_env": "ZHIPUAI_API_KEY",
        "label": "智谱 GLM-4.6V-Flash（免费）",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
        "key_env": "DASHSCOPE_API_KEY",
        "label": "阿里云百炼 qwen3-vl-plus",
    },
}

DEFAULT_PROVIDER = "zhipu"


class MissingAPIKeyError(RuntimeError):
    """未配置模型 API 密钥。"""


class ConfigError(RuntimeError):
    """配置项非法。"""


def load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """解析 .env 文件为字典。

    规则：忽略空行与 # 注释；支持 ``export KEY=VAL``；成对引号会被剥除；
    不允许覆盖已存在的进程环境变量（由调用方决定优先级）。
    """
    env_path = Path(path) if path is not None else ENV_PATH
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _build_lookup() -> Dict[str, str]:
    """环境变量优先，其次 .env。"""
    lookup = dict(load_dotenv())
    lookup.update({k: v for k, v in os.environ.items() if v})
    return lookup


def get_env(name: str, default: str = "") -> str:
    """读取单个配置项（环境变量优先，其次 .env），去首尾空白。"""
    value = _build_lookup().get(name)
    if value is None or value == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    """一次性读取的运行时配置。"""

    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    qps: float
    concurrency: int
    max_calls: int
    image_max_side: int
    jpeg_quality: int
    host: str
    port: int

    @property
    def provider_label(self) -> str:
        return PROVIDERS.get(self.provider, {}).get("label", self.provider)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def _float(name: str, default: float, lookup: Dict[str, str]) -> float:
    raw = lookup.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError("环境变量 {0} 必须是数字，当前为 {1!r}".format(name, raw)) from exc


def _int(name: str, default: int, lookup: Dict[str, str]) -> int:
    raw = lookup.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError("环境变量 {0} 必须是整数，当前为 {1!r}".format(name, raw)) from exc


def get_settings(require_key: bool = False) -> Settings:
    """读取配置。

    Args:
        require_key: 为 True 且无密钥时抛出 :class:`MissingAPIKeyError`。
    """
    lookup = _build_lookup()
    provider = (lookup.get("PIG_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(
            "不支持的 PIG_PROVIDER={0!r}，可选：{1}".format(provider, "、".join(PROVIDERS))
        )
    spec = PROVIDERS[provider]
    model = (lookup.get("PIG_MODEL") or spec["model"]).strip()
    api_key = (lookup.get(spec["key_env"]) or "").strip()

    if require_key and not api_key:
        raise MissingAPIKeyError(
            "未检测到 {env}。请在项目根目录创建 .env（可参考 .env.example），"
            "写入 {env}=你的密钥后重试。当前服务商：{label}。".format(
                env=spec["key_env"], label=spec["label"]
            )
        )

    settings = Settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=spec["base_url"],
        temperature=_float("PIG_TEMPERATURE", 0.0, lookup),
        max_tokens=_int("PIG_MAX_TOKENS", 512, lookup),
        timeout_seconds=_float("PIG_TIMEOUT", 90.0, lookup),
        max_retries=_int("PIG_MAX_RETRIES", 3, lookup),
        qps=_float("PIG_QPS", 1.0, lookup),
        concurrency=_int("PIG_CONCURRENCY", 2, lookup),
        max_calls=_int("PIG_MAX_CALLS", 1200, lookup),
        image_max_side=_int("PIG_IMAGE_MAX_SIDE", 512, lookup),
        jpeg_quality=_int("PIG_JPEG_QUALITY", 90, lookup),
        host=lookup.get("PIG_HOST", "127.0.0.1"),
        port=_int("PIG_PORT", 7860, lookup),
    )
    if settings.concurrency < 1:
        raise ConfigError("PIG_CONCURRENCY 必须 >= 1")
    if settings.qps <= 0:
        raise ConfigError("PIG_QPS 必须 > 0")
    return settings


def ensure_dirs() -> None:
    """创建运行所需目录（幂等）。"""
    for path in (
        DATA_DIR,
        SAMPLES_DIR,
        SAMPLES_DIR / "dev",
        SAMPLES_DIR / "test",
        RAW_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        VERIFICATION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
