"""多模态视觉模型 API 客户端.

设计要点
--------
1. **统一接口**：智谱与阿里云百炼都走 OpenAI 兼容的 chat/completions，
   只差 base_url / model / 若干私有参数，因此一个客户端即可覆盖。
2. **绝不伪造结果**：任何失败都返回 ``ok=False`` 并带错误原因，
   调用方据此记录失败样本，不允许用预设答案顶替。
3. **可重试错误与非可重试错误分开**：401/403/400 立即失败，
   429/5xx/超时/连接错误按指数退避重试。
4. **限流**：内置令牌桶，避免免费额度触发 QPS 限制。
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import Settings

#: 这些 HTTP 状态码值得重试；其余（尤其是鉴权类）重试无意义。
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RateLimiter:
    """简单令牌桶：保证平均 QPS 不超过设定值。"""

    def __init__(self, qps: float, clock=time.monotonic, sleep=time.sleep):
        if qps <= 0:
            raise ValueError("qps 必须大于 0")
        self._interval = 1.0 / float(qps)
        self._clock = clock
        self._sleep = sleep
        self._next_at = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """阻塞到可以发起下一次请求，返回实际等待秒数。"""
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            self._sleep(wait)
        return wait


@dataclass
class ModelResponse:
    """一次调用的完整记录。"""

    ok: bool
    raw_text: str = ""
    http_status: Optional[int] = None
    latency_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    attempt: int = 1
    model: str = ""
    notes: List[str] = field(default_factory=list)


def build_messages(data_url: str, prompt: str, system: Optional[str] = None) -> List[Dict[str, Any]]:
    """拼装 OpenAI 兼容的多模态 messages。"""
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def _extract_text(message: Any) -> str:
    """从 ``choices[0].message`` 里取出文本，兼容 content 为数组的返回。"""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def _extract_usage(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _error_message(response: requests.Response) -> str:
    """把服务端错误体压缩成一句可读的话。"""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip().replace("\n", " ")
        return text[:300] or "HTTP {0}".format(response.status_code)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return "{0}: {1}".format(err.get("code") or err.get("type") or "error", err.get("message"))
        if isinstance(err, str):
            return err
        for key in ("message", "msg", "detail"):
            if isinstance(body.get(key), str):
                return body[key]
    return str(body)[:300]


class VisionClient:
    """视觉大模型客户端。"""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None,
                 limiter: Optional[RateLimiter] = None, sleep=time.sleep):
        self.settings = settings
        self._session = session or requests.Session()
        self._limiter = limiter if limiter is not None else RateLimiter(settings.qps)
        self._sleep = sleep
        self._extra_body: Dict[str, Any] = {}
        # 混合思考模型支持 thinking 开关；强制思考模型（名字含 thinking）不要下发该参数
        if settings.provider == "zhipu" and "thinking" not in settings.model:
            self._extra_body = {"thinking": {"type": "disabled"}}

    # ---------------------------------------------------------------- payload
    def build_payload(self, messages: List[Dict[str, Any]], include_extra: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if include_extra and self._extra_body:
            payload.update(self._extra_body)
        return payload

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        headers = {
            "Authorization": "Bearer {0}".format(self.settings.api_key),
            "Content-Type": "application/json",
        }
        return self._session.post(
            self.settings.base_url,
            headers=headers,
            json=payload,
            timeout=self.settings.timeout_seconds,
        )

    # ------------------------------------------------------------------- call
    def call(self, data_url: str, prompt: str, system: Optional[str] = None) -> ModelResponse:
        """调用模型并返回结构化记录（失败也返回，不抛异常）。"""
        if not self.settings.api_key:
            return ModelResponse(
                ok=False,
                error="未配置 API 密钥（{0}），无法调用模型。".format(self.settings.provider),
                model=self.settings.model,
            )

        messages = build_messages(data_url, prompt, system)
        notes: List[str] = []
        include_extra = True
        last_error = "未知错误"
        max_attempts = max(1, self.settings.max_retries)

        for attempt in range(1, max_attempts + 1):
            self._limiter.acquire()
            payload = self.build_payload(messages, include_extra=include_extra)
            started = time.monotonic()
            try:
                response = self._post(payload)
            except requests.Timeout:
                elapsed = time.monotonic() - started
                last_error = "请求超时（>{0:g}s）".format(self.settings.timeout_seconds)
                if attempt < max_attempts:
                    self._backoff(attempt, None, None)
                    continue
                return ModelResponse(ok=False, error=last_error, latency_seconds=elapsed,
                                     attempt=attempt, model=self.settings.model, notes=notes)
            except requests.RequestException as exc:
                elapsed = time.monotonic() - started
                last_error = "网络请求失败：{0}".format(exc.__class__.__name__)
                if attempt < max_attempts:
                    self._backoff(attempt, None, None)
                    continue
                return ModelResponse(ok=False, error=last_error, latency_seconds=elapsed,
                                     attempt=attempt, model=self.settings.model, notes=notes)

            elapsed = time.monotonic() - started
            status = response.status_code

            if status == 200:
                try:
                    body = response.json()
                except ValueError:
                    return ModelResponse(
                        ok=False, http_status=status, latency_seconds=elapsed,
                        error="服务端返回的不是合法 JSON", attempt=attempt,
                        model=self.settings.model, notes=notes,
                    )
                choices = body.get("choices") or []
                text = _extract_text(choices[0].get("message")) if choices else ""
                input_tokens, output_tokens = _extract_usage(body)
                if not text.strip():
                    return ModelResponse(
                        ok=False, http_status=status, latency_seconds=elapsed,
                        error="模型返回内容为空", attempt=attempt,
                        model=self.settings.model, notes=notes,
                        input_tokens=input_tokens, output_tokens=output_tokens,
                    )
                return ModelResponse(
                    ok=True, raw_text=text, http_status=status, latency_seconds=elapsed,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    attempt=attempt, model=self.settings.model, notes=notes,
                )

            # 服务端拒绝 thinking 参数时，去掉私有参数再试一次
            if status == 400 and include_extra and self._extra_body:
                detail = _error_message(response).lower()
                if "thinking" in detail or "unknown" in detail or "invalid" in detail:
                    notes.append("服务端不接受 thinking 参数，已自动去掉后重试")
                    include_extra = False
                    continue

            last_error = "HTTP {0}：{1}".format(status, _error_message(response))
            retryable = status in RETRYABLE_STATUS
            if retryable and attempt < max_attempts:
                self._backoff(attempt, response.headers.get("Retry-After"), status)
                continue
            hint = ""
            if status in (401, 403):
                hint = "（密钥无效或无调用权限，请检查 .env 中的密钥）"
            elif status == 429:
                hint = "（触发限流，可降低 PIG_QPS 或 PIG_CONCURRENCY）"
            return ModelResponse(
                ok=False, http_status=status, latency_seconds=elapsed,
                error=last_error + hint, attempt=attempt,
                model=self.settings.model, notes=notes,
            )

        return ModelResponse(ok=False, error=last_error, attempt=max_attempts,
                             model=self.settings.model, notes=notes)

    # ---------------------------------------------------------------- helpers
    def _backoff(self, attempt: int, retry_after: Optional[str],
                 status: Optional[int] = None) -> None:
        """指数退避；429（免费模型拥堵）等待更久。"""
        base = 5.0 * attempt if status == 429 else float(2 ** (attempt - 1))
        delay = min(45.0, base) + random.uniform(0.0, 1.5)
        if retry_after:
            try:
                delay = max(delay, min(60.0, float(retry_after)))
            except (TypeError, ValueError):
                pass
        self._sleep(delay)


def check_connectivity(settings: Settings) -> Tuple[bool, str]:
    """探测端点是否可达；有真实密钥时同时验证鉴权是否通过。"""
    token = settings.api_key or "probe"
    try:
        response = requests.post(
            settings.base_url,
            headers={"Authorization": "Bearer {0}".format(token), "Content-Type": "application/json"},
            json={"model": settings.model, "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=20,
        )
    except requests.RequestException as exc:
        return False, "无法连接 {0}：{1}".format(settings.base_url, exc.__class__.__name__)
    status = response.status_code
    if status == 200:
        return True, "端点可达且鉴权通过（HTTP 200）"
    if status in (400, 404, 422, 429):
        return True, "端点可达，鉴权通过但请求被拒（HTTP {0}）".format(status)
    if status in (401, 403):
        return False, "端点可达但鉴权失败（HTTP {0}），请检查密钥".format(status)
    return False, "端点返回异常状态 HTTP {0}".format(status)
