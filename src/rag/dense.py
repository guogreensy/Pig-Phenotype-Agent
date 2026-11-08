"""向量检索索引（T4）。

用智谱 ``embedding-3``（2048 维）把切块文本转成稠密向量，再用 ``numpy``
做精确余弦检索。索引持久化为 ``.npz``（向量）+ ``.ids.json``（块顺序）
+ ``.meta.json``（元数据），不依赖 Chroma / FAISS 等外部库。

跨语言能力
----------
``embedding-3`` 中英文共用一个向量空间，所以中文提问可以命中英文文献——
这是 T3 的词法检索做不到的。但接口是免费配额，额度或网络出问题时会不可用，
因此检索层（T5）必须保留词法兜底。

可追溯性
--------
``.meta.json`` 记录模型名、维度、块数、**块 id 指纹**、构建时间与
**真实 token 用量**。这样报告里的"943 块 × 2048 维"不是写死的常量，
而是可以从产物里查证的事实；语料重建后指纹变化，``load()`` 会立刻报错
而不是拿旧向量配新块。

速率与重试
----------
默认每批 16 条、最小间隔 1 s，429 / 5xx 指数退避重试（最多 3 次），
401/404 这类不可恢复错误直接抛出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests

from src.rag.corpus import DATA_DIR, load_chunks

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ 常量
INDEX_DIR = DATA_DIR / "kb" / "index"
DENSE_PATH = INDEX_DIR / "dense.npz"
IDS_PATH = INDEX_DIR / "dense.ids.json"
META_PATH = INDEX_DIR / "dense.meta.json"

INDEX_VERSION = 1
EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
MODEL = "embedding-3"
_DIM = 2048
_BATCH_SIZE = 16
_MIN_INTERVAL_S = 1.0
_MAX_RETRIES = 3

#: 最近一次 embedding 调用的真实用量（供 build 写入元数据）
LAST_USAGE: Dict[str, int] = {"requests": 0, "prompt_tokens": 0, "total_tokens": 0}


@dataclass
class DenseHit:
    """一次稠密命中。"""

    chunk_id: str
    score: float
    position: int

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "score": round(self.score, 6),
                "position": self.position}


# ------------------------------------------------------------------ API 调用
def _call_embedding_api(texts: Sequence[str], api_key: str,
                        model: str = MODEL,
                        batch_size: int = _BATCH_SIZE,
                        min_interval_s: float = _MIN_INTERVAL_S,
                        max_retries: int = _MAX_RETRIES,
                        ) -> np.ndarray:
    """逐批请求 embedding-3，返回 ``(len(texts), dim)`` 的 numpy 数组。"""
    if not api_key:
        raise RuntimeError("ZHIPUAI_API_KEY 未设置，无法调用 embedding 接口")
    if not texts:
        raise ValueError("没有需要向量化的文本")

    LAST_USAGE.update({"requests": 0, "prompt_tokens": 0, "total_tokens": 0})

    session = requests.Session()
    session.headers.update({
        "Authorization": "Bearer " + api_key.strip(),
        "Content-Type": "application/json",
    })

    all_vectors: List[List[float]] = []
    last_time = 0.0
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info("开始请求 embedding-3：%d 条文本、%d 批", len(texts), total_batches)

    for batch_idx in range(total_batches):
        batch = texts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        payload = {"model": model, "input": list(batch)}

        for attempt in range(1, max_retries + 1):
            elapsed = time.monotonic() - last_time
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)

            try:
                resp = session.post(EMBEDDING_URL, json=payload, timeout=60)
                last_time = time.monotonic()
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt < max_retries:
                    delay = 2 ** attempt
                    logger.warning("embedding 请求失败（%s），%ds 后重试 (%d/%d)",
                                   exc, delay, attempt, max_retries)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    "embedding API 不可达（已重试 %d 次）: %s" % (max_retries, exc)
                ) from exc

            if resp.status_code == 200:
                data = resp.json()
                vectors = [item["embedding"] for item in data.get("data", [])]
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        "embedding 返回数量不匹配：期望 %d 条，收到 %d 条"
                        % (len(batch), len(vectors))
                    )
                usage = data.get("usage") or {}
                LAST_USAGE["requests"] += 1
                LAST_USAGE["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                LAST_USAGE["total_tokens"] += int(usage.get("total_tokens", 0))
                all_vectors.extend(vectors)
                break

            if resp.status_code in (429, 500, 502, 503):
                if attempt < max_retries:
                    try:
                        delay = float(resp.headers.get("Retry-After", 2 ** attempt))
                    except (TypeError, ValueError):
                        delay = float(2 ** attempt)
                    logger.warning("embedding API %d（%d/%d），%.0fs 后重试",
                                   resp.status_code, attempt, max_retries, delay)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    "embedding API 持续返回 %d（已重试 %d 次）"
                    % (resp.status_code, max_retries)
                )
            # 401 密钥错误、404 模型不存在等不可恢复
            resp.raise_for_status()
        else:
            raise RuntimeError("embedding 请求失败：已达最大重试次数 %d" % max_retries)

    dims = {len(vector) for vector in all_vectors}
    if len(dims) != 1:
        raise RuntimeError("embedding 返回的向量维度不一致：%s" % sorted(dims))

    logger.info("embedding-3 请求完成：%d 条、%d 批、%d tokens",
                len(all_vectors), LAST_USAGE["requests"], LAST_USAGE["total_tokens"])
    return np.array(all_vectors, dtype=np.float32)


# ------------------------------------------------------------------ 索引
def _fingerprint(chunk_ids: Sequence[str]) -> str:
    """块 id 序列的指纹，用于识别索引与语料是否配套。"""
    payload = "\n".join(chunk_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sidecar(npz_path: Path, suffix: str) -> Path:
    """由 ``dense.npz`` 推出 ``dense.ids.json`` / ``dense.meta.json``。"""
    stem = npz_path.name[:-4] if npz_path.name.endswith(".npz") else npz_path.name
    return npz_path.parent / (stem + suffix)


class DenseIndex:
    """基于 embedding-3 的稠密索引。"""

    def __init__(self, embeddings: np.ndarray, chunk_ids: Sequence[str],
                 meta: Optional[Dict[str, Any]] = None):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("向量矩阵必须是二维，实际形状 %s" % (embeddings.shape,))
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError(
                "向量行数 (%d) 与 chunk_id 数量 (%d) 不一致"
                % (embeddings.shape[0], len(chunk_ids))
            )
        self.embeddings = embeddings
        self.chunk_ids = list(chunk_ids)
        self.meta: Dict[str, Any] = dict(meta or {})

    # ---------------------------------------------------------------- 构建
    @classmethod
    def build(cls, chunks: Sequence[Dict[str, Any]], api_key: str, *,
              model: str = MODEL,
              batch_size: int = _BATCH_SIZE,
              chunks_file: Optional[Path] = None) -> "DenseIndex":
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            raise ValueError("没有可索引的切块")

        vectors = _call_embedding_api(texts, api_key, model=model,
                                      batch_size=batch_size)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0        # 兜底：零向量不参与缩放
        vectors = vectors / norms

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        meta: Dict[str, Any] = {
            "index_version": INDEX_VERSION,
            "model": model,
            "dim": int(vectors.shape[1]),
            "chunks": len(chunk_ids),
            "chunk_ids_sha256": _fingerprint(chunk_ids),
            "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "usage": dict(LAST_USAGE),
        }
        if chunks_file is not None and Path(chunks_file).is_file():
            meta["chunks_file"] = str(Path(chunks_file))
            meta["chunks_file_sha256"] = hashlib.sha256(
                Path(chunks_file).read_bytes()).hexdigest()
        return cls(vectors, chunk_ids, meta)

    # ---------------------------------------------------------------- 持久化
    def save(self, path: Optional[Path] = None) -> Path:
        npz_path = Path(path) if path else DENSE_PATH
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embeddings=self.embeddings)

        ids_path = _sidecar(npz_path, ".ids.json")
        ids_path.write_text(json.dumps(self.chunk_ids, ensure_ascii=False),
                            encoding="utf-8")

        meta = dict(self.meta)
        meta.setdefault("index_version", INDEX_VERSION)
        meta["chunks"] = len(self.chunk_ids)
        meta["dim"] = int(self.embeddings.shape[1])
        meta["chunk_ids_sha256"] = _fingerprint(self.chunk_ids)
        _sidecar(npz_path, ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return npz_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DenseIndex":
        npz_path = Path(path) if path else DENSE_PATH
        ids_path = _sidecar(npz_path, ".ids.json") if path \
            else IDS_PATH
        meta_path = _sidecar(npz_path, ".meta.json") if path \
            else META_PATH

        if not npz_path.is_file():
            raise FileNotFoundError(
                "稠密索引不存在：%s。请先运行 python -m src.rag.dense" % npz_path)
        if not ids_path.is_file():
            raise FileNotFoundError(
                "稠密索引的 id 列表不存在：%s（索引不完整，请重建）" % ids_path)

        data = np.load(npz_path)
        embeddings = data["embeddings"]
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))

        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dim") not in (None, int(embeddings.shape[1])):
                raise ValueError(
                    "索引维度与元数据不符（文件 %s，元数据 %s），请重建索引"
                    % (embeddings.shape[1], meta.get("dim")))
            if meta.get("chunks") not in (None, len(chunk_ids)):
                raise ValueError(
                    "索引块数与元数据不符（文件 %d，元数据 %s），请重建索引"
                    % (len(chunk_ids), meta.get("chunks")))
            expected = meta.get("chunk_ids_sha256")
            if expected and expected != _fingerprint(chunk_ids):
                raise ValueError("块 id 指纹与元数据不符，请重建索引")
        return cls(embeddings, chunk_ids, meta)

    # ---------------------------------------------------------------- 检索
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[DenseHit]:
        """余弦检索 Top-k。查询向量可以直接传入原始 embedding，内部会归一化。"""
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "查询向量维度 %d != 索引维度 %d"
                % (q.shape[0], self.embeddings.shape[1]))
        if k <= 0:
            return []

        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        scores = self.embeddings @ q
        top = np.argsort(-scores)[: min(k, scores.shape[0])]
        hits: List[DenseHit] = []
        for position in top:
            score = float(scores[position])
            if score <= 0:
                continue
            hits.append(DenseHit(chunk_id=self.chunk_ids[int(position)],
                                 score=score,
                                 position=int(position)))
        return hits

    def check_freshness(self, chunks_path: Optional[Path] = None) -> None:
        """校验索引与当前切块文件是否配套（语料重建后应重新建索引）。"""
        source = Path(chunks_path) if chunks_path else DATA_DIR / "kb" / "chunks.jsonl"
        recorded = self.meta.get("chunks_file_sha256")
        if not recorded or not source.is_file():
            return
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != recorded:
            raise ValueError(
                "切块文件已变化（索引指纹 %s…，当前 %s…），请重建稠密索引"
                % (recorded[:10], actual[:10]))

    def __len__(self) -> int:
        return len(self.chunk_ids)


# ------------------------------------------------------------------ CLI
def build_and_save(api_key: Optional[str] = None,
                   chunks_path: Optional[Path] = None,
                   out_path: Optional[Path] = None,
                   batch_size: int = _BATCH_SIZE) -> DenseIndex:
    source = Path(chunks_path) if chunks_path else DATA_DIR / "kb" / "chunks.jsonl"
    chunks = load_chunks(source)
    index = DenseIndex.build(chunks, api_key=api_key, batch_size=batch_size,
                             chunks_file=source)
    index.save(out_path)
    return index


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建稠密向量检索索引")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE,
                        help="每批请求的文本条数")
    parser.add_argument("--out", default=None, help="索引输出路径（.npz）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from config import load_dotenv
    env = load_dotenv()
    key = (env.get("ZHIPUAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("ZHIPUAI_API_KEY 未设置，请检查 .env")

    index = build_and_save(api_key=key,
                           out_path=Path(args.out) if args.out else None,
                           batch_size=args.batch_size)
    print("-" * 62)
    print("稠密索引 : {0} 块 × {1} 维".format(len(index), index.embeddings.shape[1]))
    print("块指纹   : {0}…".format(index.meta.get("chunk_ids_sha256", "")[:16]))
    print("token    : {0:,}".format(index.meta.get("usage", {}).get("total_tokens", 0)))
    print("输出     : {0} / {1} / {2}".format(
        DENSE_PATH.relative_to(DATA_DIR.parent),
        IDS_PATH.relative_to(DATA_DIR.parent),
        META_PATH.relative_to(DATA_DIR.parent)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())