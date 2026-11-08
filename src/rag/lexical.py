"""词法检索索引（T3）。

为什么要保留一条词法路径
------------------------
向量检索依赖 ``embedding-3`` 接口，一旦额度或网络出问题，整个知识库就哑了。
词法检索完全离线、确定性强，既做兜底，也用来在报告里对照"纯字面匹配"与
"语义匹配"的差距。

中文处理
--------
环境里没有分词库（jieba 未安装，且不打算安装），因此**不做词级切分**，
直接以字符 n-gram（2–3 元）作为特征：中文无需分词即可工作，英文也能被
"字母片段"覆盖。代价是索引维度偏高，但这个语料规模（约 900 块）完全够用。

持久化
------
``data/kb/index/lexical.pkl`` 存向量器、稀疏矩阵与块 id 顺序。
索引是可删除、可重建的中间产物，不进版本库。
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR  # noqa: E402
from src.rag.corpus import load_chunks  # noqa: E402

INDEX_DIR = DATA_DIR / "kb" / "index"
LEXICAL_PATH = INDEX_DIR / "lexical.pkl"

#: 索引格式版本；改动特征抽取方式时递增，避免误读旧索引
INDEX_VERSION = 1

#: 字符 n-gram 范围：2 元覆盖常见词，3 元提升区分度
NGRAM_RANGE = (2, 3)


@dataclass
class LexicalHit:
    """一次词法命中。"""

    chunk_id: str
    score: float
    position: int

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "score": round(self.score, 6),
                "position": self.position}


def _normalize_queries(texts: Iterable[str]) -> List[str]:
    return [" ".join((text or "").split()).lower() for text in texts]


class LexicalIndex:
    """基于 TF-IDF 字符 n-gram 的稀疏检索。"""

    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunk_ids: Sequence[str],
                 chunks_file_sha256: str = ""):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunk_ids = list(chunk_ids)
        #: 建索引时 chunks.jsonl 的内容指纹，供"是否过期"判断使用
        self.chunks_file_sha256 = chunks_file_sha256

    # ------------------------------------------------------------------ 构建
    @classmethod
    def build(cls, chunks: Sequence[Dict[str, Any]],
              chunks_file: Optional[Path] = None) -> "LexicalIndex":
        """从切块列表构建索引。

        Args:
            chunks: 切块字典列表。
            chunks_file: 切块文件路径，若存在则记录其内容指纹。
        """
        texts = _normalize_queries(chunk["text"] for chunk in chunks)
        if not texts:
            raise ValueError("没有可索引的切块")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=NGRAM_RANGE,
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(texts)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        fingerprint = ""
        if chunks_file is not None and Path(chunks_file).is_file():
            fingerprint = hashlib.sha256(Path(chunks_file).read_bytes()).hexdigest()
        return cls(vectorizer, matrix, chunk_ids, fingerprint)

    # ------------------------------------------------------------------ 持久化
    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else LEXICAL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "ngram_range": list(NGRAM_RANGE),
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "chunk_ids": self.chunk_ids,
            "chunks_file_sha256": self.chunks_file_sha256,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LexicalIndex":
        source = Path(path) if path else LEXICAL_PATH
        if not source.is_file():
            raise FileNotFoundError(
                "词法索引不存在：{0}。请先运行 python -m src.rag.lexical".format(source)
            )
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                "索引版本不匹配（文件 {0}，当前 {1}），请重建索引".format(version, INDEX_VERSION)
            )
        return cls(payload["vectorizer"], payload["matrix"], payload["chunk_ids"],
                   payload.get("chunks_file_sha256", ""))

    # -------------------------------------------------------------------- 检索
    def search(self, query: str, k: int = 5) -> List[LexicalHit]:
        """返回得分最高的前 k 个块。"""
        text = (query or "").strip()
        if not text or k <= 0:
            return []
        query_vector = self.vectorizer.transform([_normalize_queries([text])[0]])
        # TfidfVectorizer 默认 L2 归一化，linear_kernel 即余弦相似度
        scores = linear_kernel(query_vector, self.matrix).ravel()
        if not np.any(scores):
            return []
        top = np.argsort(-scores)[: min(k, scores.shape[0])]
        hits: List[LexicalHit] = []
        for position in top:
            score = float(scores[position])
            if score <= 0:
                continue
            hits.append(LexicalHit(
                chunk_id=self.chunk_ids[int(position)],
                score=score,
                position=int(position),
            ))
        return hits

    def __len__(self) -> int:
        return len(self.chunk_ids)


def build_and_save(path: Optional[Path] = None,
                   chunks_path: Optional[Path] = None) -> LexicalIndex:
    """读取切块、构建索引并保存。"""
    source = Path(chunks_path) if chunks_path else None
    chunks = load_chunks(source)
    index = LexicalIndex.build(chunks, chunks_file=source)
    index.save(path)
    return index


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建词法检索索引")
    parser.add_argument("--out", default=None, help="索引输出路径")
    args = parser.parse_args(argv)

    index = build_and_save(Path(args.out) if args.out else None)
    print("-" * 62)
    print("词法索引 : {0} 块".format(len(index)))
    print("特征维度 : {0:,}".format(index.matrix.shape[1]))
    print("输出     : {0}".format(LEXICAL_PATH.relative_to(DATA_DIR.parent)))

    # 自检：用两个中英文查询各打一条，确认索引真的能命中
    for query in ("猪的侧卧姿态如何判定", "pig posture recognition dataset"):
        hits = index.search(query, k=3)
        print("查询 {0!r} -> {1} 条".format(query, len(hits)))
        for hit in hits[:1]:
            print("   {0} score={1:.3f}".format(hit.chunk_id, hit.score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
