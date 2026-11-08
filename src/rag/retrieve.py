"""统一检索接口（T5）。

把 T3 词法索引与 T4 稠密索引收敛成一条对外入口：**稠密为主、词法兜底**，
并支持两种方式的结果融合（RRF）。

设计目标
--------
1. 调用方只需要 ``Retriever.retrieve(query)``，不需要关心底下用的是哪个索引；
2. 稠密不可用（缺密钥 / 额度耗尽 / 网络抖动）时**自动降级**到词法，并在
   返回值里如实声明 ``mode`` 与降级原因——不会假装是语义检索；
3. 每条结果都带 ``start`` / ``end`` 字符偏移、标题、许可、来源 URL，
   可精确回到 ``data/kb/raw/<doc_id>.txt`` 的原文段落（可溯源引用）。

三种模式
--------
* ``dense``    —— 语义检索（embedding-3 余弦）
* ``lexical``  —— 词法检索（TF-IDF 字符 n-gram），无网络的兜底
* ``hybrid``   —— 两者 Top-k 做 RRF 融合，兼顾语义与字面命中
* ``auto``     —— 默认：优先稠密，出错自动转词法
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rag.corpus import RAW_DIR, load_chunks
from src.rag.dense import DenseIndex, _call_embedding_api
from src.rag.lexical import LexicalIndex

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ 融合参数
#: RRF 融合常数（经典取值，越大越平滑）
RRF_K = 60
#: 两路在融合中的权重：稠密是主路，词法只做召回补充
DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.5
#: 词法得分低于此值视为字面噪声，不参与融合
LEXICAL_MIN_SCORE = 0.01


@dataclass
class RetrievalResult:
    """一条可溯源检索结果。"""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    license: str
    start: int
    end: int
    text: str
    score: float
    source: str          # dense / lexical / hybrid
    rank: int = 0
    #: 融合时两路各自的原始得分（单路检索为 None），用于审计排序依据
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "url": self.url,
            "license": self.license,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "score": round(self.score, 6),
            "source": self.source,
            "rank": self.rank,
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "lexical_score": None if self.lexical_score is None
            else round(self.lexical_score, 6),
        }


@dataclass
class RetrievalResponse:
    """一次检索的完整结果 + 模式说明。"""

    results: List[RetrievalResult] = field(default_factory=list)
    mode: str = "none"                    # dense / lexical / hybrid / dense-fallback-lexical
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
            "results": [r.to_dict() for r in self.results],
        }

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)


class Retriever:
    """统一检索入口。"""

    def __init__(self, api_key: Optional[str] = None,
                 dense: Optional[DenseIndex] = None,
                 lexical: Optional[LexicalIndex] = None,
                 chunks: Optional[Dict[str, Dict[str, Any]]] = None):
        self.api_key = (api_key or "").strip()
        self.dense = dense if dense is not None else DenseIndex.load()
        self.lexical = lexical if lexical is not None else LexicalIndex.load()
        auto_loaded_chunks = chunks is None
        if auto_loaded_chunks:
            self.chunks = {c["chunk_id"]: c for c in load_chunks()}
        else:
            self.chunks = chunks
        self._verify_index_coverage()
        if auto_loaded_chunks:
            # 语料重建后若忘记重建索引，块 id 可能仍然对得上、正文却已变化，
            # 因此用索引里记录的切块文件指纹再做一次一致性校验。
            self.dense.check_freshness()

    # ---------------------------------------------------------------- 装配检查
    def _verify_index_coverage(self) -> None:
        """确认两个索引引用的块都能在切块文件里找到，否则立即报错。"""
        missing = [cid for cid in self.dense.chunk_ids if cid not in self.chunks]
        if missing:
            raise RuntimeError(
                "稠密索引引用了 %d 个切块文件中不存在的块（如 %s），请重建索引"
                % (len(missing), missing[0]))
        missing_lex = [cid for cid in self.lexical.chunk_ids if cid not in self.chunks]
        if missing_lex:
            raise RuntimeError(
                "词法索引引用了 %d 个切块文件中不存在的块（如 %s），请重建索引"
                % (len(missing_lex), missing_lex[0]))

    # ---------------------------------------------------------------- 检索
    def retrieve(self, query: str, k: int = 5, *,
                 hybrid: bool = False,
                 mode: str = "auto") -> RetrievalResponse:
        """检索入口。

        ``mode``: ``dense`` / ``lexical`` / ``hybrid`` / ``auto``（默认）。
        ``hybrid`` / ``mode`` 任选其一；``hybrid=True`` 等价于 ``mode="hybrid"``。
        """
        if hybrid:
            mode = "hybrid"
        query = (query or "").strip()
        if not query:
            return RetrievalResponse([], "none")

        if mode == "lexical":
            hits = self.lexical.search(query, k=k)
            return self._respond(self._build(hits, "lexical"), "lexical")

        if mode == "dense":
            self._require_api_key()
            vector = _call_embedding_api([query], self.api_key)[0]
            hits = self.dense.search(vector, k=k)
            return self._respond(self._build(hits, "dense"), "dense")

        if mode == "hybrid":
            return self._hybrid(query, k)

        # mode == "auto"
        try:
            self._require_api_key()
            vector = _call_embedding_api([query], self.api_key)[0]
            hits = self.dense.search(vector, k=k)
            return self._respond(self._build(hits, "dense"), "dense")
        except Exception as exc:  # noqa: BLE001 —— 任何稠密路径故障都要兜底
            logger.warning("稠密检索不可用（%s），自动降级到词法检索", exc)
            hits = self.lexical.search(query, k=k)
            return self._respond(self._build(hits, "lexical"),
                                 "dense-fallback-lexical",
                                 fallback_reason=str(exc))

    # ---------------------------------------------------------------- 混合
    def _hybrid(self, query: str, k: int) -> RetrievalResponse:
        """加权 RRF 融合。

        为什么不用等权：实测发现中文查询在词法路径上的 Top-1 经常是无关文档
        （字符 n-gram 容易撞上"本系统/数据集/识别"这类高频字），等权融合会把
        噪声顶到语义结果前面。这里给词法 0.5 的权重并设绝对下限，
        让稠密结果稳定占优，词法只负责补充召回。
        """
        self._require_api_key()
        vector = _call_embedding_api([query], self.api_key)[0]
        dense_hits = self.dense.search(vector, k=min(k * 2, len(self.dense)))
        lex_hits = self.lexical.search(query, k=min(k * 2, len(self.lexical)))

        fused: Dict[str, Dict[str, Any]] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            slot = fused.setdefault(hit.chunk_id, {"rrf": 0.0, "rows": {}})
            slot["rrf"] += DENSE_WEIGHT / (RRF_K + rank)
            slot["rows"]["dense"] = hit.score
        for rank, hit in enumerate(lex_hits, start=1):
            if hit.score < LEXICAL_MIN_SCORE:
                continue          # 字面噪声不参与融合
            slot = fused.setdefault(hit.chunk_id, {"rrf": 0.0, "rows": {}})
            slot["rrf"] += LEXICAL_WEIGHT / (RRF_K + rank)
            slot["rows"]["lexical"] = hit.score

        ordered = sorted(fused.items(), key=lambda kv: -kv[1]["rrf"])[:k]
        results: List[RetrievalResult] = []
        for chunk_id, info in ordered:
            rows = info["rows"]
            source = "hybrid" if len(rows) > 1 else next(iter(rows))
            result = self._make_result(self._lookup(chunk_id), info["rrf"], source)
            result.dense_score = rows.get("dense")
            result.lexical_score = rows.get("lexical")
            results.append(result)
        return self._respond(results, "hybrid")

    # ---------------------------------------------------------------- 装配
    def _require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("ZHIPUAI_API_KEY 未设置，无法使用稠密检索")

    def _lookup(self, chunk_id: str) -> Dict[str, Any]:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            raise RuntimeError(
                "索引中的块 %s 在切块文件中不存在，请重建索引" % chunk_id)
        return chunk

    def _make_result(self, chunk: Dict[str, Any], score: float,
                     source: str) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk.get("doc_id", chunk["chunk_id"].split("#")[0]),
            title=chunk.get("title", ""),
            section=chunk.get("section", ""),
            url=chunk.get("url", ""),
            license=chunk.get("license", ""),
            start=int(chunk.get("start", 0)),
            end=int(chunk.get("end", 0)),
            text=chunk.get("text", ""),
            score=float(score),
            source=source,
        )

    def _build(self, hits: Sequence[Any], source: str) -> List[RetrievalResult]:
        return [self._make_result(self._lookup(h.chunk_id), h.score, source)
                for h in hits]

    @staticmethod
    def _respond(results: List[RetrievalResult], mode: str,
                 fallback_reason: Optional[str] = None) -> RetrievalResponse:
        for idx, result in enumerate(results, start=1):
            result.rank = idx
        return RetrievalResponse(results, mode, fallback_reason)

    # ---------------------------------------------------------------- 溯源
    def read_document(self, doc_id: str,
                      raw_dir: Path = RAW_DIR) -> Optional[str]:
        """读取某块的整篇原文（供引用回填 / 上下文展开）。"""
        path = raw_dir / "{0}.txt".format(doc_id)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def highlight(self, result: RetrievalResult,
                  raw_dir: Path = RAW_DIR) -> Optional[str]:
        """拿出结果片段在原文中的精确区间（带上下文 window 字符）。"""
        document = self.read_document(result.doc_id, raw_dir)
        if document is None:
            return None
        window = 120
        start = max(0, result.start - window)
        end = min(len(document), result.end + window)
        return document[start:end]


# ------------------------------------------------------------------ CLI
def build_retriever(api_key: Optional[str] = None) -> Retriever:
    return Retriever(api_key=api_key)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="统一检索接口（冒烟测试）")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--k", type=int, default=5, help="返回条数")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "dense", "lexical", "hybrid"])
    parser.add_argument("--offline", action="store_true",
                        help="清空密钥，强制走词法兜底（演示降级路径）")
    args = parser.parse_args(argv)

    from config import load_dotenv
    env = load_dotenv()
    key = "" if args.offline else (env.get("ZHIPUAI_API_KEY") or "").strip()

    retriever = build_retriever(api_key=key)
    response = retriever.retrieve(args.query, k=args.k, mode=args.mode)

    print("查询   : {0}".format(args.query))
    print("模式   : {0}".format(response.mode))
    if response.fallback_reason:
        print("降级   : {0}".format(response.fallback_reason[:120]))
    print("-" * 78)
    for result in response:
        components = ""
        if result.dense_score is not None:
            components += " dense={0:.4f}".format(result.dense_score)
        if result.lexical_score is not None:
            components += " lexical={0:.4f}".format(result.lexical_score)
        print("[{0}] {1:.4f}  {2}  <{3}>{4}".format(
            result.rank, result.score, result.chunk_id, result.source, components))
        print("    {0}（{1}）".format(result.title[:60], result.section[:40]))
        print("    {0}".format(" ".join(result.text.split())[:110]))
    if not response.results:
        print("（无结果）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())