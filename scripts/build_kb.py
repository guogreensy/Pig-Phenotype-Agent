"""知识库一键构建（阶段 A 收口）。

把抓取 → 清洗切块 → 词法索引 → 稠密索引串成一条命令，并且默认**增量**：
已经就绪且不过期的阶段直接跳过，避免重复下载与重复消耗 embedding 额度。

用法
----
    python scripts/build_kb.py --status        # 只看各阶段是否就绪
    python scripts/build_kb.py                 # 增量构建（只做缺失/过期的）
    python scripts/build_kb.py --only chunks   # 只做指定阶段
    python scripts/build_kb.py --force         # 全量重建（会重新调用接口）

"过期"的判定用文件时间戳与指纹，而不是靠人记：
    chunks 晚于 sources、lexical 晚于 chunks、dense 的块指纹与 chunks 一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, load_dotenv  # noqa: E402
from src.rag.corpus import build_chunks, load_sources  # noqa: E402
from src.rag.dense import DenseIndex  # noqa: E402
from src.rag.lexical import LexicalIndex  # noqa: E402
from src.rag.corpus import load_chunks  # noqa: E402

STAGES: Tuple[str, ...] = ("fetch", "chunks", "lexical", "dense")


@dataclass
class Paths:
    """所有阶段用到的路径，集中一处以便测试注入。"""

    sources: Path
    chunks: Path
    cache_dir: Path
    raw_dir: Path
    lexical: Path
    dense: Path


def default_paths() -> Paths:
    kb = DATA_DIR / "kb"
    return Paths(
        sources=kb / "sources.json",
        chunks=kb / "chunks.jsonl",
        cache_dir=kb / "cache",
        raw_dir=kb / "raw",
        lexical=kb / "index" / "lexical.pkl",
        dense=kb / "index" / "dense.npz",
    )


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else -1.0


def _newer_or_equal(target: Path, reference: Path) -> bool:
    """target 存在且不早于 reference。"""
    return target.is_file() and _mtime(target) >= _mtime(reference)


def _dense_matches_chunks(paths: Paths) -> bool:
    """稠密索引记录的切块文件指纹是否与当前 chunks 一致。"""
    meta_path = paths.dense.with_suffix(".meta.json")
    if not (paths.dense.is_file() and paths.chunks.is_file() and meta_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    recorded = meta.get("chunks_file_sha256")
    if not recorded:
        return False
    actual = hashlib.sha256(paths.chunks.read_bytes()).hexdigest()
    return recorded == actual


def _lexical_fingerprint(path: Path) -> str:
    """读出词法索引里记录的切块指纹；读不到时返回空串。"""
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 —— 索引损坏一律按"读不到"处理
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("chunks_file_sha256") or "")


def stage_status(name: str, paths: Paths) -> Tuple[bool, str]:
    """返回 (是否就绪, 说明)。"""
    if name == "fetch":
        if not paths.sources.is_file():
            return False, "缺少 sources.json"
        cached = list(paths.cache_dir.glob("*")) if paths.cache_dir.is_dir() else []
        if not cached:
            return False, "cache 目录为空"
        return True, "%d 个缓存文件" % len(cached)

    if name == "chunks":
        if not paths.chunks.is_file():
            return False, "缺少 chunks.jsonl"
        if not _newer_or_equal(paths.chunks, paths.sources):
            return False, "chunks 早于 sources（语料清单已更新）"
        return True, "%.1f KB" % (paths.chunks.stat().st_size / 1024)

    if name == "lexical":
        if not paths.lexical.is_file():
            return False, "缺少 lexical.pkl"
        # 优先比内容指纹：仅仅重写 chunks.jsonl（内容没变）不应判为过期
        recorded = _lexical_fingerprint(paths.lexical)
        if recorded:
            if not paths.chunks.is_file():
                return False, "缺少 chunks.jsonl"
            actual = hashlib.sha256(paths.chunks.read_bytes()).hexdigest()
            if recorded != actual:
                return False, "词法索引与当前 chunks 指纹不一致"
            return True, "%.1f MB（指纹一致）" % (paths.lexical.stat().st_size / 1024 / 1024)
        if not _newer_or_equal(paths.lexical, paths.chunks):
            return False, "lexical 早于 chunks（切块已更新）"
        return True, "%.1f MB（按时间戳判断）" % (paths.lexical.stat().st_size / 1024 / 1024)

    if name == "dense":
        ids = paths.dense.with_suffix(".ids.json")
        if not (paths.dense.is_file() and ids.is_file()):
            return False, "缺少 dense.npz / dense.ids.json"
        if not _dense_matches_chunks(paths):
            return False, "稠密索引与当前 chunks 指纹不一致"
        return True, "%.1f MB" % (paths.dense.stat().st_size / 1024 / 1024)

    raise ValueError("未知阶段：%s" % name)


# ------------------------------------------------------------------ 各阶段执行
def _run_fetch(paths: Paths, force: bool, limit: int) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_corpus.py"),
               "--limit", str(limit)]
    if force:
        command.append("--force")
    print("[fetch] 执行：%s" % " ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def _run_chunks(paths: Paths, force: bool, limit: int) -> None:
    documents = load_sources(paths.sources)
    print("[chunks] 抽取 %d 篇文档 ..." % len(documents))
    chunks, stats = build_chunks(documents, raw_dir=paths.raw_dir,
                                 chunks_path=paths.chunks)
    print("[chunks] %d 块，正文 %d 字，片段 %d 字" % (
        stats["chunks"], stats["chars"], stats["chunk_chars"]))


def _run_lexical(paths: Paths, force: bool, limit: int) -> None:
    chunks = load_chunks(paths.chunks)
    print("[lexical] 索引 %d 块 ..." % len(chunks))
    index = LexicalIndex.build(chunks, chunks_file=paths.chunks)
    index.save(paths.lexical)
    print("[lexical] 维度 %d" % index.matrix.shape[1])


def _run_dense(paths: Paths, force: bool, limit: int, api_key: Optional[str] = None) -> None:
    key = api_key if api_key is not None else _api_key()
    if not key:
        raise SystemExit("ZHIPUAI_API_KEY 未设置，无法构建稠密索引")
    chunks = load_chunks(paths.chunks)
    print("[dense] 向量化 %d 块（会调用 embedding 接口）..." % len(chunks))
    index = DenseIndex.build(chunks, api_key=key, chunks_file=paths.chunks)
    index.save(paths.dense)
    print("[dense] %d × %d，tokens=%s" % (
        len(index), index.embeddings.shape[1],
        index.meta.get("usage", {}).get("total_tokens")))


_RUNNERS = {
    "fetch": _run_fetch,
    "chunks": _run_chunks,
    "lexical": _run_lexical,
    "dense": _run_dense,
}


def _api_key() -> str:
    return (load_dotenv().get("ZHIPUAI_API_KEY") or "").strip()


# ---------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None,
         paths: Optional[Paths] = None) -> int:
    parser = argparse.ArgumentParser(description="知识库一键构建（增量）")
    parser.add_argument("--status", action="store_true", help="只报告状态，不执行")
    parser.add_argument("--force", action="store_true", help="忽略就绪状态，全量重建")
    parser.add_argument("--only", default="", help="只执行这些阶段（逗号分隔）")
    parser.add_argument("--skip", default="", help="跳过这些阶段（逗号分隔）")
    parser.add_argument("--limit", type=int, default=16, help="PMC 论文目标篇数（传给 fetch）")
    args = parser.parse_args(argv)

    paths = paths or default_paths()
    only = [s for s in args.only.split(",") if s.strip()]
    skip = [s for s in args.skip.split(",") if s.strip()]
    for name in only + skip:
        if name not in STAGES:
            raise SystemExit("未知阶段 %r，可选：%s" % (name, "、".join(STAGES)))

    selected = [s for s in STAGES if (not only or s in only) and s not in skip]

    print("=" * 66)
    print("%-8s %-8s %s" % ("阶段", "状态", "说明"))
    print("-" * 66)
    statuses = {}
    for name in STAGES:
        ready, detail = stage_status(name, paths)
        statuses[name] = ready
        mark = "就绪" if ready else "待建"
        print("%-8s %-8s %s" % (name, mark, detail))
    print("=" * 66)

    if args.status:
        return 0

    executed: List[str] = []
    for name in selected:
        # 现查现用：上一个阶段重建后，后续阶段可能刚好变成过期状态
        ready, detail = stage_status(name, paths)
        if ready and not args.force:
            print("[{0}] 已就绪，跳过（--force 可强制重建）".format(name))
            continue
        if not ready:
            print("[{0}] 待建：{1}".format(name, detail))
        if name == "dense":
            _run_dense(paths, args.force, args.limit)
        else:
            _RUNNERS[name](paths, args.force, args.limit)
        executed.append(name)

    print("-" * 66)
    if executed:
        print("已执行：{0}".format("、".join(executed)))
    else:
        print("无需执行：所选阶段均已就绪")
    # 执行后重新核对，而不是沿用开头的快照
    ready_count = sum(1 for name in STAGES if stage_status(name, paths)[0])
    print("知识库状态：{0} / {1} 阶段就绪".format(ready_count, len(STAGES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())