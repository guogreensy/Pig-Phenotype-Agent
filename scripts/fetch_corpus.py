"""知识库语料抓取（T1）。

数据来源与取舍
--------------
* **PMC（主源）**：用 E-utilities 检索开放获取论文，再用 ``efetch`` 取 JATS XML 全文。
  比抓 HTML 干净——标题、摘要、章节、许可都带结构化标签，且 XML 本身可追溯。
* **HuggingFace 数据集 README**：本项目所用公开数据集的官方说明（镜像站可达）。
* **本地文档**：``docs/*.md`` 与 ``data/README.md``，记录项目的真实口径（指标、限制、数据卡），
  也是语料里唯一的中文来源。

设计原则
--------
1. **可追溯**：每篇文档记录 URL、抓取时间、原始字节的 SHA256、许可信息；
   检索式也写进 ``sources.json``，保证"为什么是这几篇"可以被复核。
2. **幂等**：已缓存的文档默认跳过；``--force`` 才重新下载。
3. **不造假**：下载失败、返回体过小、抽取不到正文的文档，一律记为 ``error`` 条目并在
   终端列出，绝不静默丢弃或用占位内容顶替。
4. **克制请求**：默认 1 请求/秒（NCBI 建议值以内）。

用法::

    python scripts/fetch_corpus.py              # 增量抓取
    python scripts/fetch_corpus.py --force      # 强制重下
    python scripts/fetch_corpus.py --limit 5    # 只抓 5 篇 PMC（调试用）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR  # noqa: E402
# 许可名称的归一化与切块阶段共用同一份映射，避免两处各写一套
from src.rag.corpus import normalize_license  # noqa: E402

KB_DIR = DATA_DIR / "kb"
CACHE_DIR = KB_DIR / "cache"
RAW_DIR = KB_DIR / "raw"
SOURCES_PATH = KB_DIR / "sources.json"
CHUNKS_PATH = KB_DIR / "chunks.jsonl"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: PMC 检索式。顺序即优先级，结果按顺序去重后截断到 ``PMC_TARGET`` 篇。
PMC_QUERIES: List[str] = [
    '(pig OR swine) AND (posture OR lying) AND (deep learning OR computer vision)',
    '(pig OR swine) AND (behaviour OR behavior) AND (monitoring OR recognition)',
    '(pig OR swine) AND (body condition OR body weight) AND (image OR vision)',
    '(pig OR swine) AND welfare AND (lying OR resting OR activity)',
    '(precision livestock farming) AND (machine vision OR camera)',
]

#: PMC 论文目标篇数（去重后）。
PMC_TARGET = 14

#: 相关性门槛：标题+摘要中至少命中 ``TOPIC_MIN_HITS`` 个不同关键词，才算对题。
#: 目的是排除检索式召回但与"姿态/行为/视觉监测"无关的论文（如遗传学、营养学）。
TOPIC_KEYWORDS: List[str] = [
    "posture", "postures", "lying", "behavio", "activity", "vision", "camera",
    "video", "image", "images", "detection", "tracking", "monitor",
    "pose", "welfare", "body condition", "weight", "feeding", "drinking",
    "deep learning", "machine learning", "neural", "recognition", "sensor",
]
TOPIC_MIN_HITS = 3

#: 物种门槛：标题+摘要必须提到猪相关词，否则视为跑题（检索式会召回"狗"等其它物种）。
SPECIES_KEYWORDS: List[str] = [
    "pig", "pigs", "swine", "hog", "hogs", "sow", "sows", "piglet", "piglets",
    "farrow", "porcine", "boar", "gilt", "barrow", "weaner",
]

#: 本地文档：相对于项目根目录。
LOCAL_DOCS: List[str] = [
    "docs/data_card.md",
    "docs/experiment_report.md",
    "docs/technical_report.md",
    "docs/limitations.md",
    "docs/feasibility_report.md",
    "data/README.md",
]

HF_README_URL = (
    "https://hf-mirror.com/datasets/anilbhujel/viewpoint-aware-pig-posture-recognition"
    "/raw/main/README.md"
)

RATE_LIMIT_SECONDS = 1.0
MIN_BYTES = 3000
TIMEOUT = 60
USER_AGENT = "pig-phenotype-research/1.0 (course project; contact: local)"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_tags(text)).strip()


class Fetcher:
    """带限流与错误记录的最小 HTTP 抓取器。"""

    def __init__(self, rate_limit: float = RATE_LIMIT_SECONDS):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.rate_limit = rate_limit
        self._last = 0.0
        self.request_count = 0

    def get(self, url: str, params: Optional[Dict[str, Any]] = None,
            timeout: int = TIMEOUT, attempts: int = 3) -> requests.Response:
        """带重试的 GET。

        NCBI 偶尔会直接断开连接（``RemoteDisconnected``），这对检索式是致命的——
        第一次运行时正因为没有重试，第一条检索式被整体跳过。这里对连接错误、
        超时与 5xx 做指数退避重试，4xx 立即返回交由调用方判断。
        """
        last_error: Optional[Exception] = None
        response: Optional[requests.Response] = None
        for attempt in range(1, attempts + 1):
            wait = self.rate_limit - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                self._last = time.monotonic()
                self.request_count += 1
                if response.status_code < 500:
                    return response
                last_error = RuntimeError("HTTP {0}".format(response.status_code))
            except (requests.ConnectionError, requests.Timeout) as exc:
                self._last = time.monotonic()
                self.request_count += 1
                last_error = exc
            if attempt < attempts:
                time.sleep(2.0 * attempt)
        if response is not None:
            return response
        raise RuntimeError("连续 {0} 次请求失败：{1}".format(attempts, last_error))

    def close(self) -> None:
        self.session.close()


# --------------------------------------------------------------------- PMC
def pmc_search(fetcher: Fetcher, query: str, retmax: int) -> List[str]:
    """检索 PMC，返回数字 ID 列表（按相关性排序）。"""
    response = fetcher.get(
        "{0}/esearch.fcgi".format(EUTILS_BASE),
        params={"db": "pmc", "term": query, "retmax": retmax,
                "retmode": "json", "sort": "relevance"},
    )
    if response.status_code != 200:
        raise RuntimeError("PMC 检索失败 HTTP {0}".format(response.status_code))
    payload = response.json()
    return list(payload.get("esearchresult", {}).get("idlist") or [])


def pmc_fetch(fetcher: Fetcher, pmcid: str) -> requests.Response:
    return fetcher.get(
        "{0}/efetch.fcgi".format(EUTILS_BASE),
        params={"db": "pmc", "id": pmcid, "retmode": "xml"},
        timeout=90,
    )


def pmc_metadata(xml: str) -> Dict[str, Any]:
    """从 JATS XML 里取标题、许可、摘要长度、是否有正文。"""
    title_match = re.search(r"<article-title[^>]*>(.*?)</article-title>", xml, re.S)
    title = _clean(title_match.group(1)) if title_match else ""

    license_text = ""
    license_match = re.search(r"<license[^>]*>(.*?)</license>", xml, re.S)
    if license_match:
        raw = license_match.group(1)
        kind = re.search(r"content-type=\"([^\"]+)\"", license_match.group(0))
        license_text = kind.group(1) if kind else _clean(raw)[:120]
    if not license_text:
        copyright_match = re.search(r"<copyright-statement>(.*?)</copyright-statement>", xml, re.S)
        license_text = _clean(copyright_match.group(1))[:120] if copyright_match else "unknown"

    abstract_match = re.search(r"<abstract[^>]*>(.*?)</abstract>", xml, re.S)
    abstract_chars = len(_clean(abstract_match.group(1))) if abstract_match else 0

    return {
        "title": title,
        # 存可读名称，而不是 JATS 的 license-type 代码
        "license": normalize_license(license_text),
        "abstract_chars": abstract_chars,
        "has_body": "<body>" in xml,
    }


def _title_abstract(xml: str) -> str:
    """取出标题与摘要的纯文本，供主题/物种门槛共用。"""
    title = re.search(r"<article-title[^>]*>(.*?)</article-title>", xml, re.S)
    abstract = re.search(r"<abstract[^>]*>(.*?)</abstract>", xml, re.S)
    return " ".join(
        _clean(part.group(1)) for part in (title, abstract) if part is not None
    ).lower()


def topic_hits(xml: str) -> List[str]:
    """统计标题+摘要里命中的主题关键词（去重、保序）。

    检索式用的是布尔匹配，召回里必然混进遗传学、营养学等不对题的论文；
    这些文档进入知识库只会稀释检索质量，因此在入库前做一道门槛。
    """
    haystack = _title_abstract(xml)
    if not haystack:
        return []
    return [keyword for keyword in TOPIC_KEYWORDS if keyword in haystack]


def species_hits(xml: str) -> List[str]:
    """统计标题+摘要里命中的物种关键词。用于排除"狗/牛"等其它物种的研究。"""
    haystack = _title_abstract(xml)
    if not haystack:
        return []
    return [keyword for keyword in SPECIES_KEYWORDS if keyword in haystack]


def passes_gate(xml: str) -> tuple:
    """返回 (是否入库, 主题命中, 物种命中)。"""
    topics = topic_hits(xml)
    species = species_hits(xml)
    ok = len(topics) >= TOPIC_MIN_HITS and bool(species)
    return ok, topics, species


def collect_pmc(fetcher: Fetcher, limit: int, force: bool,
                errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """检索并下载 PMC 全文，返回文档条目列表。"""
    found: List[str] = []
    per_query = max(2, min(6, limit))
    for query in PMC_QUERIES:
        if len(found) >= limit:
            break
        try:
            ids = pmc_search(fetcher, query, per_query)
            print("  检索式 {0} -> {1} 条".format(query[:52], len(ids)))
        except Exception as exc:  # 网络问题不应中断整轮抓取
            errors.append({"stage": "pmc_search", "query": query,
                           "error": "{0}: {1}".format(type(exc).__name__, exc)})
            print("  [警告] 检索失败：{0}".format(exc))
            continue
        for pmcid in ids:
            if pmcid not in found:
                found.append(pmcid)
            if len(found) >= limit:
                break

    entries: List[Dict[str, Any]] = []
    for pmcid in found:
        doc_id = "pmc_{0}".format(pmcid)
        path = CACHE_DIR / "{0}.xml".format(doc_id)
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{0}/".format(pmcid)
        if path.is_file() and not force:
            data = path.read_bytes()
            text = data.decode("utf-8", "ignore")
            meta = pmc_metadata(text)
            ok, hits, species = passes_gate(text)
            if not ok:
                errors.append({"stage": "pmc_filter", "doc_id": doc_id,
                               "error": "不符合主题/物种门槛（主题 {0}、物种 {1}）：{2}".format(
                                   len(hits), len(species), meta["title"][:50])})
                print("  [跳过] {0} 不符合门槛 {1}".format(doc_id, meta["title"][:44]))
                continue
            entries.append({
                "doc_id": doc_id, "kind": "pmc_xml", "title": meta["title"] or doc_id,
                "url": url, "license": meta["license"], "fetched_at": _now(),
                "path": str(path.relative_to(DATA_DIR.parent)).replace("\\", "/"),
                "bytes": len(data), "sha256": sha256_bytes(data), "cached": True,
                "has_body": meta["has_body"], "abstract_chars": meta["abstract_chars"],
                "topic_hits": hits, "species_hits": species,
            })
            print("  [缓存] {0} {1}".format(doc_id, meta["title"][:60]))
            continue

        try:
            response = pmc_fetch(fetcher, pmcid)
        except Exception as exc:
            errors.append({"stage": "pmc_fetch", "doc_id": doc_id,
                           "error": "{0}: {1}".format(type(exc).__name__, exc)})
            print("  [失败] {0} 请求异常 {1}".format(doc_id, type(exc).__name__))
            continue

        if response.status_code != 200 or len(response.content) < MIN_BYTES:
            errors.append({"stage": "pmc_fetch", "doc_id": doc_id,
                           "error": "HTTP {0}, {1} bytes".format(
                               response.status_code, len(response.content))})
            print("  [失败] {0} HTTP {1} bytes={2}".format(
                doc_id, response.status_code, len(response.content)))
            continue

        meta = pmc_metadata(response.text)
        if not meta["has_body"]:
            errors.append({"stage": "pmc_body", "doc_id": doc_id,
                           "error": "无 <body>，可能是摘要级记录"})
            print("  [跳过] {0} 无正文".format(doc_id))
            continue

        ok, hits, species = passes_gate(response.text)
        if not ok:
            errors.append({"stage": "pmc_filter", "doc_id": doc_id,
                           "error": "不符合主题/物种门槛（主题 {0}、物种 {1}）：{2}".format(
                               len(hits), len(species), meta["title"][:50])})
            print("  [跳过] {0} 不符合门槛 {1}".format(doc_id, meta["title"][:44]))
            continue

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        entries.append({
            "doc_id": doc_id, "kind": "pmc_xml", "title": meta["title"] or doc_id,
            "url": url, "license": meta["license"], "fetched_at": _now(),
            "path": str(path.relative_to(DATA_DIR.parent)).replace("\\", "/"),
            "bytes": len(response.content), "sha256": sha256_bytes(response.content),
            "cached": False, "has_body": True,
            "abstract_chars": meta["abstract_chars"],
            "topic_hits": hits, "species_hits": species,
        })
        print("  [下载] {0} {1} ({2} KB)".format(
            doc_id, meta["title"][:60], len(response.content) // 1024))
    return entries


# ---------------------------------------------------------------- 其它来源
def collect_hf(fetcher: Fetcher, force: bool,
               errors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    doc_id = "hf_dataset_readme"
    path = CACHE_DIR / "{0}.md".format(doc_id)
    if path.is_file() and not force:
        data = path.read_bytes()
        cached = True
    else:
        try:
            response = fetcher.get(HF_README_URL)
        except Exception as exc:
            errors.append({"stage": "hf_fetch", "doc_id": doc_id,
                           "error": "{0}: {1}".format(type(exc).__name__, exc)})
            print("  [失败] HF README 请求异常 {0}".format(type(exc).__name__))
            return None
        if response.status_code != 200 or len(response.content) < 500:
            errors.append({"stage": "hf_fetch", "doc_id": doc_id,
                           "error": "HTTP {0}".format(response.status_code)})
            print("  [失败] HF README HTTP {0}".format(response.status_code))
            return None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        response.content and path.write_bytes(response.content)
        data = response.content
        cached = False

    return {
        "doc_id": doc_id, "kind": "hf_md", "title": "Viewpoint-Aware Pig Posture Recognition（数据集说明）",
        "url": HF_README_URL, "license": "CC BY 4.0（数据集卡片声明）", "fetched_at": _now(),
        "path": str(path.relative_to(DATA_DIR.parent)).replace("\\", "/"),
        "bytes": len(data), "sha256": sha256_bytes(data), "cached": cached,
    }


def collect_local(project_root: Path) -> List[Dict[str, Any]]:
    """本地文档不复制，只登记路径与哈希；正文由 corpus 模块直接读取。"""
    entries: List[Dict[str, Any]] = []
    for relative in LOCAL_DOCS:
        path = project_root / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        entries.append({
            "doc_id": "local_{0}".format(path.stem),
            "kind": "local_md",
            "title": path.name,
            "url": relative,
            "license": "本项目自有文档",
            "fetched_at": _now(),
            "path": relative,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "cached": True,
        })
    return entries


# ---------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="抓取知识库语料")
    parser.add_argument("--limit", type=int, default=PMC_TARGET, help="PMC 论文目标篇数")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新下载")
    parser.add_argument("--no-pmc", action="store_true", help="跳过 PMC（只用本地与 HF）")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("语料抓取")
    print("-" * 62)

    fetcher = Fetcher()
    errors: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []

    print("[1/3] 本地文档")
    local_entries = collect_local(project_root)
    documents.extend(local_entries)
    print("  登记 {0} 篇".format(len(local_entries)))

    print("[2/3] HuggingFace 数据集说明")
    hf_entry = collect_hf(fetcher, args.force, errors)
    if hf_entry:
        documents.append(hf_entry)
        print("  {0}".format("缓存命中" if hf_entry["cached"] else "下载完成"))

    print("[3/3] PMC 开放获取论文")
    if args.no_pmc:
        print("  已跳过（--no-pmc）")
    else:
        documents.extend(collect_pmc(fetcher, args.limit, args.force, errors))
    fetcher.close()

    reports = [d for d in documents if d["kind"] == "pmc_xml"]
    payload = {
        "generated_at": _now(),
        "generator": "scripts/fetch_corpus.py",
        "queries": PMC_QUERIES,
        "rate_limit_seconds": RATE_LIMIT_SECONDS,
        "request_count": fetcher.request_count,
        "documents": documents,
        "errors": errors,
        "summary": {
            "documents": len(documents),
            "pmc_papers": len(reports),
            "local_docs": len(local_entries),
            "total_bytes": sum(d["bytes"] for d in documents),
            "errors": len(errors),
        },
    }
    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 62)
    print("文档合计 : {0} 篇（PMC {1} / 本地 {2}）".format(
        len(documents), len(reports), len(local_entries)))
    print("原始体积 : {0:.1f} KB".format(payload["summary"]["total_bytes"] / 1024))
    print("请求次数 : {0}".format(fetcher.request_count))
    print("清单     : {0}".format(SOURCES_PATH.relative_to(project_root)))
    if errors:
        print("失败记录 : {0} 条（见清单 errors 字段）".format(len(errors)))
        for item in errors[:5]:
            print("   - {0} {1}".format(item.get("doc_id") or item.get("query"),
                                        item.get("error")))
    if len(reports) < 5:
        print("[提示] PMC 论文少于 5 篇，RAG 语料可能不足，请检查网络后重试。")
        return 1
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
