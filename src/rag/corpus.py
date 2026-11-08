"""知识库语料清洗与分块（T2）。

输入（两条路径）
----------------
* ``data/kb/cache/*.xml``：PMC 的 JATS 全文，结构清晰（``<sec>`` / ``<p>`` / ``<ref-list>``）
* ``data/kb/cache/*.md`` 与本地 ``docs/*.md``：Markdown，按标题切章节

输出
----
* ``data/kb/raw/<doc_id>.txt``：清洗后的整篇文本（**唯一真源**，切块偏移以它为准）
* ``data/kb/chunks.jsonl``：每行一个切块，带 ``start`` / ``end`` 字符偏移

为什么保留偏移
--------------
溯源要能"指到"原文。有了偏移，回答里引用的片段可以精确回填到整篇文本的对应位置，
不需要把整篇文档塞进提示词，也不需要靠模糊匹配去猜。

分块策略
--------
1. 以段落为最小单位，按目标长度（默认 500 字）向后累加；
2. 相邻块保留 ``overlap`` 字重叠，避免关键句被切断在两个块里；
3. 超长单段（PDF/HTML 里常见的"一大坨"）先按句末标点二次切分，再参与累加；
4. 块文本与 ``text[start:end]`` **严格一致**，由单元测试保证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR  # noqa: E402

KB_DIR = DATA_DIR / "kb"
CACHE_DIR = KB_DIR / "cache"
RAW_DIR = KB_DIR / "raw"
SOURCES_PATH = KB_DIR / "sources.json"
CHUNKS_PATH = KB_DIR / "chunks.jsonl"

#: 分块参数（可由调用方覆盖，命令行可调）
DEFAULT_TARGET_CHARS = 500
DEFAULT_OVERLAP_CHARS = 80
DEFAULT_MIN_CHARS = 120
#: 单段超过该长度就按句子二次切分，避免出现一整块几千字
OVERSIZE_FACTOR = 1.5

#: 句子边界（中英文都覆盖），用于超长段的二次切分
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|(?<=[.])\s+")

#: JATS 里不需要进知识库的结构
_SKIP_TAGS = {"ref-list", "back", "table-wrap", "fig", "supplementary-material"}


@dataclass
class Document:
    """一篇语料文档。"""

    doc_id: str
    kind: str
    title: str
    url: str
    license: str
    path: Path
    sha256: str
    bytes: int


@dataclass
class Chunk:
    """一个可检索片段。"""

    chunk_id: str
    doc_id: str
    title: str
    url: str
    license: str
    section: str
    start: int
    end: int
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "license": self.license,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "chars": len(self.text),
            "sha256": self.sha256,
            "text": self.text,
        }


# ------------------------------------------------------------------ 载入清单
#: PMC/JATS 的 ``license-type`` 原始代码 -> 可读的许可名称。
#:
#: PMC 在 ``<license license-type="ccbylicense">`` 里给的是代码，
#: 直接展示会变成 "ccbylicense" 这种不可读的字符串。这里只映射到**许可族**，
#: 不擅自补版本号——单篇文章的具体版本需要回原文核对，报告里也不声称版本。
LICENSE_MAP: Dict[str, str] = {
    "ccbylicense": "CC BY",
    "ccbynclicense": "CC BY-NC",
    "ccbyncndlicense": "CC BY-NC-ND",
    "ccbyndlicense": "CC BY-ND",
    "ccbyncsalicense": "CC BY-NC-SA",
    "ccbysalicense": "CC BY-SA",
    "cc0": "CC0",
    "public-domain": "公有领域",
}


def normalize_license(raw: Any) -> str:
    """把许可代码归一化为可读名称；无法识别时原样保留（去空白）。"""
    text = str(raw or "").strip()
    if not text:
        return "unknown"
    return LICENSE_MAP.get(text.lower(), text)


def load_sources(path: Optional[Path] = None) -> List[Document]:
    """读取 ``sources.json``，返回文档列表。"""
    manifest = Path(path) if path else SOURCES_PATH
    if not manifest.is_file():
        raise FileNotFoundError(
            "找不到语料清单 {0}，请先运行 python scripts/fetch_corpus.py".format(manifest)
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    project_root = DATA_DIR.parent
    documents: List[Document] = []
    for item in payload.get("documents", []):
        raw_path = project_root / item["path"]
        documents.append(
            Document(
                doc_id=item["doc_id"],
                kind=item["kind"],
                title=item.get("title") or item["doc_id"],
                url=item.get("url", ""),
                license=normalize_license(item.get("license", "unknown")),
                path=raw_path,
                sha256=item.get("sha256", ""),
                bytes=int(item.get("bytes", 0)),
            )
        )
    return documents


# ------------------------------------------------------------------ 文本抽取
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _node_text(node: ElementTree.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def extract_jats(xml_text: str) -> str:
    """从 JATS XML 抽取正文。

    结构：``title`` → ``## Abstract`` → 各 ``<sec>`` 的标题与段落。
    ``ref-list`` / ``back`` / 图表 / 补充材料一律丢弃——它们是目录不是知识。
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return _extract_jats_fallback(xml_text)

    blocks: List[str] = []

    title_node = next(
        (n for n in root.iter() if _localname(n.tag) == "article-title"), None
    )
    if title_node is not None:
        title = _node_text(title_node)
        if title:
            blocks.append("# " + title)

    abstract_node = next(
        (n for n in root.iter() if _localname(n.tag) == "abstract"), None
    )
    if abstract_node is not None:
        paragraphs = [
            _node_text(p) for p in abstract_node.iter() if _localname(p.tag) == "p"
        ]
        paragraphs = [p for p in paragraphs if p]
        if paragraphs:
            blocks.append("## Abstract")
            blocks.extend(paragraphs)

    body = next((n for n in root.iter() if _localname(n.tag) == "body"), None)
    if body is not None:
        for section in body.iter():
            if _localname(section.tag) != "sec":
                continue
            heading_nodes = [
                child for child in section if _localname(child.tag) == "title"
            ]
            if heading_nodes:
                heading = _node_text(heading_nodes[0])
                if heading:
                    blocks.append("## " + heading)
            for child in section:
                name = _localname(child.tag)
                if name in _SKIP_TAGS:
                    continue
                if name == "p":
                    text = _node_text(child)
                    if text:
                        blocks.append(text)

    if not blocks:
        return _extract_jats_fallback(xml_text)
    return "\n".join(blocks)


def _extract_jats_fallback(xml_text: str) -> str:
    """XML 解析失败时的降级路径：按标签剥离，保留段落感。"""
    body_match = re.search(r"<body[^>]*>(.*?)</body>", xml_text, re.S)
    source = body_match.group(1) if body_match else xml_text
    source = re.sub(r"<(ref-list|back)[^>]*>.*?</\1>", " ", source, flags=re.S)
    source = re.sub(r"</p>", "\n", source)
    source = re.sub(r"</(sec|title|abstract)>", "\n", source)
    text = re.sub(r"<[^>]+>", " ", source)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_markdown(markdown_text: str) -> str:
    """Markdown → 纯文本，保留标题层级（标题即章节名）。"""

    def strip_inline(line: str) -> str:
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)      # 图片
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)   # 链接保留文字
        line = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", line)     # 行内代码
        line = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", line)
        return line

    lines: List[str] = []
    in_code = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = min(len(heading.group(1)), 2)
            lines.append("")
            lines.append("#" * level + " " + strip_inline(heading.group(2)).strip())
            continue
        if re.match(r"^[-*+]\s+", stripped):
            lines.append(strip_inline(re.sub(r"^[-*+]\s+", "", stripped)).strip())
            continue
        if re.match(r"^>\s?", stripped):
            lines.append(strip_inline(re.sub(r"^>\s?", "", stripped)).strip())
            continue
        lines.append(strip_inline(stripped).strip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text(document: Document) -> str:
    """按文档类型抽取纯文本。"""
    if not document.path.is_file():
        raise FileNotFoundError("语料文件不存在：{0}".format(document.path))
    raw_bytes = document.path.read_bytes()
    text = raw_bytes.decode("utf-8-sig", "ignore") if document.kind.endswith("md") \
        else raw_bytes.decode("utf-8", "ignore")
    if document.kind == "pmc_xml":
        cleaned = extract_jats(text)
    else:
        cleaned = extract_markdown(text)
    return normalize_text(cleaned)


def normalize_text(text: str) -> str:
    """统一空白：段内单空格、段间单换行。"""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# -------------------------------------------------------------------- 分块
def paragraph_spans(text: str) -> List[Tuple[int, int]]:
    """返回非空段落的 (start, end) 区间。"""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for line in text.split("\n"):
        start = cursor
        end = cursor + len(line)
        if line.strip():
            spans.append((start, end))
        cursor = end + 1
    return spans


def _hard_split_text(text: str, start: int, end: int,
                     limit: int) -> List[Tuple[int, int]]:
    """把一段文本切成不超过 limit 的区间，优先在空白处断开（避免切断单词）。"""
    pieces: List[Tuple[int, int]] = []
    cursor = start
    while end - cursor > limit:
        cut = cursor + limit
        window = text[cursor:cut]
        space = window.rfind(" ")
        if space > int(limit * 0.6):      # 空白足够靠后，就在空白处断
            cut = cursor + space + 1
        if cut <= cursor:                 # 极端情况下保证前进
            cut = cursor + limit
        pieces.append((cursor, cut))
        cursor = cut
    if cursor < end:
        pieces.append((cursor, end))
    return [(a, b) for a, b in pieces if b > a]


def _split_oversize(text: str, start: int, end: int,
                    limit: int) -> List[Tuple[int, int]]:
    """把超过 limit 的段落按句末标点切成若干子区间。

    两级切分：
    1. 优先在句末标点处断开（先按累积长度达到 limit 刷出）；
    2. **单个句子本身超过 limit 时**，退化为按空白/定长再切——否则会出现
       整段一句、几千字一块的情况（曾产生 4177 字的巨型块）。
    """
    segment = text[start:end]
    if len(segment) <= limit:
        return [(start, end)]

    pieces: List[Tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END.finditer(segment):
        cut = start + match.end()
        if cut - cursor >= limit and cut < end:
            pieces.extend(_hard_split_text(text, cursor, cut, limit))
            cursor = cut
    if cursor < end:
        pieces.extend(_hard_split_text(text, cursor, end, limit))
    return pieces


def split_text(text: str, target: int = DEFAULT_TARGET_CHARS,
               overlap: int = DEFAULT_OVERLAP_CHARS,
               min_chars: int = DEFAULT_MIN_CHARS) -> List[Tuple[int, int, str]]:
    """把整篇文本切成带偏移的块。

    Returns:
        ``[(start, end, chunk_text), ...]``，保证 ``text[start:end] == chunk_text``。
    """
    if not text.strip():
        return []

    spans: List[Tuple[int, int]] = []
    hard_max = max(int(target * OVERSIZE_FACTOR), min_chars * 2)
    for span_start, span_end in paragraph_spans(text):
        spans.extend(_split_oversize(text, span_start, span_end, hard_max))

    # 段落对齐累加：只要"再加入下一段"不超上限就继续并入，否则换块
    built: List[Tuple[int, int]] = []
    start: Optional[int] = None
    end: Optional[int] = None
    for span_start, span_end in spans:
        if start is None:
            start, end = span_start, span_end
            continue
        if span_end - start <= hard_max:
            end = span_end
        else:
            built.append((start, end))
            boundary = max(end - overlap, span_start)
            start = boundary if boundary < span_end else span_start
            end = span_end
    if start is not None:
        built.append((start, end))

    # 过短的块并入前一块；末块过短则并入前一块（不硬切，避免切断词语）
    merged: List[Tuple[int, int]] = []
    for span_start, span_end in built:
        if merged and (span_end - span_start) < min_chars:
            merged[-1] = (merged[-1][0], span_end)
        else:
            merged.append((span_start, span_end))
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_chars:
        merged[-2] = (merged[-2][0], merged[-1][1])
        merged.pop()

    return [(s, e, text[s:e]) for s, e in merged if text[s:e].strip()]


def section_of(text: str, offset: int) -> str:
    """取偏移位置之前最近的标题，作为块的章节名。"""
    prefix = text[:offset]
    matches = list(re.finditer(r"(?m)^#{1,2}\s+(.+)$", prefix))
    return matches[-1].group(1).strip() if matches else ""


# ---------------------------------------------------------------------- 构建
def build_chunks(documents: List[Document], raw_dir: Path = RAW_DIR,
                 chunks_path: Path = CHUNKS_PATH,
                 target: int = DEFAULT_TARGET_CHARS,
                 overlap: int = DEFAULT_OVERLAP_CHARS) -> Tuple[List[Chunk], Dict[str, Any]]:
    """对全部文档抽取、清洗、切块，落盘并返回统计。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Chunk] = []
    stats: Dict[str, Any] = {"documents": 0, "chunks": 0, "chars": 0,
                             "chunk_chars": 0, "empty": []}

    with chunks_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            text = extract_text(document)
            if len(text.strip()) < 200:
                stats["empty"].append(document.doc_id)
                continue
            text_path = raw_dir / "{0}.txt".format(document.doc_id)
            text_path.write_text(text, encoding="utf-8")

            stats["documents"] += 1
            stats["chars"] += len(text)
            for order, (start, end, piece) in enumerate(
                    split_text(text, target=target, overlap=overlap), start=1):
                chunk = Chunk(
                    chunk_id="{0}#{1:04d}".format(document.doc_id, order),
                    doc_id=document.doc_id,
                    title=document.title,
                    url=document.url,
                    license=document.license,
                    section=section_of(text, start),
                    start=start,
                    end=end,
                    text=piece,
                )
                all_chunks.append(chunk)
                stats["chunk_chars"] += len(piece)
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            stats["chunks"] = len(all_chunks)

    return all_chunks, stats


def load_chunks(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取已保存的块（供检索层使用）。"""
    chunk_path = Path(path) if path else CHUNKS_PATH
    if not chunk_path.is_file():
        raise FileNotFoundError(
            "找不到切块文件 {0}，请先运行 python -m src.rag.corpus".format(chunk_path)
        )
    items: List[Dict[str, Any]] = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="清洗语料并切块")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_CHARS, help="目标块长（字）")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CHARS, help="块间重叠（字）")
    args = parser.parse_args(argv)

    documents = load_sources()
    print("读取清单：{0} 篇文档".format(len(documents)))
    chunks, stats = build_chunks(documents, target=args.target, overlap=args.overlap)

    lengths = [len(c.text) for c in chunks] or [0]
    print("-" * 62)
    print("可用文档 : {0} 篇".format(stats["documents"]))
    print("正文字符 : {0:,}（抽取后的文档正文）".format(stats["chars"]))
    print("片段字符 : {0:,}（入库片段合计，含重叠）".format(stats["chunk_chars"]))
    print("切块数量 : {0}".format(stats["chunks"]))
    print("块长       : 最短 {0} / 平均 {1:.0f} / 最长 {2}".format(
        min(lengths), sum(lengths) / len(lengths), max(lengths)))
    print("输出       : {0} / {1}".format(
        CHUNKS_PATH.relative_to(DATA_DIR.parent),
        RAW_DIR.relative_to(DATA_DIR.parent)))
    if stats["empty"]:
        print("[警告] 以下文档正文过短，已跳过：{0}".format("、".join(stats["empty"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
