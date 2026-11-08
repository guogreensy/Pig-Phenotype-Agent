"""语料清洗与分块测试（T2）。

全部离线：使用内存中的示例文本与临时目录，不访问网络、不依赖已抓取的语料。
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from src.rag import corpus

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_rag"

JATS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Pig Posture Recognition with Cameras</article-title></title-group>
      <abstract><p>We study lying and standing postures of pigs.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title>
      <p>Precision livestock farming needs automatic posture monitoring.</p>
    </sec>
    <sec><title>Methods</title>
      <p>We annotate 300 crops and train a vision model.</p>
    </sec>
    <ref-list><ref><p>Smith et al. 2020.</p></ref></ref-list>
  </body>
</article>
"""

MARKDOWN_SAMPLE = """# 数据集说明

本研究使用公开数据集。详见 [链接](https://example.org)。

## 类别定义

- 站立
- 侧卧
- 坐姿

```python
print("这段代码不应进入知识库")
```

> 引用块也要保留内容。

### 标注规范

每头猪给出一个 bbox。
"""


class TestMarkdownExtraction(unittest.TestCase):
    def test_headings_and_lists_kept(self):
        text = corpus.extract_markdown(MARKDOWN_SAMPLE)
        self.assertIn("# 数据集说明", text)
        self.assertIn("## 类别定义", text)
        self.assertIn("站立", text)
        self.assertIn("每头猪给出一个 bbox", text)

    def test_links_flattened_and_code_dropped(self):
        text = corpus.extract_markdown(MARKDOWN_SAMPLE)
        self.assertIn("公开数据集", text)
        self.assertNotIn("https://example.org", text)
        self.assertNotIn("print(", text)

    def test_blockquote_kept(self):
        text = corpus.extract_markdown(MARKDOWN_SAMPLE)
        self.assertIn("引用块也要保留内容", text)


class TestJatsExtraction(unittest.TestCase):
    def test_sections_and_abstract(self):
        text = corpus.extract_jats(JATS_SAMPLE)
        self.assertIn("Pig Posture Recognition with Cameras", text)
        self.assertIn("## Abstract", text)
        self.assertIn("## Introduction", text)
        self.assertIn("Precision livestock farming", text)
        self.assertIn("## Methods", text)

    def test_reference_list_dropped(self):
        text = corpus.extract_jats(JATS_SAMPLE)
        self.assertNotIn("Smith et al. 2020", text)

    def test_fallback_when_xml_broken(self):
        broken = "<article><body><p>unclosed paragraph <b>bold</b>"
        text = corpus.extract_jats(broken)
        self.assertIn("unclosed paragraph", text)


class TestNormalize(unittest.TestCase):
    def test_collapses_whitespace_and_blank_lines(self):
        text = corpus.normalize_text("a  b\n\n\n c\t d\n")
        self.assertEqual(text, "a b\nc d")


class TestSplitText(unittest.TestCase):
    def _doc(self, paragraphs=40, size=120):
        return "\n".join(
            "段落{0}".format(i) + "内容" * size for i in range(paragraphs)
        )

    def test_offsets_are_exact(self):
        text = self._doc()
        for start, end, piece in corpus.split_text(text, target=500, overlap=80):
            self.assertEqual(text[start:end], piece)

    def test_covers_whole_document(self):
        text = self._doc()
        chunks = corpus.split_text(text, target=500, overlap=80)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], len(text))

    def test_no_chunk_much_longer_than_target(self):
        text = self._doc()
        limit = int(500 * corpus.OVERSIZE_FACTOR) + corpus.DEFAULT_MIN_CHARS + 10
        for _, _, piece in corpus.split_text(text, target=500, overlap=80):
            self.assertLessEqual(len(piece), limit, "块过长：{0}".format(len(piece)))

    def test_oversize_paragraph_is_split(self):
        # 单个超长段落（无换行）也必须被切开
        long_paragraph = "这是一句话。" * 400
        chunks = corpus.split_text(long_paragraph, target=500, overlap=80)
        self.assertGreater(len(chunks), 1)
        for _, _, piece in chunks:
            self.assertLessEqual(len(piece), int(500 * corpus.OVERSIZE_FACTOR) + 10)

    def test_short_tail_merged(self):
        text = "第一段" + "内容" * 300 + "\n短尾"
        chunks = corpus.split_text(text, target=500, overlap=80, min_chars=120)
        self.assertGreaterEqual(len(chunks[-1][2]), 120)

    def test_huge_paragraph_with_final_sentence_at_end(self):
        """回归：最后一句正好结束在段尾时，曾把整段（4177 字）一次性输出。"""
        sentence = "This is a fairly long sentence about pig posture detection. "
        text = (sentence * 60).strip()
        self.assertGreater(len(text), 3000)
        chunks = corpus.split_text(text, target=500, overlap=80)
        self.assertGreater(len(chunks), 1)
        limit = int(500 * corpus.OVERSIZE_FACTOR) + corpus.DEFAULT_MIN_CHARS + 10
        for start, end, piece in chunks:
            self.assertLessEqual(len(piece), limit, "块过长：{0}".format(len(piece)))
            self.assertEqual(text[start:end], piece)

    def test_at_most_one_tiny_fragment(self):
        text = self._doc() + "\n5.\n" + self._doc(paragraphs=5)
        chunks = corpus.split_text(text, target=500, overlap=80, min_chars=120)
        tiny = [c for c in chunks if len(c[2]) < 120]
        self.assertLessEqual(len(tiny), 1, "碎片过多：{0}".format([len(c[2]) for c in tiny]))

    def test_empty_text(self):
        self.assertEqual(corpus.split_text("   \n\n "), [])

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        text = self._doc(paragraphs=60)
        with_overlap = corpus.split_text(text, target=500, overlap=100)
        without = corpus.split_text(text, target=500, overlap=0)
        self.assertGreaterEqual(len(with_overlap), len(without))


class TestSectionOf(unittest.TestCase):
    def test_nearest_heading(self):
        text = "# 总标题\n正文一\n## 章节甲\n正文二\n## 章节乙\n正文三"
        offset = text.index("正文三")
        self.assertEqual(corpus.section_of(text, offset), "章节乙")


class TestBuildChunks(unittest.TestCase):
    def setUp(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        (TMP_ROOT / "cache").mkdir(parents=True)
        self.md_path = TMP_ROOT / "cache" / "doc.md"
        self.md_path.write_text(MARKDOWN_SAMPLE * 6, encoding="utf-8")

    def tearDown(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def _document(self):
        return corpus.Document(
            doc_id="local_demo", kind="local_md", title="数据集说明",
            url="docs/demo.md", license="本项目自有文档",
            path=self.md_path, sha256="x" * 64, bytes=self.md_path.stat().st_size,
        )

    def test_build_writes_text_and_chunks(self):
        chunks, stats = corpus.build_chunks(
            [self._document()],
            raw_dir=TMP_ROOT / "raw",
            chunks_path=TMP_ROOT / "chunks.jsonl",
        )
        self.assertGreaterEqual(stats["chunks"], 1)
        self.assertEqual(stats["documents"], 1)
        self.assertTrue((TMP_ROOT / "raw" / "local_demo.txt").is_file())

        lines = (TMP_ROOT / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(chunks))
        record = json.loads(lines[0])
        for key in ("chunk_id", "doc_id", "title", "url", "license",
                    "section", "start", "end", "chars", "sha256", "text"):
            self.assertIn(key, record)
        self.assertTrue(record["chunk_id"].startswith("local_demo#"))

    def test_chunk_offsets_match_saved_text(self):
        chunks, _ = corpus.build_chunks(
            [self._document()],
            raw_dir=TMP_ROOT / "raw",
            chunks_path=TMP_ROOT / "chunks.jsonl",
        )
        saved = (TMP_ROOT / "raw" / "local_demo.txt").read_text(encoding="utf-8")
        for chunk in chunks[:5]:
            self.assertEqual(saved[chunk.start:chunk.end], chunk.text)

    def test_short_document_skipped(self):
        minimal = corpus.Document(
            doc_id="tiny", kind="local_md", title="t", url="u", license="x",
            path=self.md_path, sha256="y" * 64, bytes=1,
        )
        text_path = TMP_ROOT / "cache" / "tiny.md"
        text_path.write_text("# 只有标题", encoding="utf-8")
        minimal = corpus.Document(
            doc_id="tiny", kind="local_md", title="t", url="u", license="x",
            path=text_path, sha256="y" * 64, bytes=10,
        )
        _, stats = corpus.build_chunks(
            [minimal], raw_dir=TMP_ROOT / "raw2",
            chunks_path=TMP_ROOT / "chunks2.jsonl",
        )
        self.assertEqual(stats["documents"], 0)
        self.assertEqual(stats["empty"], ["tiny"])

    def test_chunk_chars_matches_written_chunks(self):
        """正文与片段是两个不同的量，统计必须分别成立。"""
        chunks, stats = corpus.build_chunks(
            [self._document()],
            raw_dir=TMP_ROOT / "raw3",
            chunks_path=TMP_ROOT / "chunks3.jsonl",
        )
        self.assertEqual(stats["chunk_chars"], sum(len(c.text) for c in chunks))
        saved = (TMP_ROOT / "raw3" / "local_demo.txt").read_text(encoding="utf-8")
        self.assertEqual(stats["chars"], len(saved))


class TestLoadSources(unittest.TestCase):
    def test_missing_manifest_raises(self):
        with self.assertRaises(FileNotFoundError):
            corpus.load_sources(TMP_ROOT / "no_such_sources.json")


class TestLicenseNormalization(unittest.TestCase):
    """JATS 的 license-type 代码必须在入库前变成可读名称。"""

    def test_known_codes(self):
        self.assertEqual(corpus.normalize_license("ccbylicense"), "CC BY")
        self.assertEqual(corpus.normalize_license("ccbynclicense"), "CC BY-NC")
        self.assertEqual(corpus.normalize_license("ccbyncndlicense"), "CC BY-NC-ND")

    def test_case_insensitive(self):
        self.assertEqual(corpus.normalize_license("CCBYLICENSE"), "CC BY")

    def test_readable_values_pass_through(self):
        self.assertEqual(corpus.normalize_license("本项目自有文档"), "本项目自有文档")
        self.assertEqual(corpus.normalize_license("CC BY 4.0（数据集卡片声明）"),
                         "CC BY 4.0（数据集卡片声明）")

    def test_empty_becomes_unknown(self):
        self.assertEqual(corpus.normalize_license(None), "unknown")
        self.assertEqual(corpus.normalize_license("   "), "unknown")

    def test_load_sources_normalizes(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        manifest = TMP_ROOT / "sources.json"
        manifest.write_text(json.dumps({"documents": [{
            "doc_id": "d1", "kind": "pmc", "title": "T", "url": "",
            "license": "ccbynclicense", "path": "cache/d1.xml",
            "sha256": "0", "bytes": 1}]}, ensure_ascii=False), encoding="utf-8")
        try:
            documents = corpus.load_sources(manifest)
            self.assertEqual(documents[0].license, "CC BY-NC")
        finally:
            shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
