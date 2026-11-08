"""题集产物自洽性检查。

不重新构建题集，只校验已生成的文件是否内部一致：
题号唯一、每类题该有的字段齐全、期望文献真实存在、图片路径有效。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "data" / "experiment" / "agent_questions.jsonl"
SOURCES_FILE = PROJECT_ROOT / "data" / "kb" / "sources.json"
CHUNKS_FILE = PROJECT_ROOT / "data" / "kb" / "chunks.jsonl"


def load_questions():
    if not QUESTIONS_FILE.exists():
        return []
    return [json.loads(line) for line in
            QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestQuestionArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = load_questions()

    def test_file_exists_and_has_30_questions(self):
        self.assertTrue(QUESTIONS_FILE.exists(), "题集文件不存在")
        self.assertEqual(len(self.questions), 30)

    def test_ids_unique_and_prefixed_by_category(self):
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(set(ids)), len(ids))
        prefix = {"data": "D", "knowledge": "K", "vision": "V", "unanswerable": "U"}
        for question in self.questions:
            self.assertTrue(question["id"].startswith(prefix[question["category"]]),
                            question["id"])

    def test_category_counts(self):
        counts = {}
        for question in self.questions:
            counts[question["category"]] = counts.get(question["category"], 0) + 1
        self.assertEqual(counts, {"data": 10, "knowledge": 10,
                                  "vision": 5, "unanswerable": 5})

    def test_data_questions_have_numbers_or_entity_tokens(self):
        """数据题要么有可比对的数值（带容差），要么有可核对的实体词。"""
        for question in self.questions:
            if question["category"] != "data":
                continue
            expected = question["expected"]
            numbers = expected.get("numbers") or []
            tokens = expected.get("tokens") or []
            self.assertTrue(numbers or tokens, question["id"])
            for item in numbers:
                self.assertIsInstance(item["value"], (int, float), question["id"])
                self.assertGreater(item["tol"], 0, question["id"])
                self.assertIn("label", item)

    def test_anomaly_question_has_no_trivial_token(self):
        """指认题的实体词不能出现在题面里，否则照抄问题就算命中。"""
        for question in self.questions:
            if question["category"] != "data":
                continue
            for token in question["expected"].get("tokens") or []:
                if token in question["question"]:
                    continue
                # 题面不含该实体 → 这是一道真正的指认题
                self.assertIn(question["id"], {"D2", "D3", "D7", "D9"},
                              question["id"])

    def test_excluded_question_is_documented(self):
        excluded = [q for q in self.questions if q.get("exclude_from_aggregate")]
        self.assertEqual([q["id"] for q in excluded], ["D6"])
        self.assertTrue(excluded[0]["exclude_reason"])

    def test_knowledge_expected_docs_exist(self):
        if not SOURCES_FILE.exists():
            self.skipTest("知识库来源文件不存在")
        payload = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
        documents = payload["documents"] if isinstance(payload, dict) else payload
        known = {doc["doc_id"] for doc in documents}
        for question in self.questions:
            if question["category"] != "knowledge":
                continue
            docs = question["expected"].get("doc_ids") or []
            self.assertTrue(docs, question["id"])
            missing = [d for d in docs if d not in known]
            self.assertEqual(missing, [], "{0} 引用了不存在的文献".format(question["id"]))

    def test_knowledge_keywords_actually_appear_in_corpus(self):
        if not CHUNKS_FILE.exists():
            self.skipTest("片段文件不存在")
        texts = {}
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            texts.setdefault(chunk["doc_id"], []).append(chunk["text"])
        for question in self.questions:
            if question["category"] != "knowledge":
                continue
            hits = question["expected"].get("keyword_hits") or {}
            self.assertTrue(hits, question["id"])
            for doc_id, keywords in hits.items():
                body = "\n".join(texts.get(doc_id, []))
                for keyword in keywords:
                    self.assertIn(keyword, body,
                                  "{0}/{1} 未命中 {2}".format(question["id"], doc_id, keyword))

    def test_vision_image_paths_exist(self):
        for question in self.questions:
            if question["category"] != "vision":
                continue
            self.assertTrue(question.get("image_path"), question["id"])
            path = PROJECT_ROOT / question["image_path"]
            self.assertTrue(path.is_file(), question["id"])

    def test_vision_path_does_not_leak_label(self):
        """题目里不能出现类别名，否则模型读路径就能答对。"""
        for question in self.questions:
            if question["category"] != "vision":
                continue
            label = question["expected"]["label"]
            self.assertNotIn(label, question["question"], question["id"])
            self.assertNotIn(label, question["image_path"], question["id"])
            for other in ("lying", "standing", "sitting"):
                self.assertNotIn(other, question["question"], question["id"])

    def test_vision_source_image_matches_label(self):
        """中性文件必须来自对应类别目录，标注才有依据。"""
        for question in self.questions:
            if question["category"] != "vision":
                continue
            self.assertTrue(question.get("source_image"), question["id"])
            source = PROJECT_ROOT / question["source_image"]
            self.assertTrue(source.is_file(), question["id"])
            self.assertEqual(source.parent.name, question["expected"]["label"],
                             question["id"])

    def test_unanswerable_marked_not_answerable(self):
        for question in self.questions:
            if question["category"] != "unanswerable":
                continue
            self.assertFalse(question["answerable"], question["id"])
            self.assertTrue(question["expected"].get("must_admit"), question["id"])

    def test_every_question_has_text_and_evidence_hint(self):
        for question in self.questions:
            self.assertTrue(question["question"].strip(), question["id"])
            self.assertTrue(question.get("evidence_hint"), question["id"])


if __name__ == "__main__":
    unittest.main()