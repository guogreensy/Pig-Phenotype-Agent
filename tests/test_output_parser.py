"""``output_parser`` 的单元测试：覆盖需求 10.4 要求的解析场景。

全部为**模拟测试**，不调用任何真实 API。
"""

from __future__ import annotations

import unittest

from src.output_parser import (
    extract_answer,
    extract_freeform,
    find_json_object,
    light_repair,
    parse_model_output,
    regex_extract,
    strip_code_fence,
    strip_reasoning,
)


class TestStripAndScan(unittest.TestCase):
    def test_strip_code_fence(self):
        text = '好的，结果如下：\n```json\n{"posture": "standing"}\n```\n以上。'
        self.assertEqual(strip_code_fence(text), '{"posture": "standing"}')

    def test_strip_code_fence_without_language(self):
        text = "```\n{\"a\": 1}\n```"
        self.assertEqual(strip_code_fence(text), '{"a": 1}')

    def test_strip_code_fence_absent(self):
        self.assertIsNone(strip_code_fence('{"a": 1}'))

    def test_find_json_object_handles_braces_in_string(self):
        text = 'prefix {"description": "含 } 和 { 的说明", "posture": "sitting"} suffix'
        self.assertEqual(
            find_json_object(text),
            '{"description": "含 } 和 { 的说明", "posture": "sitting"}',
        )

    def test_find_json_object_handles_escaped_quote(self):
        text = '{"description": "他说\\"站立\\"", "posture": "standing"}'
        self.assertEqual(find_json_object(text), text)

    def test_find_json_object_none(self):
        self.assertIsNone(find_json_object("这里没有任何大括号"))


class TestLightRepair(unittest.TestCase):
    def test_trailing_comma(self):
        self.assertEqual(light_repair('{"a": 1,}'), '{"a": 1}')

    def test_python_literals(self):
        self.assertEqual(light_repair('{"u": True, "r": None}'), '{"u": true, "r": null}')

    def test_smart_quotes(self):
        self.assertEqual(light_repair("{\u201ca\u201d: 1}"), '{"a": 1}')


class TestParseModelOutput(unittest.TestCase):
    def test_direct_json(self):
        result = parse_model_output('{"posture": "standing", "description": "猪站立", "uncertain": false}')
        self.assertTrue(result.json_valid)
        self.assertEqual(result.method, "direct")
        self.assertEqual(result.data["posture"], "standing")

    def test_fenced_json(self):
        result = parse_model_output('```json\n{"posture": "sitting"}\n```')
        self.assertTrue(result.json_valid)
        self.assertEqual(result.method, "fenced")

    def test_prose_wrapped_json(self):
        result = parse_model_output(
            '根据图片判断，这只猪呈侧卧姿态。\n{"posture": "lying_lateral", '
            '"description": "身体侧向躺卧", "uncertain": false}\n希望有帮助。'
        )
        self.assertTrue(result.json_valid)
        self.assertEqual(result.method, "extracted")
        self.assertEqual(result.data["posture"], "lying_lateral")

    def test_trailing_comma_repaired(self):
        result = parse_model_output('{"posture": "standing", "uncertain": false,}')
        self.assertTrue(result.json_valid)
        self.assertEqual(result.method, "repaired")

    def test_single_quotes_repaired(self):
        result = parse_model_output("{'posture': 'standing', 'description': '站立', 'uncertain': False}")
        self.assertTrue(result.json_valid)
        self.assertEqual(result.data["posture"], "standing")

    def test_missing_fields_still_valid_json(self):
        result = parse_model_output('{"posture": "standing"}')
        self.assertTrue(result.json_valid)
        self.assertNotIn("description", result.data)

    def test_pure_prose_falls_back_to_regex(self):
        result = parse_model_output("posture: standing, 这只猪处于站立姿态")
        self.assertFalse(result.json_valid)
        self.assertEqual(result.method, "regex")
        self.assertEqual(result.data.get("posture"), "standing")

    def test_unparseable(self):
        result = parse_model_output("我没有看清这张图片。")
        self.assertFalse(result.json_valid)
        self.assertEqual(result.method, "failed")
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)

    def test_empty_text(self):
        for value in ("", "   ", None):
            result = parse_model_output(value)
            self.assertFalse(result.json_valid)
            self.assertEqual(result.method, "failed")

    def test_json_array_wrapped_object_is_extracted(self):
        # 模型偶尔把结果套在数组里；这里记录既定行为：从数组中抽出第一个对象
        result = parse_model_output('[{"posture": "standing"}]')
        self.assertTrue(result.json_valid)
        self.assertEqual(result.method, "extracted")
        self.assertEqual(result.data["posture"], "standing")

    def test_truncated_json_not_accepted_as_valid(self):
        result = parse_model_output('{"posture": "standing", "description": "猪站')
        self.assertFalse(result.json_valid)
        # 允许从残文中抽出已出现的字段，但绝不能凭空补全未出现的字段
        if result.data is not None:
            self.assertNotIn("description", result.data)
            self.assertNotIn("uncertain", result.data)


class TestReasoningStripping(unittest.TestCase):
    """思考型模型会输出 <think> 推理 + <answer> 结论，必须先剥离再解析。"""

    def test_strip_reasoning(self):
        text = "<think>我觉得可能是 standing 也可能是 sitting</think><answer>sitting</answer>"
        cleaned, had = strip_reasoning(text)
        self.assertTrue(had)
        self.assertNotIn("standing", cleaned)

    def test_extract_answer(self):
        text = "<think>推理</think><answer>确定结果</answer>"
        self.assertEqual(extract_answer(text), "确定结果")

    def test_reasoning_does_not_pollute_result(self):
        text = (
            "<think>先看 standing 的定义……不对，应该是侧卧。"
            "lying_lateral 符合定义。</think>"
            '<answer>{"posture": "lying_lateral", "description": "身体侧面贴地", '
            '"uncertain": false, "uncertainty_reason": null}</answer>'
        )
        result = parse_model_output(text)
        self.assertTrue(result.json_valid)
        self.assertEqual(result.data["posture"], "lying_lateral")

    def test_answer_in_fence(self):
        text = "<think>推理</think><answer>```json\n{\"posture\": \"standing\"}\n```</answer>"
        result = parse_model_output(text)
        self.assertTrue(result.json_valid)
        self.assertEqual(result.data["posture"], "standing")


class TestFreeformExtraction(unittest.TestCase):
    """A/B 组是自然语言回答，只抄写类别词，不做语义猜测。"""

    def test_extract_label_and_description(self):
        found = extract_freeform("standing | 这只猪呈站立姿态，四肢支撑于地面。")
        self.assertEqual(found["posture"], "standing")
        self.assertIn("站立姿态", found["description"])

    def test_extract_with_annotation(self):
        found = extract_freeform("sitting（坐姿）：后躯着地，前肢支撑。")
        self.assertEqual(found["posture"], "sitting")
        self.assertIn("后躯着地", found["description"])

    def test_space_separated_label(self):
        found = extract_freeform("lying lateral。猪呈侧卧姿态，身体侧躺。")
        self.assertEqual(found["posture"], "lying_lateral")

    def test_hyphen_separated_label(self):
        found = extract_freeform("结论：lying-other，胸腹着地。")
        self.assertEqual(found["posture"], "lying_other")

    def test_no_label_returns_none(self):
        self.assertIsNone(extract_freeform("图片里有一只猪，看不清姿态。"))

    def test_parse_freeform_method(self):
        result = parse_model_output("lying_lateral：身体侧向躺卧，躯干贴地。")
        self.assertFalse(result.json_valid)
        self.assertEqual(result.method, "freeform")
        self.assertEqual(result.data["posture"], "lying_lateral")

    def test_think_block_alone_does_not_produce_result(self):
        result = parse_model_output("<think>可能是 standing 吧</think>")
        self.assertFalse(result.json_valid)
        self.assertEqual(result.method, "failed")

    def test_freeform_ignores_reasoning_label(self):
        text = "<think>可能是 sitting</think><answer>standing，四肢支撑站立。</answer>"
        result = parse_model_output(text)
        self.assertEqual(result.data["posture"], "standing")


class TestRegexExtract(unittest.TestCase):
    def test_extract_fields(self):
        text = '"posture": "lying_other", "description": "胸腹着地趴卧", "uncertain": false, "uncertainty_reason": null'
        found = regex_extract(text)
        self.assertEqual(found["posture"], "lying_other")
        self.assertEqual(found["description"], "胸腹着地趴卧")
        self.assertFalse(found["uncertain"])
        self.assertIsNone(found["uncertainty_reason"])

    def test_rejects_non_label_posture(self):
        # 推理残句被宽松模式捕获时，不应被当成类别
        found = regex_extract("姿态类别：这只猪的四肢是否支撑")
        self.assertNotIn("posture", found)

    def test_chinese_label_pattern(self):
        found = regex_extract("姿态类别：standing")
        self.assertEqual(found["posture"], "standing")

    def test_fullwidth_colon(self):
        found = regex_extract("结论：posture：sitting")
        self.assertEqual(found["posture"], "sitting")


if __name__ == "__main__":
    unittest.main()
