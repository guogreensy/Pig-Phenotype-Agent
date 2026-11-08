"""``src.schema`` 的单元测试。"""

from __future__ import annotations

import unittest

from src.schema import (
    DATASET_CLASS_MAP,
    POSTURES,
    map_dataset_class,
    normalize_posture,
    validate_result,
)


class TestNormalizePosture(unittest.TestCase):
    def test_canonical(self):
        for value in POSTURES:
            self.assertEqual(normalize_posture(value), value)

    def test_case_and_spacing(self):
        self.assertEqual(normalize_posture("Standing"), "standing")
        self.assertEqual(normalize_posture("  lying lateral "), "lying_lateral")
        self.assertEqual(normalize_posture("LYING-OTHER"), "lying_other")

    def test_punctuation(self):
        self.assertEqual(normalize_posture("standing。"), "standing")
        self.assertEqual(normalize_posture('"sitting"'), "sitting")

    def test_chinese(self):
        self.assertEqual(normalize_posture("站立"), "standing")
        self.assertEqual(normalize_posture("侧卧"), "lying_lateral")
        self.assertEqual(normalize_posture("俯卧"), "lying_other")
        self.assertEqual(normalize_posture("坐姿"), "sitting")
        self.assertEqual(normalize_posture("无法判断"), "unknown")

    def test_mixed_form(self):
        self.assertEqual(normalize_posture("standing（站立）"), "standing")

    def test_illegal(self):
        for value in ("jumping", "", None, 123, "飞", []):
            self.assertIsNone(normalize_posture(value))


class TestDatasetMap(unittest.TestCase):
    def test_all_five_classes_mapped(self):
        self.assertEqual(len(DATASET_CLASS_MAP), 5)
        self.assertEqual(map_dataset_class(0), "lying_lateral")
        self.assertEqual(map_dataset_class(1), "lying_lateral")
        self.assertEqual(map_dataset_class(2), "sitting")
        self.assertEqual(map_dataset_class(3), "standing")
        self.assertEqual(map_dataset_class(4), "lying_other")

    def test_out_of_range(self):
        self.assertIsNone(map_dataset_class(5))
        self.assertIsNone(map_dataset_class(-1))
        self.assertIsNone(map_dataset_class("abc"))
        self.assertIsNone(map_dataset_class(None))

    def test_string_class_id(self):
        self.assertEqual(map_dataset_class("3"), "standing")


class TestValidateResult(unittest.TestCase):
    def _good(self):
        return {
            "posture": "standing",
            "description": "猪只四肢着地站立，身体大部分可见。",
            "uncertain": False,
            "uncertainty_reason": None,
        }

    def test_valid(self):
        result = validate_result(self._good())
        self.assertTrue(result.valid)
        self.assertEqual(result.posture, "standing")
        self.assertFalse(result.uncertain)
        self.assertFalse(result.is_refusal)

    def test_missing_field(self):
        data = self._good()
        del data["uncertain"]
        result = validate_result(data)
        self.assertFalse(result.valid)
        self.assertTrue(any("uncertain" in e for e in result.errors))

    def test_illegal_posture(self):
        data = self._good()
        data["posture"] = "jumping"
        result = validate_result(data)
        self.assertFalse(result.valid)
        self.assertTrue(any("posture" in e for e in result.errors))

    def test_unknown_is_legal_and_refusal(self):
        data = {"posture": "unknown", "description": "图片模糊无法判断。",
                "uncertain": True, "uncertainty_reason": "画面模糊"}
        result = validate_result(data)
        self.assertTrue(result.valid)
        self.assertTrue(result.is_refusal)

    def test_unknown_with_uncertain_false_warns(self):
        data = {"posture": "unknown", "description": "无法判断。",
                "uncertain": False, "uncertainty_reason": None}
        result = validate_result(data)
        self.assertTrue(result.valid)
        self.assertTrue(result.warnings)

    def test_wrong_types(self):
        cases = [
            {"posture": "standing", "description": 123, "uncertain": False},
            {"posture": "standing", "description": "站立", "uncertain": "maybe"},
            {"posture": "standing", "description": "站立", "uncertain": False,
             "uncertainty_reason": 5},
        ]
        for data in cases:
            result = validate_result(data)
            self.assertFalse(result.valid, msg=str(data))

    def test_empty_description(self):
        data = self._good()
        data["description"] = "   "
        result = validate_result(data)
        self.assertFalse(result.valid)

    def test_bool_string_coerced(self):
        data = self._good()
        data["uncertain"] = "false"
        result = validate_result(data)
        self.assertTrue(result.valid)
        self.assertFalse(result.uncertain)

    def test_non_dict(self):
        for value in ([], "x", None, 42):
            result = validate_result(value)
            self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
