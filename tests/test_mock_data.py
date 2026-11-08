"""模拟数据测试（T7）：确定性、规模、预设场景是否真的存在。"""

from __future__ import annotations

import unittest
from datetime import date

from src.tools import mock_data


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_data(self):
        first = mock_data.build_farm()
        second = mock_data.build_farm()
        self.assertEqual(len(first.records), len(second.records))
        self.assertEqual(first.records, second.records)

    def test_different_seed_differs(self):
        other = mock_data.build_farm(seed=1)
        self.assertNotEqual(mock_data.build_farm().records[:5], other.records[:5])

    def test_get_farm_is_cached(self):
        self.assertIs(mock_data.get_farm(), mock_data.get_farm())


class TestShape(unittest.TestCase):
    def setUp(self):
        self.farm = mock_data.build_farm()

    def test_counts(self):
        self.assertEqual(len(self.farm.pens), mock_data.PEN_COUNT)
        self.assertEqual(len(self.farm.pigs),
                         mock_data.PEN_COUNT * mock_data.PIGS_PER_PEN)
        self.assertEqual(len(self.farm.records),
                         mock_data.PEN_COUNT * mock_data.PIGS_PER_PEN * mock_data.DAYS)

    def test_date_range(self):
        self.assertEqual(self.farm.start_date, date(2026, 10, 8))
        self.assertEqual(self.farm.end_date, date(2026, 11, 6))
        days = {record.date for record in self.farm.records}
        self.assertEqual(len(days), mock_data.DAYS)
        self.assertEqual(min(days), "2026-10-08")
        self.assertEqual(max(days), "2026-11-06")

    def test_every_pig_has_full_trajectory(self):
        counts = {}
        for record in self.farm.records:
            counts[record.pig_id] = counts.get(record.pig_id, 0) + 1
        self.assertEqual(len(counts), len(self.farm.pigs))
        self.assertTrue(all(value == mock_data.DAYS for value in counts.values()))

    def test_values_in_sane_ranges(self):
        for record in self.farm.records:
            self.assertGreater(record.weight_kg, 40.0)
            self.assertGreater(record.feed_kg, 0.5)
            self.assertGreaterEqual(record.activity_index, 0.0)
            self.assertLessEqual(record.activity_index, 1.0)
            self.assertGreaterEqual(record.lying_ratio, 0.0)
            self.assertLessEqual(record.lying_ratio, 1.0)

    def test_meta_marks_simulated(self):
        meta = self.farm.meta()
        self.assertEqual(meta["source"], "simulated")
        self.assertIn("不代表任何真实养殖场", meta["disclaimer"])
        self.assertEqual(meta["seed"], mock_data.SEED)
        self.assertEqual(len(meta["scenarios"]), 3)


class TestScenarios(unittest.TestCase):
    """预设异常必须真实存在于数据里，否则异常检测工具就是空跑。"""

    def setUp(self):
        self.farm = mock_data.build_farm()
        self.by_key = {(r.date, r.pig_id): r for r in self.farm.records}

    def _pen_day_records(self, pen_id, day):
        rows = [r for r in self.farm.records
                if r.pen_id == pen_id and r.date == day]
        return rows

    def test_heat_stress_raises_pen3_temperature(self):
        before = self._pen_day_records("P3", "2026-10-24")   # 第 17 天，窗口前
        during = self._pen_day_records("P3", "2026-10-25")   # 第 18 天，窗口首日
        self.assertTrue(before and during)
        self.assertGreater(during[0].temp_c - before[0].temp_c, 3.0)

    def test_heat_stress_reduces_feed(self):
        window = [r for r in self.farm.records
                  if r.pen_id == "P3" and "2026-10-25" <= r.date <= "2026-10-31"]
        outside = [r for r in self.farm.records
                   if r.pen_id != "P3" and "2026-10-25" <= r.date <= "2026-10-31"]
        avg_in = sum(r.feed_kg for r in window) / len(window)
        avg_out = sum(r.feed_kg for r in outside) / len(outside)
        self.assertLess(avg_in, avg_out * 0.95)

    def test_pig_p2_007_feed_drops(self):
        target = [r for r in self.farm.records
                  if r.pig_id == "P2-007" and r.date >= "2026-10-27"]
        others = [r for r in self.farm.records
                  if r.pen_id == "P2" and r.pig_id != "P2-007"
                  and r.date >= "2026-10-27"]
        avg_target = sum(r.feed_kg for r in target) / len(target)
        avg_others = sum(r.feed_kg for r in others) / len(others)
        self.assertLess(avg_target, avg_others * 0.8)

    def test_pig_p4_003_activity_drops(self):
        target = [r for r in self.farm.records
                  if r.pig_id == "P4-003" and r.date >= "2026-10-19"]
        others = [r for r in self.farm.records
                  if r.pen_id == "P4" and r.pig_id != "P4-003"
                  and r.date >= "2026-10-19"]
        avg_target = sum(r.activity_index for r in target) / len(target)
        avg_others = sum(r.activity_index for r in others) / len(others)
        self.assertLess(avg_target, avg_others * 0.75)


if __name__ == "__main__":
    unittest.main()