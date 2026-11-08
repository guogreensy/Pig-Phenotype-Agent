"""数据工具测试（T8）。"""

from __future__ import annotations

import unittest

from src.tools.data_tools import (
    aggregate_metric,
    build_data_tools,
    detect_anomalies,
    list_pens,
    query_pig_records,
)
from src.tools.mock_data import build_farm
from src.tools.registry import ToolError, ToolRegistry


class TestListPens(unittest.TestCase):
    def test_returns_all_pens(self):
        payload = list_pens()
        self.assertEqual(payload["source"], "simulated")
        self.assertEqual(len(payload["pens"]), 4)
        for pen in payload["pens"]:
            self.assertEqual(pen["pigs"], 12)
            self.assertIsNotNone(pen["avg_weight_kg"])
            self.assertTrue(pen["camera_id"].startswith("cam"))


class TestQueryRecords(unittest.TestCase):
    def test_filter_by_pen(self):
        payload = query_pig_records(pen_id="P1", limit=20)
        self.assertEqual(payload["returned"], 20)
        self.assertTrue(all(item["pen_id"] == "P1" for item in payload["records"]))

    def test_filter_by_pig_and_date(self):
        payload = query_pig_records(pig_id="P2-007",
                                    start_date="2026-10-27",
                                    end_date="2026-10-29")
        self.assertEqual(payload["returned"], 3)
        self.assertTrue(all(item["pig_id"] == "P2-007" for item in payload["records"]))
        self.assertEqual([item["date"] for item in payload["records"]],
                         ["2026-10-27", "2026-10-28", "2026-10-29"])

    def test_metric_selection(self):
        payload = query_pig_records(pig_id="P1-001", start_date="2026-10-08",
                                    end_date="2026-10-08", metrics=["weight_kg"])
        record = payload["records"][0]
        self.assertIn("weight_kg", record)
        self.assertNotIn("feed_kg", record)

    def test_truncation_flag(self):
        payload = query_pig_records(limit=10)
        self.assertEqual(payload["returned"], 10)
        self.assertTrue(payload["truncated"])

    def test_unknown_pen_raises_with_options(self):
        with self.assertRaises(ToolError) as ctx:
            query_pig_records(pen_id="P9")
        self.assertIn("P1", str(ctx.exception))

    def test_unknown_metric_raises(self):
        with self.assertRaises(ToolError):
            query_pig_records(metrics=["bogus"])

    def test_bad_date_raises(self):
        with self.assertRaises(ToolError) as ctx:
            query_pig_records(start_date="2026/10/08")
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_reversed_range_raises(self):
        with self.assertRaises(ToolError):
            query_pig_records(start_date="2026-10-20", end_date="2026-10-10")

    def test_empty_window_returns_empty(self):
        payload = query_pig_records(start_date="2025-01-01", end_date="2025-01-02")
        self.assertEqual(payload["records"], [])
        self.assertEqual(payload["returned"], 0)


class TestAggregate(unittest.TestCase):
    def test_mean_by_pen(self):
        payload = aggregate_metric("weight_kg", group_by="pen")
        self.assertEqual(payload["stat"], "mean")
        self.assertEqual(len(payload["groups"]), 4)
        for group in payload["groups"]:
            self.assertGreater(group["value"], 80)
            self.assertGreater(group["n"], 0)

    def test_count_by_day(self):
        payload = aggregate_metric("weight_kg", group_by="day", stat="count",
                                   start_date="2026-10-08", end_date="2026-10-10")
        self.assertEqual(len(payload["groups"]), 3)
        self.assertTrue(all(group["value"] == 48 for group in payload["groups"]))

    def test_max_by_pig(self):
        payload = aggregate_metric("weight_kg", group_by="pig", stat="max")
        self.assertEqual(len(payload["groups"]), 48)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ToolError):
            aggregate_metric("bogus")

    def test_unknown_group_raises(self):
        with self.assertRaises(ToolError):
            aggregate_metric("weight_kg", group_by="barn")

    def test_unknown_stat_raises(self):
        with self.assertRaises(ToolError):
            aggregate_metric("weight_kg", stat="median")


class TestAnomalies(unittest.TestCase):
    def test_finds_low_feed_pig(self):
        payload = detect_anomalies("feed_kg", days=7, z_threshold=2.0)
        targets = [item["target"] for item in payload["flags"]]
        self.assertIn("P2-007", targets)
        flag = next(item for item in payload["flags"] if item["target"] == "P2-007")
        self.assertLess(flag["z_score"], 0)
        self.assertEqual(flag["direction"], "偏低")

    def test_finds_low_activity_pig(self):
        payload = detect_anomalies("activity_index", days=7, z_threshold=2.0)
        targets = [item["target"] for item in payload["flags"]]
        self.assertIn("P4-003", targets)

    def test_finds_hot_pen(self):
        payload = detect_anomalies("temp_c", days=7, z_threshold=1.5, scope="pen")
        targets = [item["target"] for item in payload["flags"]]
        self.assertIn("P3", targets)
        flag = next(item for item in payload["flags"] if item["target"] == "P3")
        self.assertEqual(flag["direction"], "偏高")

    def test_no_flags_with_high_threshold(self):
        payload = detect_anomalies("weight_kg", days=7, z_threshold=5.0)
        self.assertEqual(payload["flags"], [])
        self.assertIn("未发现", payload["note"])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ToolError):
            detect_anomalies("bogus")

    def test_unknown_scope_raises(self):
        with self.assertRaises(ToolError):
            detect_anomalies(scope="barn")

    def test_window_is_reported(self):
        payload = detect_anomalies(days=7)
        self.assertEqual(payload["window"]["end_date"], "2026-11-06")
        self.assertEqual(payload["window"]["start_date"], "2026-10-31")


class TestRegistryIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        build_data_tools(self.registry)

    def test_all_tools_registered(self):
        self.assertEqual(sorted(self.registry.names()),
                         ["aggregate_metric", "detect_anomalies",
                          "list_pens", "query_pig_records"])

    def test_call_through_registry(self):
        result = self.registry.call("list_pens", {})
        self.assertTrue(result.ok)
        self.assertEqual(result.data["source"], "simulated")

    def test_enum_violation_reported_not_raised(self):
        result = self.registry.call("aggregate_metric",
                                    {"metric": "weight_kg", "group_by": "barn"})
        self.assertFalse(result.ok)
        self.assertIn("只能是", result.error)

    def test_handler_error_returns_ok_false(self):
        result = self.registry.call("query_pig_records", {"pen_id": "P9"})
        self.assertFalse(result.ok)
        self.assertIn("未知栏位", result.error)

    def test_defaults_from_schema(self):
        result = self.registry.call("detect_anomalies", {})
        self.assertTrue(result.ok)
        self.assertEqual(result.data["metric"], "feed_kg")
        self.assertEqual(result.data["days"], 7)


if __name__ == "__main__":
    unittest.main()