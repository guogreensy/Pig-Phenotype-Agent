"""工具注册表测试（T6）。"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from src.tools.registry import (
    ToolError,
    ToolRegistry,
    ToolSpec,
    validate_args,
)

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_tools"


def _echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="回显输入",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "times": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            },
            "required": ["text"],
        },
        handler=lambda args: {"echo": args["text"] * args["times"]},
    )


class TestRegistration(unittest.TestCase):
    def test_register_and_introspect(self):
        registry = ToolRegistry()
        registry.register(_echo_spec())
        self.assertEqual(registry.names(), ["echo"])
        self.assertIn("echo", registry)
        self.assertEqual(len(registry), 1)
        schema = registry.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "echo")

    def test_duplicate_name_rejected(self):
        registry = ToolRegistry()
        registry.register(_echo_spec())
        with self.assertRaises(ValueError):
            registry.register(_echo_spec())

    def test_empty_name_rejected(self):
        registry = ToolRegistry()
        spec = ToolSpec(name="", description="x", parameters={}, handler=lambda a: None)
        with self.assertRaises(ValueError):
            registry.register(spec)

    def test_describe_lists_params(self):
        registry = ToolRegistry()
        registry.register(_echo_spec())
        text = registry.describe()
        self.assertIn("echo(", text)
        self.assertIn("回显输入", text)


class TestArgumentValidation(unittest.TestCase):
    def setUp(self):
        self.schema = _echo_spec().parameters

    def test_missing_required(self):
        with self.assertRaises(ToolError) as ctx:
            validate_args(self.schema, {})
        self.assertIn("缺少必填参数 text", str(ctx.exception))

    def test_unknown_param_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            validate_args(self.schema, {"text": "x", "bogus": 1})
        self.assertIn("不支持的参数 bogus", str(ctx.exception))

    def test_type_mismatch(self):
        with self.assertRaises(ToolError) as ctx:
            validate_args(self.schema, {"text": 123})
        self.assertIn("应为 string", str(ctx.exception))

    def test_bool_is_not_integer(self):
        # True 是 int 的子类，若不特判会被误认为合法整数
        with self.assertRaises(ToolError):
            validate_args(self.schema, {"text": "x", "times": True})

    def test_range_check(self):
        with self.assertRaises(ToolError) as ctx:
            validate_args(self.schema, {"text": "x", "times": 9})
        self.assertIn("不能大于 5", str(ctx.exception))

    def test_enum_check(self):
        with self.assertRaises(ToolError) as ctx:
            validate_args(self.schema, {"text": "x", "tags": ["c"]})
        self.assertIn("只能是", str(ctx.exception))

    def test_defaults_applied(self):
        cleaned = validate_args(self.schema, {"text": "x"})
        self.assertEqual(cleaned["times"], 1)

    def test_original_args_not_mutated(self):
        args = {"text": "x"}
        validate_args(self.schema, args)
        self.assertNotIn("times", args)

    def test_additional_properties_allowed(self):
        schema = dict(self.schema)
        schema["additionalProperties"] = True
        cleaned = validate_args(schema, {"text": "x", "extra": 1})
        self.assertEqual(cleaned["extra"], 1)

    def test_non_object_args_rejected(self):
        with self.assertRaises(ToolError):
            validate_args(self.schema, ["text"])

    def test_none_args_treated_as_empty(self):
        with self.assertRaises(ToolError):
            validate_args(self.schema, None)


class TestCalling(unittest.TestCase):
    def setUp(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.registry = ToolRegistry()
        self.registry.register(_echo_spec())

    def tearDown(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_successful_call(self):
        result = self.registry.call("echo", {"text": "ab", "times": 2})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"echo": "abab"})
        self.assertGreaterEqual(result.duration_ms, 0)

    def test_unknown_tool(self):
        result = self.registry.call("nope", {})
        self.assertFalse(result.ok)
        self.assertIn("未注册的工具", result.error)
        self.assertIn("echo", result.error)

    def test_invalid_args_do_not_raise(self):
        result = self.registry.call("echo", {})
        self.assertFalse(result.ok)
        self.assertIn("缺少必填参数", result.error)

    def test_handler_exception_is_captured(self):
        def boom(args):
            raise ZeroDivisionError("division by zero")

        self.registry.register(ToolSpec(name="boom", description="x",
                                        parameters={"type": "object", "properties": {}},
                                        handler=boom))
        result = self.registry.call("boom", {})
        self.assertFalse(result.ok)
        self.assertIn("ZeroDivisionError", result.error)
        self.assertIn("division by zero", result.error)

    def test_log_records_every_call(self):
        self.registry.call("echo", {"text": "a"})
        self.registry.call("echo", {})
        self.registry.call("nope", {})
        self.assertEqual(len(self.registry.log), 3)
        self.assertEqual([e["seq"] for e in self.registry.log], [1, 2, 3])
        self.assertEqual(self.registry.log[1]["ok"], False)

    def test_log_written_to_disk(self):
        path = TMP_ROOT / "calls.jsonl"
        registry = ToolRegistry(log_path=path)
        registry.register(_echo_spec())
        registry.call("echo", {"text": "z"})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["tool"], "echo")
        self.assertTrue(entry["ok"])

    def test_max_calls_guard(self):
        registry = ToolRegistry(max_calls=2)
        registry.register(_echo_spec())
        self.assertTrue(registry.call("echo", {"text": "a"}).ok)
        self.assertTrue(registry.call("echo", {"text": "b"}).ok)
        blocked = registry.call("echo", {"text": "c"})
        self.assertFalse(blocked.ok)
        self.assertIn("次数已达上限", blocked.error)

    def test_stats(self):
        self.registry.call("echo", {"text": "a"})
        self.registry.call("echo", {})
        stats = self.registry.stats()
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["by_tool"], {"echo": 2})

    def test_stats_empty(self):
        stats = ToolRegistry().stats()
        self.assertEqual(stats["calls"], 0)
        self.assertIsNone(stats["success_rate"])

    def test_reset_log(self):
        self.registry.call("echo", {"text": "a"})
        self.registry.reset_log()
        self.assertEqual(self.registry.log, [])

    def test_to_message_truncates(self):
        result = self.registry.call("echo", {"text": "x" * 5000})
        message = result.to_message(max_chars=100)
        self.assertLessEqual(len(message), 100 + 20)
        self.assertIn("已截断", message)


if __name__ == "__main__":
    unittest.main()