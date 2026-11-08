"""``config`` 的单元测试：覆盖"API 配置缺失时的提示"（需求 10.4）。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import config
from config import ConfigError, MissingAPIKeyError, get_settings, load_dotenv


class TestLoadDotenv(unittest.TestCase):
    def test_parse_basic(self):
        # 不用 tempfile：沙箱可能禁止写系统临时目录，改用项目内临时目录
        from pathlib import Path

        tmp = Path(__file__).resolve().parent / "_tmp_env"
        tmp.mkdir(exist_ok=True)
        path = tmp / ".env"
        try:
            path.write_text(
                "# 注释行\n"
                "ZHIPUAI_API_KEY=abc123\n"
                "\n"
                "export PIG_QPS=2.5\n"
                'PIG_MODEL="glm-4.6v-flash"\n'
                "PIG_TEMPERATURE=0\n",
                encoding="utf-8",
            )
            values = load_dotenv(path)
        finally:
            path.unlink(missing_ok=True)
            tmp.rmdir()
        self.assertEqual(values["ZHIPUAI_API_KEY"], "abc123")
        self.assertEqual(values["PIG_QPS"], "2.5")
        self.assertEqual(values["PIG_MODEL"], "glm-4.6v-flash")
        self.assertEqual(values["PIG_TEMPERATURE"], "0")

    def test_missing_file(self):
        from pathlib import Path

        self.assertEqual(load_dotenv(Path("不存在/.env")), {})


class TestGetSettings(unittest.TestCase):
    def setUp(self):
        # 清空可能影响结果的环境变量，避免机器状态干扰测试
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in list(os.environ):
            if name.startswith("PIG_") or name in ("ZHIPUAI_API_KEY", "DASHSCOPE_API_KEY"):
                os.environ.pop(name, None)
        # 屏蔽真实 .env 文件，保证测试与本地配置无关
        no_env_patch = mock.patch.object(config, "ENV_PATH", config.ROOT / ".env.not-exist")
        no_env_patch.start()
        self.addCleanup(no_env_patch.stop)

    def test_missing_key_raises_chinese_error(self):
        with self.assertRaises(MissingAPIKeyError) as ctx:
            get_settings(require_key=True)
        message = str(ctx.exception)
        self.assertIn("ZHIPUAI_API_KEY", message)
        self.assertIn(".env", message)

    def test_missing_key_allowed_when_not_required(self):
        settings = get_settings(require_key=False)
        self.assertFalse(settings.has_api_key)
        self.assertEqual(settings.provider, "zhipu")
        self.assertEqual(settings.model, "glm-4.6v-flash")

    def test_defaults(self):
        settings = get_settings()
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.max_tokens, 512)
        self.assertGreaterEqual(settings.concurrency, 1)
        self.assertGreater(settings.qps, 0)

    def test_key_from_env(self):
        os.environ["ZHIPUAI_API_KEY"] = "sk-test"
        settings = get_settings(require_key=True)
        self.assertTrue(settings.has_api_key)
        self.assertEqual(settings.api_key, "sk-test")

    def test_unsupported_provider(self):
        os.environ["PIG_PROVIDER"] = "openai"
        with self.assertRaises(ConfigError):
            get_settings()

    def test_invalid_number(self):
        os.environ["PIG_QPS"] = "abc"
        with self.assertRaises(ConfigError) as ctx:
            get_settings()
        self.assertIn("PIG_QPS", str(ctx.exception))

    def test_provider_switch_to_dashscope(self):
        os.environ["PIG_PROVIDER"] = "dashscope"
        settings = get_settings()
        self.assertEqual(settings.provider, "dashscope")
        self.assertEqual(settings.model, "qwen3-vl-plus")
        self.assertIn("dashscope.aliyuncs.com", settings.base_url)


if __name__ == "__main__":
    unittest.main()
