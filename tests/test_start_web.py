"""一键启动器的单元测试：不启动服务、不访问网络。"""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import start_web  # noqa: E402


class TestProbePort(unittest.TestCase):
    def test_returns_preferred_when_free(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        self.assertEqual(start_web.probe_port("127.0.0.1", free, span=1), free)

    def test_skips_occupied_port(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            occupied = busy.getsockname()[1]
            got = start_web.probe_port("127.0.0.1", occupied, span=2)
        self.assertIsNotNone(got)
        self.assertNotEqual(got, occupied)

    def test_none_when_span_exhausted(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            occupied = busy.getsockname()[1]
            self.assertIsNone(start_web.probe_port("127.0.0.1", occupied, span=1))


class TestBuildUrl(unittest.TestCase):
    def test_localhost_by_default(self):
        self.assertEqual(start_web.build_url("127.0.0.1", 7860), "http://127.0.0.1:7860/")

    def test_wildcard_host_maps_to_loopback(self):
        self.assertEqual(start_web.build_url("0.0.0.0", 7860), "http://127.0.0.1:7860/")
        self.assertEqual(start_web.build_url("", 8080), "http://127.0.0.1:8080/")

    def test_custom_path(self):
        self.assertEqual(start_web.build_url("127.0.0.1", 7861, "/api/health"),
                         "http://127.0.0.1:7861/api/health")


class TestFindHealthyPort(unittest.TestCase):
    def test_returns_running_port(self):
        def probe(url):
            return url.endswith(":7862/api/health")

        self.assertEqual(
            start_web.find_healthy_port("127.0.0.1", 7860, probe=probe), 7862)

    def test_returns_none_when_nothing_running(self):
        self.assertIsNone(start_web.find_healthy_port("127.0.0.1", 7860,
                                                      probe=lambda url: False))


class TestWaitUntilReady(unittest.TestCase):
    def test_returns_port_once_ready(self):
        state = {"calls": 0}

        def finder(host, preferred, span):
            state["calls"] += 1
            return 7861 if state["calls"] >= 3 else None

        now = {"t": 0.0}

        def clock():
            return now["t"]

        def sleep(seconds):
            now["t"] += seconds

        got = start_web.wait_until_ready("127.0.0.1", 7860, 5.0,
                                         finder=finder, sleep=sleep, clock=clock)
        self.assertEqual(got, 7861)
        self.assertEqual(state["calls"], 3)

    def test_returns_none_on_timeout(self):
        now = {"t": 0.0}

        def clock():
            return now["t"]

        def sleep(seconds):
            now["t"] += seconds

        got = start_web.wait_until_ready("127.0.0.1", 7860, 2.0,
                                         finder=lambda h, p, s: None,
                                         sleep=sleep, clock=clock)
        self.assertIsNone(got)


class TestDescribeEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = PROJECT_ROOT / "tests" / "_tmp_start_web"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp.exists():
            for item in self.tmp.iterdir():
                item.unlink()
            self.tmp.rmdir()

    def test_missing_file(self):
        missing = PROJECT_ROOT / "no_such_env_file.env"
        self.assertIn(".env", start_web.describe_env(missing))

    def test_file_without_key(self):
        path = self.tmp / ".env"
        path.write_text("PIG_PROVIDER=zhipu\n", encoding="utf-8")
        self.assertIn("API_KEY", start_web.describe_env(path))

    def test_file_with_key(self):
        path = self.tmp / ".env"
        path.write_text("ZHIPUAI_API_KEY=abc\n", encoding="utf-8")
        self.assertIsNone(start_web.describe_env(path))


class TestOpenBrowser(unittest.TestCase):
    def test_failure_does_not_raise(self):
        def boom(url):
            raise OSError("no browser")

        self.assertFalse(start_web.open_browser("http://127.0.0.1:7860/", opener=boom))

    def test_success(self):
        seen = []
        self.assertTrue(start_web.open_browser("http://127.0.0.1:7860/",
                                               opener=seen.append))
        self.assertEqual(seen, ["http://127.0.0.1:7860/"])


if __name__ == "__main__":
    unittest.main()