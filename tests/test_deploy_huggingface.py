from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from deploy_huggingface import (  # noqa: E402
    INCLUDE, SPACE_README, check_health, expected_health, space_host, stage_files, wait_until_running,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeApi:
    def __init__(self, stages, host="https://user-space.hf.space"):
        self.stages, self.host, self.calls = list(stages), host, 0

    def space_info(self, repo_id):
        stage = self.stages[min(self.calls, len(self.stages) - 1)]
        self.calls += 1
        return SimpleNamespace(runtime=SimpleNamespace(stage=stage), host=self.host)


class StageTests(unittest.TestCase):
    def test_stages_only_what_the_dockerfile_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = stage_files(ROOT, INCLUDE, SPACE_README, Path(tmp) / "stage")
        self.assertIn("Dockerfile", files)
        self.assertIn("README.md", files)
        self.assertIn("artifacts/model_metadata.json", files)
        self.assertTrue(any(f.startswith("src/hotel_cancellation/") for f in files))
        self.assertFalse(any(f.startswith(("data/", "outputs/", "notebooks/", ".git")) for f in files))
        self.assertFalse(any("__pycache__" in f for f in files))

    def test_space_readme_declares_a_docker_space_on_the_api_port(self):
        header = SPACE_README.read_text().split("---")[1]
        self.assertIn("sdk: docker", header)
        self.assertIn("app_port: 8000", header)
        self.assertLessEqual(len(header.split("short_description:")[1].strip()), 60)


class WaitTests(unittest.TestCase):
    def wait(self, stages, **kwargs):
        clock = FakeClock()
        return wait_until_running(FakeApi(stages), "user/space", sleep=clock.sleep, clock=clock, log=lambda _: None, **kwargs)

    def test_a_stale_running_stage_does_not_end_the_wait(self):
        api = FakeApi(["RUNNING", "RUNNING_BUILDING", "RUNNING_BUILDING", "RUNNING"])
        clock = FakeClock()
        wait_until_running(api, "user/space", sleep=clock.sleep, clock=clock, log=lambda _: None)
        self.assertEqual(api.calls, 4)  # it kept waiting past the first, stale RUNNING

    def test_an_unchanged_space_returns_after_the_grace_period(self):
        api = FakeApi(["RUNNING"])
        clock = FakeClock()
        wait_until_running(api, "user/space", grace=60, interval=15, sleep=clock.sleep, clock=clock, log=lambda _: None)
        self.assertEqual(clock.now, 60)

    def test_build_errors_fail_with_a_pointer_to_the_logs(self):
        with self.assertRaisesRegex(RuntimeError, "Logs tab"):
            self.wait(["BUILDING", "BUILD_ERROR"])

    def test_times_out(self):
        with self.assertRaises(TimeoutError):
            self.wait(["BUILDING"], timeout=100)


class HealthTests(unittest.TestCase):
    def test_waits_for_the_committed_model_version(self):
        responses = [OSError("not routed yet"), {"status": "ok", "model_version": "old"},
                     {"status": "ok", "model_version": expected_health()["model_version"]}]
        calls = []

        def fetch(url):
            calls.append(url)
            item = responses[len(calls) - 1]
            if isinstance(item, Exception):
                raise item
            return item

        body = check_health("https://h", expected_health(), fetch=fetch, sleep=lambda _: None, log=lambda _: None)
        self.assertEqual(len(calls), 3)
        self.assertEqual(body["status"], "ok")

    def test_gives_up_when_the_version_never_matches(self):
        with self.assertRaises(RuntimeError):
            check_health("https://h", {"model_version": "x"}, attempts=3, sleep=lambda _: None,
                         fetch=lambda url: {"model_version": "y"}, log=lambda _: None)

    def test_host_falls_back_to_the_documented_pattern(self):
        self.assertEqual(space_host(SimpleNamespace(host=None), "My-User/hotel_api"), "https://my-user-hotel-api.hf.space")
        self.assertEqual(space_host(SimpleNamespace(host="https://a.hf.space/"), "x/y"), "https://a.hf.space")


if __name__ == "__main__":
    unittest.main()
