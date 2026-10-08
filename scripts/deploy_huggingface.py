"""Deploy the scoring API to a Hugging Face Docker Space and verify the live service.

CI (.github/workflows/deploy-huggingface.yml) sets HF_TOKEN and HF_SPACE_ID:

    python scripts/deploy_huggingface.py

It copies only the files the Dockerfile needs, uploads them as one commit, waits for the
Space to build and start, then checks that /health reports the committed model version.
"""

import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["Dockerfile", "requirements-api.txt", "pyproject.toml", "src", "artifacts"]
SPACE_README = ROOT / "deploy" / "huggingface" / "README.md"
BUILDING = {"BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"}
FAILED = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "DELETING"}


def stage_files(root: Path, include: list[str], readme: Path, target: Path) -> list[str]:
    """Copy exactly the listed files, plus the Space README, into a clean folder."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for name in include:
        source, destination = root / name, target / name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=ignore)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(readme, target / "README.md")
    return sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())


def stage_name(runtime) -> str:
    value = getattr(runtime, "stage", None)
    return str(getattr(value, "value", value) or "UNKNOWN")


def wait_until_running(api, repo_id, *, timeout=3300, grace=180, interval=15,
                       sleep=time.sleep, clock=time.monotonic, log=print):
    """Wait for the new build to finish.

    Right after an upload the Space can still report the previous build as RUNNING. A
    RUNNING stage therefore only counts once a build has been seen, or after ``grace``
    seconds when nothing rebuilt (for example an identical upload).
    """
    start, last, saw_build = clock(), None, False
    while clock() - start < timeout:
        info = api.space_info(repo_id)
        stage = stage_name(info.runtime)
        if stage != last:
            log(f"Space stage: {stage}")
            last = stage
        saw_build = saw_build or stage in BUILDING
        if stage in FAILED:
            raise RuntimeError(
                f"The Space ended in {stage}. Open https://huggingface.co/spaces/{repo_id} and read its Logs tab."
            )
        if stage == "RUNNING" and (saw_build or clock() - start >= grace):
            return info
        sleep(interval)
    raise TimeoutError(f"The Space was not running after {timeout} s; last stage {last}.")


def space_host(info, repo_id: str) -> str:
    if getattr(info, "host", None):
        return info.host.rstrip("/")
    slug = repo_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    return f"https://{slug}.hf.space"


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=20) as response:
        return json.load(response)


def check_health(host: str, expected: dict, *, attempts=40, interval=10,
                 sleep=time.sleep, fetch=fetch_json, log=print) -> dict:
    """Poll /health until it reports the expected values (a stale build would not)."""
    last = None
    for _ in range(attempts):
        try:
            body = fetch(f"{host}/health")
            if all(body.get(key) == value for key, value in expected.items()):
                return body
            last = f"unexpected health response {body}"
        except Exception as error:  # noqa: BLE001 - the route can lag behind the stage
            last = repr(error)
        log(f"Waiting for {host}/health ({last})")
        sleep(interval)
    raise RuntimeError(f"{host}/health never matched {expected}; last: {last}")


def expected_health(root: Path = ROOT) -> dict:
    metadata = json.loads((root / "artifacts" / "model_metadata.json").read_text())
    return {"status": "ok", "model_version": metadata["artifact_sha256"][:12]}


def main() -> None:
    from huggingface_hub import HfApi

    token, repo_id = os.environ["HF_TOKEN"], os.environ["HF_SPACE_ID"]
    api = HfApi(token=token)
    target = ROOT / "stage"
    files = stage_files(ROOT, INCLUDE, SPACE_README, target)
    print(f"Uploading {len(files)} files to {repo_id}")
    api.upload_folder(
        folder_path=str(target), repo_id=repo_id, repo_type="space",
        commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
        delete_patterns=["*"],
    )
    info = wait_until_running(api, repo_id)
    host = space_host(info, repo_id)
    health = check_health(host, expected_health())
    print(f"Live: {host}/docs  {health}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(f"### Deployed\n\n[{host}/docs]({host}/docs) is serving model `{health['model_version']}`.\n")


if __name__ == "__main__":
    main()
