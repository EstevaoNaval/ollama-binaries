import hashlib
import json
import subprocess

import pytest

import mirror
from mirror import (
    GITHUB_RELEASE_MAX_BYTES,
    WINDOWS_AMD64,
    ArtifactError,
    UnsupportedPlatformError,
    candidate_release_tag,
    fetch_latest,
    files_to_mirror,
    locks_match,
    mirrored_asset_url,
    select_candidate_assets,
    stable_lock_from_candidate,
)


def _release(names, *, digest="sha256:" + ("ab" * 32), size=10):
    assets = [
        {
            "name": name,
            "browser_download_url": f"https://example.test/{name}",
            "digest": digest,
            "size": size,
        }
        for name in names
    ]
    return {"tag_name": "v0.33.2", "html_url": "https://example.test", "assets": assets}


def test_candidate_tag_and_mirrored_url():
    assert candidate_release_tag("v0.34.0") == "v0.34.0"
    url = mirrored_asset_url(
        "molmodcs/ollama-binaries",
        "v0.34.0",
        "ollama-windows-amd64.zip",
    )
    assert url == (
        "https://github.com/molmodcs/ollama-binaries/releases/download/"
        "v0.34.0/ollama-windows-amd64.zip"
    )


def test_candidate_tag_rejects_empty():
    with pytest.raises(ArtifactError, match="tag_name"):
        candidate_release_tag("  ")


def test_mirrored_url_rejects_bad_repo():
    with pytest.raises(ArtifactError, match="owner/name"):
        mirrored_asset_url("ollama-binaries", "tag", "a.zip")


def test_select_candidate_missing_platform():
    release = _release(["ollama-linux-amd64.tar.zst", "ollama-darwin.tgz"])
    with pytest.raises(UnsupportedPlatformError, match="windows-amd64"):
        select_candidate_assets(release, platforms=(WINDOWS_AMD64,))


def test_select_candidate_empty_digest_leaves_sha_blank():
    names = [
        "ollama-windows-amd64.zip",
        "ollama-linux-amd64.tar.zst",
        "ollama-darwin.tgz",
    ]
    release = _release(names, digest="")
    candidate = select_candidate_assets(release)
    assert all(item["sha256"] == "" for item in candidate["assets"])


def test_select_candidate_digest_without_prefix():
    names = [
        "ollama-windows-amd64.zip",
        "ollama-linux-amd64.tar.zst",
        "ollama-darwin.tgz",
    ]
    raw = "cd" * 32
    release = _release(names, digest=raw)
    candidate = select_candidate_assets(release)
    assert candidate["assets"][0]["sha256"] == raw


def test_select_candidate_rejects_oversize():
    names = [
        "ollama-windows-amd64.zip",
        "ollama-linux-amd64.tar.zst",
        "ollama-darwin.tgz",
    ]
    release = _release(names, size=GITHUB_RELEASE_MAX_BYTES)
    with pytest.raises(ArtifactError, match="2 GiB"):
        select_candidate_assets(release)


def test_stable_lock_rewrites_urls_and_ignores_sha256sum():
    names = [
        "ollama-windows-amd64.zip",
        "ollama-linux-amd64.tar.zst",
        "ollama-darwin.tgz",
        "sha256sum.txt",
    ]
    candidate = select_candidate_assets(_release(names))
    assert candidate["sha256sum_txt"]["name"] == "sha256sum.txt"
    assert len(files_to_mirror(candidate)) == 4
    lock = stable_lock_from_candidate(
        candidate,
        "molmodcs/ollama-binaries",
        "v0.33.2",
    )
    assert lock["channel"] == "stable"
    assert "sha256sum_txt" not in lock
    assert all(
        item["url"].startswith(
            "https://github.com/molmodcs/ollama-binaries/releases/download/"
        )
        for item in lock["assets"]
    )
    assert "huggingface" not in json.dumps(lock)
    assert ".gguf" not in json.dumps(lock)


def test_locks_match_ignores_note():
    left = {
        "channel": "stable",
        "ollama_version": "0.33.2",
        "upstream_tag": "v0.33.2",
        "note": "a",
        "assets": [
            {
                "platform_id": "darwin",
                "name": "ollama-darwin.tgz",
                "url": "https://example.test/a",
                "sha256": "AB" * 32,
                "size": 1,
            }
        ],
    }
    right = dict(left)
    right["note"] = "b"
    right["assets"] = [dict(left["assets"][0], sha256="ab" * 32)]
    assert locks_match(left, right)


def test_fetch_latest_sends_bearer(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"tag_name":"v0.33.2","assets":[]}'

    def fake_urlopen(request, timeout=0):
        captured["auth"] = request.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr("mirror.urllib.request.urlopen", fake_urlopen)
    fetch_latest(token="abc")
    assert captured["auth"] == "Bearer abc"


def test_select_candidate_from_latest_shape():
    names = [
        "install.ps1",
        "install.sh",
        "ollama-darwin.tgz",
        "Ollama-darwin.zip",
        "ollama-linux-amd64-mlx.tar.zst",
        "ollama-linux-amd64-rocm.tar.zst",
        "ollama-linux-amd64.tar.zst",
        "ollama-linux-arm64-jetpack5.tar.zst",
        "ollama-linux-arm64.tar.zst",
        "ollama-windows-amd64-mlx.zip",
        "ollama-windows-amd64-rocm.zip",
        "ollama-windows-amd64.zip",
        "ollama-windows-arm64.zip",
        "Ollama.dmg",
        "OllamaSetup.exe",
        "sha256sum.txt",
    ]
    assets = [
        {
            "name": name,
            "browser_download_url": f"https://example.test/{name}",
            "digest": "sha256:" + ("ab" * 32),
            "size": 10,
        }
        for name in names
    ]
    release = {"tag_name": "v0.33.2", "html_url": "https://example.test", "assets": assets}
    candidate = select_candidate_assets(release)
    ids = {item["platform_id"]: item["name"] for item in candidate["assets"]}
    assert ids["windows-amd64"] == "ollama-windows-amd64.zip"
    assert ids["linux-amd64"] == "ollama-linux-amd64.tar.zst"
    assert ids["darwin"] == "ollama-darwin.tgz"
    assert candidate["channel"] == "candidate"
    assert candidate["ollama_version"] == "0.33.2"


def _candidate():
    names = [
        "ollama-windows-amd64.zip",
        "ollama-linux-amd64.tar.zst",
        "ollama-darwin.tgz",
        "sha256sum.txt",
    ]
    assets = [
        {
            "name": name,
            "browser_download_url": f"https://github.com/ollama/ollama/releases/download/v0.34.0/{name}",
            "digest": "sha256:" + hashlib.sha256(name.encode()).hexdigest(),
            "size": len(name),
        }
        for name in names
    ]
    release = {
        "tag_name": "v0.34.0",
        "html_url": "https://github.com/ollama/ollama/releases/tag/v0.34.0",
        "assets": assets,
    }
    return select_candidate_assets(release)


def _fake_download(url, dest, *, sha256, size, timeout=0, progress=True):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(dest.name.encode())
    return dest


class _Recorder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv, cwd=None, check=True):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        if argv[:3] == ["gh", "release", "view"]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="not found")
        if argv[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(argv, 0, stdout="[]", stderr="")
        if argv[:2] == ["git", "diff"] and "--cached" in argv:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        if check is False:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "huggingface" in joined.lower() or ".gguf" in joined.lower():
            raise ArtifactError(joined)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_apply_mirrors_binaries_and_opens_pr(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "molmodcs/ollama-binaries")
    monkeypatch.setattr(mirror.shutil, "which", lambda name: f"/usr/bin/{name}")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    previous = {
        "channel": "stable",
        "ollama_version": "0.33.2",
        "upstream_tag": "v0.33.2",
        "note": "old",
        "assets": [
            {
                "platform_id": "darwin",
                "name": "ollama-darwin.tgz",
                "url": "https://github.com/ollama/ollama/releases/download/v0.33.2/ollama-darwin.tgz",
                "sha256": "aa" * 32,
                "size": 1,
            }
        ],
    }
    (repo_root / "runtime.lock.json").write_text(json.dumps(previous, indent=2) + "\n")

    recorder = _Recorder()
    work = tmp_path / "work"
    candidate = _candidate()
    downloaded = []

    def tracking_download(url, dest, **kwargs):
        downloaded.append(url)
        assert "huggingface" not in url.lower()
        assert not url.endswith(".gguf")
        return _fake_download(url, dest, **kwargs)

    mirror.apply(
        repo_root=repo_root,
        work_dir=work,
        candidate=candidate,
        download_fn=tracking_download,
        run=recorder,
    )

    lock = json.loads((repo_root / "runtime.lock.json").read_text())
    legacy = json.loads((repo_root / "runtime.legacy.lock.json").read_text())
    assert legacy["ollama_version"] == "0.33.2"
    assert lock["channel"] == "stable"
    assert lock["ollama_version"] == "0.34.0"
    for asset in lock["assets"]:
        assert asset["url"].startswith(
            "https://github.com/molmodcs/ollama-binaries/releases/download/v0.34.0/"
        )
    assert any(call[:3] == ["gh", "release", "create"] for call in recorder.calls)
    create = next(call for call in recorder.calls if call[:3] == ["gh", "release", "create"])
    assert "--draft" not in create
    assert "--prerelease" in create
    assert any(call[:3] == ["gh", "pr", "create"] for call in recorder.calls)
    blob = " ".join(" ".join(call) for call in recorder.calls)
    assert "huggingface" not in blob.lower()
    assert ".gguf" not in blob.lower()
    assert downloaded
    assert all("ollama/ollama" in url for url in downloaded)


def test_apply_skips_when_lock_already_mirrored(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "molmodcs/ollama-binaries")
    monkeypatch.setattr(mirror.shutil, "which", lambda name: f"/usr/bin/{name}")

    candidate = _candidate()
    proposed = stable_lock_from_candidate(candidate, "molmodcs/ollama-binaries", "v0.34.0")
    (tmp_path / "runtime.lock.json").write_text(json.dumps(proposed, indent=2) + "\n")

    def boom(*args, **kwargs):
        raise AssertionError("download should not run")

    recorder = _Recorder()
    mirror.apply(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        candidate=candidate,
        download_fn=boom,
        run=recorder,
    )
    assert recorder.calls == []


def test_download_mirror_files_fills_empty_digest(tmp_path):
    payload = b"hello-ollama"
    candidate = {
        "assets": [
            {
                "platform_id": "darwin",
                "name": "ollama-darwin.tgz",
                "url": "https://github.com/ollama/ollama/releases/download/v0.34.0/ollama-darwin.tgz",
                "sha256": "",
                "size": 0,
            }
        ],
        "sha256sum_txt": None,
    }

    def fake_download(url, dest, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return dest

    files = mirror.download_mirror_files(
        candidate, tmp_path, download_fn=fake_download
    )
    assert files[0].name == "ollama-darwin.tgz"
    assert candidate["assets"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert candidate["assets"][0]["size"] == len(payload)


def test_apply_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "molmodcs/ollama-binaries")
    with pytest.raises(SystemExit, match="GITHUB_TOKEN"):
        mirror.apply(candidate=_candidate())
