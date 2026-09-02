#!/usr/bin/env python3
"""Discover ollama/ollama latest portable assets and optionally mirror them.

Default: print candidate JSON. No writes. No GGUF.

  python mirror.py
  python mirror.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen

OLLAMA_LATEST_URL = "https://api.github.com/repos/ollama/ollama/releases/latest"
GITHUB_RELEASE_MAX_BYTES = 2 * 1024 * 1024 * 1024
USER_AGENT = "molmodcs-ollama-binaries"
LOCK_RELATIVE = Path("runtime.lock.json")
LEGACY_RELATIVE = Path("runtime.legacy.lock.json")
DOWNLOAD_TIMEOUT = 600.0
_CHUNK = 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parent
RunFn = Callable[..., subprocess.CompletedProcess]


class ArtifactError(Exception):
    """Download, hash, GitHub payload, or disk-space failure."""


class UnsupportedPlatformError(Exception):
    """No portable archive matched the requested platform."""


@dataclass(frozen=True)
class Platform:
    archive_kind: str
    platform_id: str
    required: tuple[str, ...]
    excluded: tuple[str, ...]
    preferred: str

    def matches_asset(self, name: str) -> bool:
        lower = name.lower()
        if any(token.lower() not in lower for token in self.required):
            return False
        if any(token.lower() in lower for token in self.excluded):
            return False
        return True

    def preferred_name(self) -> str:
        return self.preferred


WINDOWS_AMD64 = Platform(
    archive_kind="zip",
    platform_id="windows-amd64",
    required=("ollama-windows-amd64", ".zip"),
    excluded=("-rocm", "-mlx", "setup.exe"),
    preferred="ollama-windows-amd64.zip",
)
LINUX_AMD64 = Platform(
    archive_kind="tar.zst",
    platform_id="linux-amd64",
    required=("ollama-linux-amd64", ".tar.zst"),
    excluded=("-rocm", "-mlx"),
    preferred="ollama-linux-amd64.tar.zst",
)
DARWIN = Platform(
    archive_kind="tgz",
    platform_id="darwin",
    required=("ollama-darwin.tgz",),
    excluded=(".dmg", "ollama-darwin.zip"),
    preferred="ollama-darwin.tgz",
)

DEFAULT_PLATFORMS: tuple[Platform, ...] = (WINDOWS_AMD64, LINUX_AMD64, DARWIN)


def select_asset_name(assets: Iterable[str], host: Platform) -> str:
    names = list(assets)
    preferred = host.preferred_name().lower()
    for name in names:
        if name.lower() == preferred:
            return name
    matches = [name for name in names if host.matches_asset(name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact = [name for name in matches if name.lower() == preferred]
        if exact:
            return exact[0]
        raise UnsupportedPlatformError(
            f"ambiguous Ollama assets for {host.platform_id}: {matches}"
        )
    raise UnsupportedPlatformError(
        f"no Ollama asset for {host.platform_id} in {names}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_disk_space(dest_dir: Path, needed_bytes: int, *, margin: float = 1.1) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    required = int(needed_bytes * margin)
    if free < required:
        raise ArtifactError(
            f"not enough disk space under {dest_dir}: "
            f"need {required} bytes, have {free}"
        )


def file_matches(path: Path, sha256: str, size: int) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != size:
        return False
    return sha256_file(path).lower() == sha256.lower()


def download_hashed(
    url: str,
    dest: Path,
    *,
    sha256: str,
    size: int,
    timeout: float = 120.0,
    progress: bool = True,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sha256 and file_matches(dest, sha256, size):
        return dest

    ensure_disk_space(dest.parent, size if size > 0 else _CHUNK)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if progress:
                    print(
                        f"[ollama-binaries] download {dest.name}: "
                        f"{written}/{size} bytes",
                        file=sys.stderr,
                    )
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise ArtifactError(f"download failed: {url}: {exc}") from exc

    got = digest.hexdigest().lower()
    if sha256:
        if written != size or got != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise ArtifactError(
                f"hash/size mismatch for {dest.name}: "
                f"got {written} bytes sha256={got}, expected {size} {sha256}"
            )
    elif size > 0 and written != size:
        tmp.unlink(missing_ok=True)
        raise ArtifactError(
            f"size mismatch for {dest.name}: got {written} bytes, expected {size}"
        )
    tmp.replace(dest)
    return dest


def fetch_latest(url: str = OLLAMA_LATEST_URL, token: str | None = None) -> dict[str, Any]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    secret = token if token is not None else (
        os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    )
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _digest(asset: dict[str, Any]) -> str:
    raw = str(asset.get("digest") or "")
    if raw.startswith("sha256:"):
        return raw.split(":", 1)[1]
    return raw


def candidate_release_tag(upstream_tag: str) -> str:
    tag = str(upstream_tag or "").strip()
    if not tag:
        raise ArtifactError("upstream release has no tag_name")
    return tag


def mirrored_asset_url(repo: str, tag: str, name: str) -> str:
    repo = str(repo or "").strip()
    if not repo or "/" not in repo:
        raise ArtifactError(f"GITHUB_REPOSITORY must be owner/name, got {repo!r}")
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def lock_identity(doc: dict[str, Any]) -> tuple[Any, ...]:
    assets = tuple(
        (
            str(item.get("platform_id") or ""),
            str(item.get("name") or ""),
            str(item.get("url") or ""),
            str(item.get("sha256") or "").lower(),
            int(item.get("size") or 0),
        )
        for item in doc.get("assets") or []
    )
    return (
        str(doc.get("channel") or ""),
        str(doc.get("ollama_version") or ""),
        str(doc.get("upstream_tag") or ""),
        assets,
    )


def locks_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return lock_identity(left) == lock_identity(right)


def stable_lock_from_candidate(
    candidate: dict[str, Any],
    repo: str,
    tag: str,
) -> dict[str, Any]:
    assets = []
    for item in candidate.get("assets") or []:
        name = str(item.get("name") or "")
        assets.append(
            {
                "platform_id": item["platform_id"],
                "name": name,
                "url": mirrored_asset_url(repo, tag, name),
                "sha256": str(item.get("sha256") or ""),
                "size": int(item.get("size") or 0),
            }
        )
    version = str(candidate.get("ollama_version") or "")
    upstream = candidate.get("upstream_tag")
    return {
        "channel": "stable",
        "ollama_version": version,
        "upstream_tag": upstream,
        "note": (
            f"Pinned Ollama {upstream} mirrored from ollama/ollama to "
            f"{repo} GitHub Release {tag}. GGUF stays on Hugging Face."
        ),
        "assets": assets,
    }


def files_to_mirror(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    items = [dict(asset) for asset in candidate.get("assets") or []]
    sha_txt = candidate.get("sha256sum_txt")
    if sha_txt:
        items.append(dict(sha_txt))
    return items


def select_candidate_assets(
    release: dict[str, Any],
    platforms: tuple[Platform, ...] = DEFAULT_PLATFORMS,
) -> dict[str, Any]:
    assets = list(release.get("assets") or [])
    by_name = {str(item.get("name") or ""): item for item in assets}

    selected = []
    for host in platforms:
        name = select_asset_name(by_name.keys(), host)
        item = by_name.get(name)
        if item is None:
            raise ArtifactError(
                f"no GitHub asset payload for {host.platform_id} ({name})"
            )
        url = item.get("browser_download_url")
        if not url:
            raise ArtifactError(f"asset {name} has no browser_download_url")
        size = int(item.get("size") or 0)
        if size >= GITHUB_RELEASE_MAX_BYTES:
            raise ArtifactError(
                f"{name} is {size} bytes; GitHub Releases cap is 2 GiB"
            )
        selected.append(
            {
                "platform_id": host.platform_id,
                "name": name,
                "url": url,
                "sha256": _digest(item),
                "size": size,
            }
        )

    sha_txt = by_name.get("sha256sum.txt")
    return {
        "channel": "candidate",
        "ollama_version": str(release.get("tag_name") or "").lstrip("v"),
        "upstream_tag": release.get("tag_name"),
        "html_url": release.get("html_url"),
        "assets": selected,
        "sha256sum_txt": None
        if sha_txt is None
        else {
            "name": "sha256sum.txt",
            "url": sha_txt.get("browser_download_url"),
            "sha256": _digest(sha_txt),
            "size": int(sha_txt.get("size") or 0),
        },
        "note": (
            "Mirror these files to this repository's GitHub Release "
            "(prerelease <upstream-tag>) and wait for human approval "
            "before merging the promote PR into runtime.lock.json."
        ),
    }


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(argv)}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def _require_apply_tools() -> tuple[str, str]:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    if not token:
        raise SystemExit("GITHUB_TOKEN or GH_TOKEN required for --apply")
    if not repo:
        raise SystemExit("GITHUB_REPOSITORY required for --apply (owner/name)")
    if shutil.which("gh") is None:
        raise SystemExit("gh CLI not found")
    if shutil.which("git") is None:
        raise SystemExit("git not found")
    return token, repo


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fill_digest(item: dict[str, Any], dest: Path) -> None:
    if not item.get("sha256"):
        item["sha256"] = sha256_file(dest)
    if not int(item.get("size") or 0):
        item["size"] = dest.stat().st_size


def download_mirror_files(
    candidate: dict[str, Any],
    work_dir: Path,
    *,
    download_fn: Callable[..., Path] = download_hashed,
) -> list[Path]:
    paths: list[Path] = []
    for item in files_to_mirror(candidate):
        url = str(item.get("url") or "")
        name = str(item.get("name") or "")
        if not url or not name:
            raise ArtifactError(f"mirror item missing url or name: {item!r}")
        dest = work_dir / name
        download_fn(
            url,
            dest,
            sha256=str(item.get("sha256") or ""),
            size=int(item.get("size") or 0),
            timeout=DOWNLOAD_TIMEOUT,
        )
        _fill_digest(item, dest)
        paths.append(dest)
    for asset in candidate["assets"]:
        dest = work_dir / str(asset["name"])
        if dest.is_file():
            _fill_digest(asset, dest)
    sha_txt = candidate.get("sha256sum_txt")
    if sha_txt:
        dest = work_dir / str(sha_txt["name"])
        if dest.is_file():
            _fill_digest(sha_txt, dest)
    return paths


def _release_asset_sizes(tag: str, run: RunFn) -> dict[str, int] | None:
    result = run(["gh", "release", "view", tag, "--json", "assets"], check=False)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout or "{}")
    return {
        str(item.get("name") or ""): int(item.get("size") or 0)
        for item in payload.get("assets") or []
    }


def publish_prerelease(
    tag: str,
    version: str,
    files: list[Path],
    *,
    run: RunFn = _run,
) -> None:
    notes = (
        f"Candidate Ollama runtime {version} mirrored from ollama/ollama. "
        "Not stable until the promote PR is merged. GGUF is not included."
    )
    present = _release_asset_sizes(tag, run)
    if present is None:
        argv = [
            "gh",
            "release",
            "create",
            tag,
            "--prerelease",
            "--title",
            tag,
            "--notes",
            notes,
        ]
        argv.extend(str(path) for path in files)
        run(argv)
        return
    missing = [
        path
        for path in files
        if path.name not in present or present[path.name] != path.stat().st_size
    ]
    if not missing:
        print(f"[ollama-binaries] release {tag} already has assets", file=sys.stderr)
        return
    argv = ["gh", "release", "upload", tag, "--clobber"]
    argv.extend(str(path) for path in missing)
    run(argv)


def open_promote_pr(
    tag: str,
    version: str,
    repo_root: Path,
    *,
    run: RunFn = _run,
) -> None:
    branch = f"ci/promote-{tag}"
    run(["git", "checkout", "-B", branch], cwd=repo_root)
    run(
        ["git", "add", str(LOCK_RELATIVE), str(LEGACY_RELATIVE)],
        cwd=repo_root,
    )
    staged = run(["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False)
    if staged.returncode == 0:
        print("[ollama-binaries] no lock file changes", file=sys.stderr)
        return
    message = (
        f"chore: promote Ollama {version} candidate to stable\n\n"
        f"Mirror portable archives to GitHub Release {tag}. "
        "Previous pin is in runtime.legacy.lock.json."
    )
    run(["git", "commit", "-m", message], cwd=repo_root)
    push = run(["git", "push", "-u", "origin", "HEAD"], cwd=repo_root, check=False)
    if push.returncode != 0:
        run(
            ["git", "push", "-u", "origin", "HEAD", "--force-with-lease"],
            cwd=repo_root,
        )
    listed = run(
        ["gh", "pr", "list", "--head", branch, "--json", "url"],
        cwd=repo_root,
    )
    existing = json.loads(listed.stdout or "[]")
    if existing:
        print(
            f"[ollama-binaries] updated PR {existing[0].get('url')}",
            file=sys.stderr,
        )
        return
    body = (
        f"## Summary\n"
        f"- Mirror ollama/ollama `{version}` portable archives to GitHub Release `{tag}`.\n"
        f"- Point `runtime.lock.json` at this repository's release URLs (`channel: stable`).\n"
        f"- Snapshot the previous pin in `runtime.legacy.lock.json`.\n\n"
        f"Merge after binary review. Does not download GGUF."
    )
    run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            f"chore: promote Ollama {version} candidate to stable",
            "--body",
            body,
        ],
        cwd=repo_root,
    )


def apply(
    *,
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    candidate: dict[str, Any] | None = None,
    download_fn: Callable[..., Path] = download_hashed,
    run: RunFn = _run,
) -> None:
    _, repo = _require_apply_tools()
    if candidate is None:
        candidate = select_candidate_assets(fetch_latest(), DEFAULT_PLATFORMS)
    tag = candidate_release_tag(str(candidate.get("upstream_tag") or ""))
    proposed = stable_lock_from_candidate(candidate, repo, tag)
    lock_path = repo_root / LOCK_RELATIVE
    if lock_path.is_file() and locks_match(_load_json(lock_path), proposed):
        print(
            f"[ollama-binaries] already current ({tag}); nothing to do",
            file=sys.stderr,
        )
        return

    scratch = Path(work_dir) if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="ollama-binaries-")
    )
    scratch.mkdir(parents=True, exist_ok=True)
    print(
        f"[ollama-binaries] downloading {len(files_to_mirror(candidate))} files into {scratch}",
        file=sys.stderr,
    )
    files = download_mirror_files(candidate, scratch, download_fn=download_fn)
    proposed = stable_lock_from_candidate(candidate, repo, tag)
    if lock_path.is_file() and locks_match(_load_json(lock_path), proposed):
        print(
            f"[ollama-binaries] already current ({tag}); nothing to do",
            file=sys.stderr,
        )
        return

    publish_prerelease(
        tag,
        str(candidate.get("ollama_version") or ""),
        files,
        run=run,
    )
    legacy_path = repo_root / LEGACY_RELATIVE
    if lock_path.is_file():
        shutil.copyfile(lock_path, legacy_path)
    _write_json(lock_path, proposed)
    open_promote_pr(
        tag,
        str(candidate.get("ollama_version") or ""),
        repo_root,
        run=run,
    )
    print(
        f"[ollama-binaries] proposed stable {tag} ({len(files)} assets)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose or mirror an Ollama candidate lock from GitHub latest."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Download hashed archives, publish a prerelease, and open a promote PR.",
    )
    args = parser.parse_args()
    release = fetch_latest()
    candidate = select_candidate_assets(release, DEFAULT_PLATFORMS)
    print(json.dumps(candidate, indent=2))
    if not args.apply:
        return
    apply(candidate=candidate)


if __name__ == "__main__":
    main()
