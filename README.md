# ollama-binaries

Hashed portable Ollama archives mirrored from [`ollama/ollama`](https://github.com/ollama/ollama). Not a library. Not upstream.

Consumers (for example pdf2chemicals) vendor `runtime.lock.json` and download the URLs in that pin. This repository owns the mirror cycle: discover latest upstream assets, hash them, publish a GitHub Release, open a promote PR **here**.

GGUF / model weights are never stored here.

## Layout

```
mirror.py                 discover / download / publish / promote PR
runtime.lock.json         canonical stable pin (URLs + sha256 + size)
runtime.legacy.lock.json  previous pin (written by --apply)
tests/                    CPU unit tests (no binary download in CI)
.github/workflows/
  ollama-candidate.yml    cron + workflow_dispatch → python mirror.py --apply
  test.yml                pytest
```

Release tags match upstream (`v0.33.2`, `v0.34.0`, …).

## Commands

```
python mirror.py            # print candidate JSON; no writes
python mirror.py --apply    # download, prerelease, open promote PR
pytest
```

`--apply` needs `GITHUB_TOKEN` or `GH_TOKEN`, `GITHUB_REPOSITORY=owner/name`, `gh`, and `git`.

## Promote

1. Workflow publishes a **prerelease** with the portable archives.
2. PR updates `runtime.lock.json` so `url` fields point at this repo's Release.
3. Previous pin is copied to `runtime.legacy.lock.json`.
4. Human merges the PR in **this** repository.

Downstream projects copy the merged lock when they choose to bump. That copy is not part of this CI.
