"""Resolve artifact files from the local folder or the configured Hugging Face Hub repository.

Only relative paths under a fixed set of top-level folders are accepted, so an artifact name
can never escape the artifact root or point at an arbitrary file.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath

from . import config

ALLOWED_ROOTS = ("models", "data", "results", "extras")
_SAFE_PART = re.compile(r"^[A-Za-z0-9_.\-]+$")


class ArtifactMissing(FileNotFoundError):
    """Raised when an artifact is neither available locally nor in the Hub repository."""


def _validate(rel: str) -> str:
    path = PurePosixPath(rel)
    parts = path.parts
    if path.is_absolute() or len(parts) < 2 or parts[0] not in ALLOWED_ROOTS:
        raise ValueError(f"Invalid artifact path: {rel!r}")
    if any(p in ("", ".", "..") or not _SAFE_PART.match(p) for p in parts):
        raise ValueError(f"Invalid artifact path: {rel!r}")
    return str(path)


@lru_cache(maxsize=1)
def _remote_files() -> frozenset[str]:
    if not config.ARTIFACTS_REPO:
        return frozenset()
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        config.ARTIFACTS_REPO, revision=config.ARTIFACTS_REVISION
    )
    return frozenset(files)


def exists(rel: str) -> bool:
    rel = _validate(rel)
    if (config.LOCAL_ARTIFACTS / rel).is_file():
        return True
    try:
        return rel in _remote_files()
    except Exception:  # Hub unreachable: behave as if the artifact is missing.
        return False


def path(rel: str) -> Path:
    """Return a local filesystem path for an artifact, downloading it from the Hub if needed."""
    rel = _validate(rel)
    local = config.LOCAL_ARTIFACTS / rel
    if local.is_file():
        return local
    if not config.ARTIFACTS_REPO:
        raise ArtifactMissing(
            f"{rel} not found in {config.LOCAL_ARTIFACTS} and ARTIFACTS_REPO is not set"
        )
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                config.ARTIFACTS_REPO, rel, revision=config.ARTIFACTS_REVISION
            )
        )
    except Exception as exc:
        raise ArtifactMissing(
            f"{rel} could not be fetched from {config.ARTIFACTS_REPO}: {exc}"
        ) from exc


def read_json(rel: str) -> dict:
    with open(path(rel), encoding="utf-8") as fh:
        return json.load(fh)
