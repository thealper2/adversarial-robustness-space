"""Runtime configuration, read once from environment variables set by the Space owner."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Local artifact folder (models, sample banks, precomputed results). Used when present,
# e.g. during local development or when artifacts are baked into the image.
LOCAL_ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))

# Hugging Face Hub model repository that holds the artifacts, e.g. "user/art-robustness-artifacts".
ARTIFACTS_REPO = os.environ.get("ARTIFACTS_REPO", "").strip() or None
# Optional commit hash or tag so the Space always loads a reviewed artifact version.
ARTIFACTS_REVISION = os.environ.get("ARTIFACTS_REVISION", "").strip() or None

# "cpu" on the Space. Local training / precomputation may use "cuda".
DEVICE = os.environ.get("ROBUSTLAB_DEVICE", "cpu")
NUM_THREADS = int(os.environ.get("ROBUSTLAB_THREADS", "2"))

# Live runs are serialized; a request waits at most this long for the running job to finish.
LIVE_RUN_WAIT_SECONDS = int(os.environ.get("ROBUSTLAB_LIVE_WAIT", "5"))
