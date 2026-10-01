"""Upload the artifact folder to a Hugging Face Hub model repository.

    hf auth login                                   # once, with a write token
    python -m training.upload --repo <user>/art-robustness-artifacts [--private]

The printed commit hash can be set as the Space variable ``ARTIFACTS_REVISION`` so the Space
always loads exactly the reviewed artifact version.
"""

from __future__ import annotations

import argparse
import json

from .common import ARTIFACTS

PATTERNS = ["models/*", "data/*", "results/*", "extras/*", "README.md"]


def model_card(repo: str) -> str:
    rows = []
    for meta_file in sorted((ARTIFACTS / "models").glob("*.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        metrics = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in meta.get("metrics", {}).items()
        )
        rows.append(f"| `{meta['id']}` | {metrics} | {meta.get('created', '')} |")
    return "\n".join(
        [
            "---",
            "tags: [adversarial-robustness-toolbox, adversarial-machine-learning, security]",
            "---",
            "",
            f"# {repo.split('/')[-1]}",
            "",
            "Artifacts for the *ART Robustness Lab* Space: pickle-free model weights (safetensors / skops / npz), fixed "
            "sample banks, offline-crafted poisons and precomputed experiment results.",
            "",
            "Several models are **intentionally backdoored or poisoned** for demonstration purposes. Do not use them in production.",
            "",
            "| model | metrics | created |",
            "|---|---|---|",
            *rows,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    (ARTIFACTS / "README.md").write_text(model_card(args.repo), encoding="utf-8")
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(ARTIFACTS),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=PATTERNS,
        commit_message="Update ART Robustness Lab artifacts",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo}")
    print(f"Pin this version in the Space: ARTIFACTS_REVISION={info.oid}")


if __name__ == "__main__":
    main()
