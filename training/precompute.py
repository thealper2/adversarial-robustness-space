"""Run every experiment with its default parameters and store the results as artifacts.

    python -m training.precompute                  # all missing results
    python -m training.precompute fgm boundary     # selected experiments
    python -m training.precompute --force --device cpu

``--device auto`` (default) runs Space-live experiments on the CPU with the Space's thread count, so
the stored runtime is representative, and precomputed-only experiments on the GPU if available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    import torch

    from robustlab import config, experiments  # noqa: F401
    from robustlab import results as results_io
    from robustlab.core import REGISTRY, execute

    from .common import ARTIFACTS, log

    out = ARTIFACTS / "results"
    out.mkdir(parents=True, exist_ok=True)
    timings_file = out / "_timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else {}
    ids = args.ids or list(REGISTRY)
    failed = []
    for exp_id in ids:
        exp = REGISTRY[exp_id]
        if not args.force and (out / f"{exp_id}.json").exists():
            log(f"skip {exp_id}")
            continue
        device = args.device
        if device == "auto":
            device = "cpu" if exp.live or not torch.cuda.is_available() else "cuda"
        config.DEVICE = device
        _reset_caches()
        log(f"=== {exp_id} on {device} ===")
        start = time.perf_counter()
        try:
            result = execute(exp)
        except Exception:
            traceback.print_exc()
            failed.append(exp_id)
            continue
        results_io.save(result, out / f"{exp_id}.json", out / f"{exp_id}.npz")
        timings[exp_id] = {
            "seconds": round(time.perf_counter() - start, 1),
            "device": device,
            "cost": exp.cost,
            "live": exp.live,
        }
        timings_file.write_text(json.dumps(timings, indent=1))
        log(
            f"{exp_id}: {timings[exp_id]['seconds']} s · "
            + " · ".join(f"{m.label}={m.value}" for m in result.metrics[:4])
        )
    log("failed: " + (", ".join(failed) if failed else "none"))


def _reset_caches() -> None:
    """Models are cached per device; clear them when switching between CPU and GPU."""
    from robustlab.models import zoo

    for fn in (zoo.torch_model, zoo.classifier):
        fn.cache_clear()


if __name__ == "__main__":
    os.environ.setdefault("ROBUSTLAB_THREADS", "2")
    main()
