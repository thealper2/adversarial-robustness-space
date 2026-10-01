"""Run experiments with their default parameters and report failures (no files are written).

python -m training.smoke            # every live experiment (what the Space can run on demand)
python -m training.smoke fgm strip  # selected experiments
python -m training.smoke --all      # also the precomputed-only ones (slow)
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ids", nargs="*")
    parser.add_argument(
        "--all", action="store_true", help="include precomputed-only experiments"
    )
    args = parser.parse_args()

    from robustlab import experiments  # noqa: F401
    from robustlab.core import REGISTRY, execute

    ids = args.ids or [e.id for e in REGISTRY.values() if e.live or args.all]
    failed, start = [], time.time()
    for exp_id in ids:
        t = time.time()
        try:
            result = execute(REGISTRY[exp_id])
            print(
                f"ok   {exp_id:34s} {time.time() - t:6.1f}s  blocks={len(result.blocks)}",
                flush=True,
            )
        except Exception as exc:
            failed.append(exp_id)
            print(f"FAIL {exp_id:34s} {type(exc).__name__}: {exc}", flush=True)
    print(
        f"{len(ids) - len(failed)}/{len(ids)} passed in {time.time() - start:.0f}s"
        + (f" · failed: {', '.join(failed)}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
