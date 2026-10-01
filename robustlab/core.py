"""Experiment registry, bounded parameters and execution.

Every user-facing control maps to a ``Param`` with a fixed domain. Values coming from the UI
are always passed through ``Param.coerce`` so a run can only ever use whitelisted options or
clamped numbers, never free-form input.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import numpy as np

from . import artifacts, config
from .results import Result
from . import results as results_io

Category = Literal[
    "evasion_white",
    "evasion_black",
    "universal",
    "evasion_classic",
    "defence_pre",
    "defence_train",
    "certification",
    "poisoning",
    "poison_detection",
    "inference",
    "extraction",
]

CATEGORIES: dict[str, str] = {
    "evasion_white": "Evasion · white-box",
    "evasion_black": "Evasion · black-box",
    "universal": "Universal & physical",
    "evasion_classic": "Evasion · classic ML",
    "defence_pre": "Defence · pre-processing & augmentation",
    "defence_train": "Defence · adversarial training",
    "certification": "Defence · certification",
    "poisoning": "Poisoning & backdoors",
    "poison_detection": "Defence · poison detection",
    "inference": "Privacy · inference attacks",
    "extraction": "Model extraction",
}


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: Literal["int", "float", "choice"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[tuple[Any, str], ...] = ()
    help: str = ""

    @classmethod
    def int(cls, key, label, default, lo, hi, step=1, help=""):
        return cls(key, label, "int", default, lo, hi, step, help=help)

    @classmethod
    def float(cls, key, label, default, lo, hi, step, help=""):
        return cls(key, label, "float", default, lo, hi, step, help=help)

    @classmethod
    def choice(cls, key, label, default, options, help=""):
        opts = tuple(o if isinstance(o, tuple) else (o, str(o)) for o in options)
        return cls(key, label, "choice", default, options=opts, help=help)

    def coerce(self, value: Any) -> Any:
        if self.kind == "choice":
            allowed = [v for v, _ in self.options]
            return value if value in allowed else self.default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        if not np.isfinite(number):
            return self.default
        number = min(max(number, self.min), self.max)
        if self.step:
            number = self.min + round((number - self.min) / self.step) * self.step
            number = min(max(number, self.min), self.max)
        return int(round(number)) if self.kind == "int" else round(number, 6)


Cost = Literal["fast", "medium", "slow"]
COST_HINT = {
    "fast": "a few seconds",
    "medium": "up to ~1 minute",
    "slow": "minutes – precomputed only",
}


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    category: Category
    dataset: str
    models: tuple[str, ...]
    art: tuple[str, ...]
    notebook: str
    summary: str
    run: Callable[[dict], Result]
    params: tuple[Param, ...] = ()
    cost: Cost = "fast"
    live: bool = True
    details: str = ""
    kind: Literal["attack", "defence"] = "attack"
    extra_artifacts: tuple[str, ...] = field(default_factory=tuple)

    def defaults(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.params}

    def coerce(self, values: dict[str, Any] | None) -> dict[str, Any]:
        values = values or {}
        return {p.key: p.coerce(values.get(p.key, p.default)) for p in self.params}


REGISTRY: dict[str, Experiment] = {}


def experiment(
    **meta,
) -> Callable[[Callable[[dict], Result]], Callable[[dict], Result]]:
    """Decorator registering ``run(params) -> Result`` as an experiment."""

    def decorator(fn: Callable[[dict], Result]):
        exp = Experiment(run=fn, **meta)
        if exp.id in REGISTRY:
            raise ValueError(f"Duplicate experiment id {exp.id}")
        REGISTRY[exp.id] = exp
        return fn

    return decorator


def by_category() -> dict[str, list[Experiment]]:
    grouped: dict[str, list[Experiment]] = {c: [] for c in CATEGORIES}
    for exp in REGISTRY.values():
        grouped[exp.category].append(exp)
    return {c: v for c, v in grouped.items() if v}


def seed_everything(seed: int = 0) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def execute(
    exp: Experiment, params: dict[str, Any] | None = None, seed: int = 0
) -> Result:
    """Run an experiment with coerced parameters and attach provenance metadata."""
    import art
    import torch

    params = exp.coerce(params)
    torch.set_num_threads(config.NUM_THREADS)
    seed_everything(seed)
    start = time.perf_counter()
    result = exp.run(params)
    result.meta.update(
        experiment=exp.id,
        params=params,
        seconds=round(time.perf_counter() - start, 2),
        device=config.DEVICE,
        art_version=art.__version__,
        torch_version=torch.__version__.split("+")[0],
        created=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    return result


# --- Precomputed results -----------------------------------------------------------------------


def precomputed_paths(exp_id: str) -> tuple[str, str]:
    return f"results/{exp_id}.json", f"results/{exp_id}.npz"


def has_precomputed(exp: Experiment) -> bool:
    return artifacts.exists(precomputed_paths(exp.id)[0])


def load_precomputed(exp: Experiment) -> Result | None:
    json_rel, npz_rel = precomputed_paths(exp.id)
    if not artifacts.exists(json_rel):
        return None
    return results_io.load(artifacts.path(json_rel), artifacts.path(npz_rel))
