"""Framework-agnostic experiment results.

A ``Result`` is plain data (numbers, strings, numpy arrays) so that the same object can be
rendered live by Streamlit or stored as a precomputed ``results/<id>.json`` + ``.npz`` pair.
The npz is read with ``allow_pickle=False``; the JSON only references arrays by key.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Metric:
    label: str
    value: float | str
    fmt: str = "pct"  # "pct" | "float" | "int" | "text"
    delta: float | None = None
    help: str | None = None


@dataclass
class Markdown:
    text: str


@dataclass
class ImageGrid:
    """A flat gallery. ``status`` colours captions: "ok" (green), "bad" (red) or None."""

    title: str
    images: np.ndarray  # (N, H, W, C) float32 in [0, 1]
    captions: list[str]
    status: list[str | None] | None = None
    columns: int = 8


@dataclass
class ImageTable:
    """Rows of aligned images, e.g. original / adversarial / perturbation per example."""

    title: str
    headers: list[str]
    images: np.ndarray  # (R, C, H, W, ch) float32 in [0, 1]
    captions: list[list[str]]
    status: list[list[str | None]] | None = None


@dataclass
class BarChart:
    title: str
    categories: list[str]
    series: dict[str, list[float]]
    y_label: str = ""
    percent: bool = True


@dataclass
class LineChart:
    title: str
    x: list[float]
    series: dict[str, list[float]]
    x_label: str = ""
    y_label: str = ""
    percent: bool = True


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[list[Any]]


@dataclass
class Detections:
    """Images with predicted/annotated boxes (object detection)."""

    title: str
    images: np.ndarray  # (N, H, W, C)
    boxes: list[list[list[float]]]
    labels: list[list[int]]
    scores: list[list[float]] | None
    class_names: list[str]
    captions: list[str]
    columns: int = 3


BLOCK_TYPES = {
    cls.__name__: cls
    for cls in (Markdown, ImageGrid, ImageTable, BarChart, LineChart, Table, Detections)
}
Block = Markdown | ImageGrid | ImageTable | BarChart | LineChart | Table | Detections


@dataclass
class Result:
    metrics: list[Metric] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, *blocks: Block | None) -> "Result":
        self.blocks.extend(b for b in blocks if b is not None)
        return self

    def metric(
        self, label: str, value, fmt: str = "pct", delta=None, help=None
    ) -> "Result":
        self.metrics.append(Metric(label, _plain(value), fmt, _plain(delta), help))
        return self


def _plain(value):
    """Convert numpy scalars/containers to JSON-friendly Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


# --- Serialisation -----------------------------------------------------------------------------


def save(result: Result, json_path: Path, npz_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    blocks = []
    for i, block in enumerate(result.blocks):
        entry: dict[str, Any] = {"type": type(block).__name__}
        for f in fields(block):
            value = getattr(block, f.name)
            if isinstance(value, np.ndarray):
                key = f"b{i}_{f.name}"
                arrays[key] = (
                    value.astype(np.float16) if value.dtype.kind == "f" else value
                )
                entry[f.name] = {"__array__": key}
            else:
                entry[f.name] = _plain(value)
        blocks.append(entry)
    doc = {
        "schema": SCHEMA_VERSION,
        "metrics": [asdict(m) for m in result.metrics],
        "blocks": blocks,
        "meta": _plain(result.meta),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)


def load(json_path: Path, npz_path: Path) -> Result:
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unsupported result schema")
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    blocks = []
    for entry in doc["blocks"]:
        cls = BLOCK_TYPES[entry.pop("type")]  # KeyError for unknown types
        kwargs = {}
        for f in fields(cls):
            if f.name not in entry:
                continue
            value = entry[f.name]
            if isinstance(value, dict) and set(value) == {"__array__"}:
                value = arrays[value["__array__"]].astype(np.float32)
            kwargs[f.name] = value
        blocks.append(cls(**kwargs))
    metrics = [Metric(**m) for m in doc["metrics"]]
    return Result(metrics=metrics, blocks=blocks, meta=doc.get("meta", {}))
