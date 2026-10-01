"""Streamlit rendering of ``Result`` objects. All content comes from experiment code and artifacts."""

from __future__ import annotations

import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from .results import (
    BarChart,
    Detections,
    ImageGrid,
    ImageTable,
    LineChart,
    Markdown,
    Metric,
    Result,
    Table,
)

STATUS_COLOR = {"ok": "green", "bad": "red"}
PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2"]
BOX_COLORS = {1: (217, 51, 51), 2: (51, 178, 77), 3: (64, 102, 230)}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-!|~:$])")


def _escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _format(value, fmt: str) -> str:
    if isinstance(value, str) or fmt == "text":
        return str(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if fmt == "pct":
        return f"{value:.1%}"
    if fmt == "int":
        return f"{int(value):,}"
    return f"{value:.3f}" if abs(value) < 100 else f"{value:,.1f}"


def _delta(m: Metric) -> str | None:
    if m.delta is None:
        return None
    return f"{m.delta * 100:+.1f} pp" if m.fmt == "pct" else f"{m.delta:+.3f}"


def _upscale(img: np.ndarray, min_side: int = 112) -> Image.Image:
    arr = np.clip(np.asarray(img, dtype=np.float32), 0, 1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    pil = Image.fromarray((arr * 255).round().astype(np.uint8))
    factor = max(1, int(np.ceil(min_side / min(pil.size))))
    return (
        pil.resize((pil.width * factor, pil.height * factor), Image.NEAREST)
        if factor > 1
        else pil
    )


def _caption(text: str, status: str | None) -> None:
    lines = [_escape(t) for t in str(text).split("\n")]
    body = "  \n".join(lines)
    color = STATUS_COLOR.get(status or "")
    st.caption(f":{color}[{body}]" if color else body)


def metrics(items: list[Metric]) -> None:
    if not items:
        return
    per_row = 4
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, m in zip(cols, items[start : start + per_row]):
            col.metric(m.label, _format(m.value, m.fmt), _delta(m), help=m.help)


def image_grid(b: ImageGrid) -> None:
    st.markdown(f"**{_escape(b.title)}**")
    cols_n = max(1, min(b.columns, len(b.images)))
    for start in range(0, len(b.images), cols_n):
        cols = st.columns(cols_n)
        for j, col in enumerate(cols):
            i = start + j
            if i >= len(b.images):
                break
            with col:
                if cols_n >= 5:
                    st.image(_upscale(b.images[i]), use_container_width=True)
                else:
                    st.image(
                        _upscale(b.images[i]),
                        width=224 if b.images[i].shape[0] > 64 else 168,
                    )
                _caption(
                    b.captions[i] if i < len(b.captions) else "",
                    b.status[i] if b.status else None,
                )


def image_table(b: ImageTable) -> None:
    st.markdown(f"**{_escape(b.title)}**")
    header = st.columns(len(b.headers))
    for col, h in zip(header, b.headers):
        col.markdown(f"*{_escape(h)}*")
    for r in range(len(b.images)):
        cols = st.columns(len(b.headers))
        for c, col in enumerate(cols):
            with col:
                img = b.images[r][c]
                st.image(_upscale(img), width=224 if img.shape[0] > 64 else 150)
                _caption(b.captions[r][c], b.status[r][c] if b.status else None)


def _color(series: list[str]) -> alt.Color:
    """Series keep their declared order and colour (first = baseline blue, second = red …)."""
    scale = alt.Scale(
        domain=series,
        range=PALETTE[: len(series)] if len(series) <= len(PALETTE) else None,
    )
    return alt.Color(
        "series:N",
        title=None,
        sort=series,
        scale=scale,
        legend=alt.Legend(orient="bottom"),
    )


def bar_chart(b: BarChart) -> None:
    rows = [
        {"category": cat, "series": name, "value": vals[i]}
        for name, vals in b.series.items()
        for i, cat in enumerate(b.categories)
    ]
    df = pd.DataFrame(rows)
    fmt = ".0%" if b.percent else ",.2f"
    chart = (
        alt.Chart(df, title=b.title)
        .mark_bar()
        .encode(
            x=alt.X(
                "category:N",
                sort=list(b.categories),
                title=None,
                axis=alt.Axis(labelAngle=-30 if len(b.categories) > 5 else 0),
            ),
            xOffset=alt.XOffset("series:N", sort=list(b.series)),
            y=alt.Y("value:Q", title=b.y_label or None, axis=alt.Axis(format=fmt)),
            color=_color(list(b.series)),
            tooltip=["category", "series", alt.Tooltip("value:Q", format=fmt)],
        )
    )
    st.altair_chart(
        chart.properties(height=320, width="container"), use_container_width=True
    )


def line_chart(b: LineChart) -> None:
    rows = [
        {"x": x, "series": name, "value": vals[i]}
        for name, vals in b.series.items()
        for i, x in enumerate(b.x)
    ]
    fmt = ".0%" if b.percent else ",.2f"
    chart = (
        alt.Chart(pd.DataFrame(rows), title=b.title)
        .mark_line(point=True)
        .encode(
            x=alt.X("x:Q", title=b.x_label or None),
            y=alt.Y("value:Q", title=b.y_label or None, axis=alt.Axis(format=fmt)),
            color=_color(list(b.series)),
            tooltip=[
                alt.Tooltip("x:Q", title=b.x_label or "x"),
                "series",
                alt.Tooltip("value:Q", format=fmt),
            ],
        )
    )
    st.altair_chart(
        chart.properties(height=320, width="container"), use_container_width=True
    )


def table(b: Table) -> None:
    st.markdown(f"**{_escape(b.title)}**")
    st.dataframe(
        pd.DataFrame(b.rows, columns=b.columns),
        hide_index=True,
        use_container_width=True,
    )


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def detections(b: Detections) -> None:
    st.markdown(f"**{_escape(b.title)}**")
    font = _font(14)
    cols_n = max(1, min(b.columns, len(b.images)))
    for start in range(0, len(b.images), cols_n):
        cols = st.columns(cols_n)
        for j, col in enumerate(cols):
            i = start + j
            if i >= len(b.images):
                break
            scale = 3
            pil = _upscale(b.images[i], b.images[i].shape[0] * scale).convert("RGB")
            draw = ImageDraw.Draw(pil)
            for k, (box, lab) in enumerate(zip(b.boxes[i], b.labels[i])):
                x1, y1, x2, y2 = (v * scale for v in box)
                color = BOX_COLORS.get(int(lab), (0, 0, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                text = (
                    b.class_names[int(lab)]
                    if int(lab) < len(b.class_names)
                    else str(lab)
                )
                if b.scores:
                    text += f" {b.scores[i][k]:.2f}"
                tw = draw.textlength(text, font=font)
                ty = max(0, y1 - 18)
                draw.rectangle([x1, ty, x1 + tw + 6, ty + 18], fill=color)
                draw.text((x1 + 3, ty + 1), text, fill=(255, 255, 255), font=font)
            with col:
                st.image(pil, use_container_width=True)
                _caption(b.captions[i], None)


RENDERERS = {
    Markdown: lambda b: st.markdown(b.text),
    ImageGrid: image_grid,
    ImageTable: image_table,
    BarChart: bar_chart,
    LineChart: line_chart,
    Table: table,
    Detections: detections,
}


def render(result: Result) -> None:
    metrics(result.metrics)
    for block in result.blocks:
        if block is None:
            continue
        st.divider()
        RENDERERS[type(block)](block)
