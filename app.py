"""ART Robustness Lab — Streamlit entry point for the Hugging Face Space.

The app never accepts files, text or models from visitors: the only inputs are selections
from fixed lists and bounded sliders, which are re-validated by ``Param.coerce`` before use.
"""

from __future__ import annotations

import logging
import threading
from urllib.parse import quote

import streamlit as st

from robustlab import config
from robustlab import experiments  # noqa: F401  (registers all experiments)
from robustlab.artifacts import ArtifactMissing
from robustlab.core import (
    CATEGORIES,
    COST_HINT,
    REGISTRY,
    Experiment,
    by_category,
    execute,
    has_precomputed,
    load_precomputed,
)
from robustlab.data import DATASETS
from robustlab.models import MODELS, meta
from robustlab.ui import render

log = logging.getLogger("robustlab")
NOTEBOOK_REPO = "https://github.com/thealper2/ai-security-notebooks/blob/main/"
OVERVIEW = "overview"

st.set_page_config(page_title="ART Robustness Lab", page_icon="🛡️", layout="wide")


# --- Cached resources ----------------------------------------------------------------------------


@st.cache_resource
def run_lock() -> threading.Lock:
    """One live CPU job at a time for the whole Space."""
    return threading.Lock()


@st.cache_data(show_spinner=False, max_entries=128)
def cached_run(exp_id: str, params: tuple):
    return execute(REGISTRY[exp_id], dict(params))


@st.cache_data(show_spinner=False, max_entries=128)
def cached_precomputed(exp_id: str):
    return load_precomputed(REGISTRY[exp_id])


@st.cache_data(show_spinner=False, ttl=600)
def precomputed_available(exp_id: str) -> bool:
    return has_precomputed(REGISTRY[exp_id])


# --- Navigation ----------------------------------------------------------------------------------


def sidebar() -> str:
    groups = by_category()
    requested = st.query_params.get("exp", OVERVIEW)
    current = requested if requested in REGISTRY or requested == OVERVIEW else OVERVIEW

    st.sidebar.title("🛡️ ART Robustness Lab")
    st.sidebar.caption(
        "Adversarial Robustness Toolbox demos on pre-trained models — CPU inference only."
    )
    cat_keys = [OVERVIEW, *groups]
    current_cat = OVERVIEW if current == OVERVIEW else REGISTRY[current].category
    cat = st.sidebar.selectbox(
        "Category",
        cat_keys,
        index=cat_keys.index(current_cat),
        format_func=lambda c: (
            "Overview" if c == OVERVIEW else f"{CATEGORIES[c]} ({len(groups[c])})"
        ),
    )
    if cat == OVERVIEW:
        selected = OVERVIEW
    else:
        ids = [e.id for e in groups[cat]]
        selected = st.sidebar.radio(
            "Experiment",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=lambda i: REGISTRY[i].title,
        )
    if selected != requested:
        st.query_params["exp"] = selected

    st.sidebar.divider()
    source = (
        f"Hub · `{config.ARTIFACTS_REPO}`" if config.ARTIFACTS_REPO else "local folder"
    )
    if config.ARTIFACTS_REVISION:
        source += f" @ `{config.ARTIFACTS_REVISION[:10]}`"
    st.sidebar.caption(
        f"Device: **{config.DEVICE}** · threads: {config.NUM_THREADS}  \nArtifacts: {source}"
    )
    return selected


# --- Pages ---------------------------------------------------------------------------------------


def overview() -> None:
    st.title("Adversarial Robustness Lab")
    st.markdown(
        "Interactive, read-only demos of the **Adversarial Robustness Toolbox (ART)** experiments from "
        f"[thealper2/ai-security-notebooks]({NOTEBOOK_REPO.replace('/blob/main/', '')}), rewritten as clean PyTorch / "
        "scikit-learn modules.\n\n"
        "* All models were **trained offline** and are only loaded here (pickle-free *safetensors* / *skops*).\n"
        "* Attacks and defences run **live on CPU** against a fixed bank of dataset samples, or show "
        "**precomputed results** produced by the very same code when an attack is too slow for a CPU Space.\n"
        "* Visitors cannot upload anything: every control is a fixed list or a bounded slider. No external system is "
        "contacted except the Hugging Face Hub repository holding the artifacts."
    )
    groups = by_category()
    cols = st.columns(4)
    cols[0].metric("Experiments", len(REGISTRY))
    cols[1].metric("Attacks", sum(e.kind == "attack" for e in REGISTRY.values()))
    cols[2].metric("Defences", sum(e.kind == "defence" for e in REGISTRY.values()))
    cols[3].metric("Models", len(MODELS))

    rows = []
    for cat, exps in groups.items():
        for e in exps:
            rows.append(
                {
                    "Category": CATEGORIES[cat],
                    "Experiment": e.title,
                    "ART": ", ".join(e.art),
                    "Dataset": DATASETS[e.dataset].title,
                    "Live": "yes" if e.live else "precomputed only",
                    "Cost": e.cost,
                    "Link": f"?exp={e.id}",
                }
            )
    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Link": st.column_config.LinkColumn("Open", display_text="open")
        },
    )


def experiment_header(exp: Experiment) -> None:
    st.title(exp.title)
    st.caption(
        " · ".join(
            [
                CATEGORIES[exp.category],
                DATASETS[exp.dataset].title,
                *[f"`{a}`" for a in exp.art],
            ]
        )
    )
    st.markdown(exp.summary)
    with st.expander("Details, source notebook and models"):
        if exp.details:
            st.info(exp.details)
        st.markdown(
            f"**Notebook:** [{exp.notebook}]({NOTEBOOK_REPO}{quote(exp.notebook)})"
        )
        for mid in exp.models:
            s = MODELS[mid]
            info = meta(mid)
            acc = info.get("metrics", {})
            acc_txt = " · ".join(
                f"{k}: {v:.1%}" if isinstance(v, float) and v <= 1 else f"{k}: {v}"
                for k, v in acc.items()
            )
            st.markdown(
                f"- **{s.title}** (`{mid}`) — {s.description or s.kind}"
                + (f"  \n  {acc_txt}" if acc_txt else "")
            )


def params_form(exp: Experiment) -> tuple[dict, bool]:
    values = exp.defaults()
    if not exp.params:
        clicked = (
            st.button("Run live on CPU", disabled=not exp.live, type="primary")
            if exp.live
            else False
        )
        return values, clicked
    with st.form(f"form-{exp.id}"):
        cols = st.columns(min(len(exp.params), 3))
        for i, p in enumerate(exp.params):
            with cols[i % len(cols)]:
                if p.kind == "choice":
                    labels = dict(p.options)
                    keys = list(labels)
                    values[p.key] = st.selectbox(
                        p.label,
                        keys,
                        index=keys.index(p.default),
                        format_func=labels.get,
                        help=p.help or None,
                        disabled=not exp.live,
                    )
                elif p.kind == "int":
                    values[p.key] = st.slider(
                        p.label,
                        int(p.min),
                        int(p.max),
                        int(p.default),
                        int(p.step or 1),
                        help=p.help or None,
                        disabled=not exp.live,
                    )
                else:
                    values[p.key] = st.slider(
                        p.label,
                        float(p.min),
                        float(p.max),
                        float(p.default),
                        float(p.step),
                        help=p.help or None,
                        disabled=not exp.live,
                        format="%.4g",
                    )
        clicked = st.form_submit_button(
            "Run live on CPU", type="primary", disabled=not exp.live
        )
    return exp.coerce(values), clicked


def live_run(exp: Experiment, params: dict):
    lock = run_lock()
    if not lock.acquire(timeout=config.LIVE_RUN_WAIT_SECONDS):
        st.warning(
            "Another live run is in progress on this CPU Space. Please try again in a moment."
        )
        return None
    try:
        with st.spinner(
            f"Running {exp.title} on CPU (typically {COST_HINT[exp.cost]})…"
        ):
            return cached_run(exp.id, tuple(sorted(params.items())))
    finally:
        lock.release()


def experiment_page(exp: Experiment) -> None:
    experiment_header(exp)
    if not exp.live:
        st.info(
            "This attack needs minutes of CPU time, so the Space shows results precomputed with the same code "
            "and the default parameters."
        )
    params, clicked = params_form(exp)

    live_results = st.session_state.setdefault("live", {})
    try:
        if clicked:
            result = live_run(exp, params)
            if result is not None:
                live_results[exp.id] = result
        result = live_results.get(exp.id)
        origin = "live"
        if result is None and precomputed_available(exp.id):
            result, origin = cached_precomputed(exp.id), "precomputed"
    except ArtifactMissing as exc:
        st.error(f"Required artifact is missing: {exc}")
        return
    except Exception as exc:  # show a concise error instead of a stack trace
        log.exception("Run failed for %s", exp.id)
        st.error(f"The run failed: {type(exc).__name__}: {exc}")
        return

    if result is None:
        st.warning(
            "No precomputed result yet. "
            + ("Press **Run live on CPU**." if exp.live else "")
        )
        return
    m = result.meta
    shown = ", ".join(f"{k}={v}" for k, v in m.get("params", {}).items()) or "defaults"
    label = "Precomputed" if origin == "precomputed" else "Live result"
    st.success(
        f"{label} · {shown} · {m.get('seconds', '?')} s on {m.get('device', '?')} · ART {m.get('art_version', '?')} · {m.get('created', '')}"
    )
    render(result)


def main() -> None:
    selected = sidebar()
    if selected == OVERVIEW:
        overview()
    else:
        experiment_page(REGISTRY[selected])


main()
