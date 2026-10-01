"""Shared helpers that turn predictions and adversarial examples into ``Result`` blocks."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import data
from .results import BarChart, ImageGrid, ImageTable, LineChart, Result, Table

# --- Predictions -------------------------------------------------------------------------------


def probabilities(clf, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Class probabilities, applying softmax when the estimator returns logits."""
    out = np.asarray(
        clf.predict(x, batch_size=batch_size) if _has_batch(clf) else clf.predict(x),
        dtype=np.float64,
    )
    if out.ndim == 1:
        out = np.stack([1 - out, out], axis=1)
    if out.min() < -1e-6 or not np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
        out = np.exp(out - out.max(axis=1, keepdims=True))
        out /= out.sum(axis=1, keepdims=True)
    return out


def _has_batch(clf) -> bool:
    import inspect

    try:
        return "batch_size" in inspect.signature(clf.predict).parameters
    except (TypeError, ValueError):
        return False


def predict(clf, x: np.ndarray) -> np.ndarray:
    return probabilities(clf, x).argmax(axis=1)


def accuracy(clf, x: np.ndarray, y: np.ndarray) -> float:
    return float((predict(clf, x) == np.asarray(y)).mean()) if len(x) else float("nan")


def weighted_scores(y_true, y_pred) -> dict[str, float]:
    """ACC / PREC / REC / F1 (weighted), the score set printed by the notebooks."""
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "PREC": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "REC": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# --- Images ------------------------------------------------------------------------------------


def to_display(dataset: str, x: np.ndarray) -> np.ndarray | None:
    """Convert model inputs to (N, H, W, C) float images in [0, 1], or None for tabular data."""
    x = np.asarray(x, dtype=np.float32)
    if dataset in ("mnist", "fashion_mnist") and x.ndim == 2:
        x = x.reshape(-1, 1, 28, 28)
    if dataset == "digits":
        return np.clip(x.reshape(-1, 8, 8, 1) / 16.0, 0, 1)
    if x.ndim != 4:
        return None
    if x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.transpose(0, 2, 3, 1)
    return np.clip(x, 0.0, 1.0)


def perturbation_view(delta: np.ndarray) -> np.ndarray:
    """Scale each perturbation to [0, 1] around mid-grey so tiny changes become visible."""
    delta = np.asarray(delta, dtype=np.float32)
    scale = np.abs(delta).reshape(len(delta), -1).max(axis=1)
    scale = np.where(scale > 0, scale, 1.0).reshape(-1, *([1] * (delta.ndim - 1)))
    return delta / (2 * scale) + 0.5


def label(dataset: str, idx: int) -> str:
    names = data.class_names(dataset)
    return names[idx] if 0 <= idx < len(names) else str(idx)


def status(ok: bool) -> str:
    return "ok" if ok else "bad"


def norms(x: np.ndarray, x_adv: np.ndarray) -> dict[str, np.ndarray]:
    delta = (
        np.asarray(x_adv, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    ).reshape(len(x), -1)
    return {
        "l0": (np.abs(delta) > 1e-6).mean(axis=1),
        "l2": np.linalg.norm(delta, axis=1),
        "linf": np.abs(delta).max(axis=1),
    }


def pick_examples(clean_ok: np.ndarray, changed: np.ndarray, k: int) -> np.ndarray:
    """Prefer successful attacks on correctly classified inputs, then fill with the rest."""
    order = np.lexsort((np.arange(len(clean_ok)), ~(clean_ok & changed)))
    return order[:k]


def comparison_table(
    dataset: str,
    title: str,
    x: np.ndarray,
    x_adv: np.ndarray,
    y: np.ndarray,
    p_clean: np.ndarray,
    p_adv: np.ndarray,
    idx: Sequence[int],
    headers: tuple[str, str, str] = (
        "Original",
        "Adversarial",
        "Perturbation (amplified)",
    ),
) -> ImageTable | None:
    orig, adv = to_display(dataset, x[idx]), to_display(dataset, x_adv[idx])
    if orig is None:
        return None
    pert = perturbation_view(adv - orig)
    images = np.stack([orig, adv, pert], axis=1)
    captions, states = [], []
    for row, i in enumerate(idx):
        pc, pa = int(p_clean[i].argmax()), int(p_adv[i].argmax())
        captions.append(
            [
                f"true {label(dataset, int(y[i]))} · pred {label(dataset, pc)} ({p_clean[i, pc]:.0%})",
                f"pred {label(dataset, pa)} ({p_adv[i, pa]:.0%})",
                f"L∞ {np.abs(adv[row] - orig[row]).max():.3f}",
            ]
        )
        states.append([status(pc == y[i]), status(pa == y[i]), None])
    return ImageTable(title, list(headers), images, captions, states)


def prediction_grid(
    dataset: str,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    k: int = 16,
    columns: int = 8,
    target: np.ndarray | None = None,
) -> ImageGrid | None:
    imgs = to_display(dataset, x[:k])
    if imgs is None:
        return None
    pred = p[:k].argmax(axis=1)
    caps = [
        f"true {label(dataset, int(t))}\npred {label(dataset, int(q))}"
        for t, q in zip(y[:k], pred)
    ]
    good = pred == (y[:k] if target is None else target[:k])
    return ImageGrid(
        title,
        imgs,
        caps,
        [status(g if target is None else not g) for g in good],
        columns,
    )


# --- Standard evasion report -------------------------------------------------------------------


def evasion_report(
    clf,
    dataset: str,
    x: np.ndarray,
    y: np.ndarray,
    x_adv: np.ndarray,
    *,
    target: int | np.ndarray | None = None,
    examples: int = 6,
    grid: int = 16,
    extra: Callable[[Result], None] | None = None,
    feature_names: Sequence[str] | None = None,
) -> Result:
    """Clean vs adversarial accuracy, weighted scores, perturbation norms and example images."""
    y = np.asarray(y, dtype=int)
    p_clean, p_adv = probabilities(clf, x), probabilities(clf, x_adv)
    pred_clean, pred_adv = p_clean.argmax(1), p_adv.argmax(1)
    clean_ok = pred_clean == y
    changed = pred_adv != pred_clean
    n = norms(x, x_adv)

    res = Result()
    acc_clean, acc_adv = clean_ok.mean(), (pred_adv == y).mean()
    res.metric("Clean accuracy", acc_clean)
    res.metric("Adversarial accuracy", acc_adv, delta=acc_adv - acc_clean)
    if target is None:
        success = (clean_ok & (pred_adv != y)).sum() / max(clean_ok.sum(), 1)
        res.metric(
            "Attack success rate",
            success,
            help="Correctly classified inputs that became misclassified.",
        )
    else:
        tgt = np.broadcast_to(np.asarray(target), y.shape)
        res.metric(
            "Targeted success rate",
            (pred_adv == tgt).mean(),
            help="Adversarial inputs classified as the target.",
        )
    res.metric(
        "Mean L2 / L∞", f"{n['l2'].mean():.3f} / {n['linf'].mean():.3f}", fmt="text"
    )
    res.metric("Samples", len(x), fmt="int")

    s_clean, s_adv = weighted_scores(y, pred_clean), weighted_scores(y, pred_adv)
    res.add(
        BarChart(
            "Weighted scores (notebook metrics)",
            list(s_clean),
            {"clean": list(s_clean.values()), "adversarial": list(s_adv.values())},
        )
    )
    idx = pick_examples(clean_ok, changed, examples)
    table = comparison_table(dataset, "Examples", x, x_adv, y, p_clean, p_adv, idx)
    if table is not None:
        res.add(table)
        tgt_arr = (
            None if target is None else np.broadcast_to(np.asarray(target), y.shape)
        )
        res.add(
            prediction_grid(
                dataset,
                "Adversarial test samples (red = attack succeeded)",
                x_adv,
                y,
                p_adv,
                grid,
                target=tgt_arr,
            )
        )
    else:
        res.add(
            feature_table(
                dataset, x, x_adv, y, pred_clean, pred_adv, idx, feature_names
            )
        )
    if extra is not None:
        extra(res)
    return res


def feature_table(
    dataset,
    x,
    x_adv,
    y,
    pred_clean,
    pred_adv,
    idx,
    feature_names: Sequence[str] | None = None,
) -> Table:
    k = x.shape[1]
    names = list(feature_names) if feature_names else [f"f{j}" for j in range(k)]
    rows = []
    for i in idx:
        rows.append(
            [
                "original",
                int(i),
                label(dataset, int(y[i])),
                label(dataset, int(pred_clean[i])),
                *np.round(x[i], 3).tolist(),
            ]
        )
        rows.append(
            [
                "adversarial",
                int(i),
                "",
                label(dataset, int(pred_adv[i])),
                *np.round(x_adv[i], 3).tolist(),
            ]
        )
    return Table(
        "Feature values before / after the attack",
        ["row", "sample", "true", "pred", *names],
        rows,
    )


def accuracy_curve(
    title: str, xs: Sequence[float], series: dict[str, Sequence[float]], x_label: str
) -> LineChart:
    return LineChart(
        title,
        [float(v) for v in xs],
        {k: [float(v) for v in vs] for k, vs in series.items()},
        x_label=x_label,
        y_label="accuracy",
    )
