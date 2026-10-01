"""Privacy attacks: attribute inference, membership inference, reconstruction, model inversion."""

from __future__ import annotations

import numpy as np
from art.attacks.inference.attribute_inference import (
    AttributeInferenceBlackBox,
    AttributeInferenceMembership,
)
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.model_inversion import MIFace
from art.attacks.inference.reconstruction import DatabaseReconstruction
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier
from ..models.zoo import sklearn_model
from ..results import BarChart, ImageTable, Markdown, Result, Table
from .evasion_classic import IRIS_FEATURES

FEATURE = 9  # the binary "sensitive" attribute of the synthetic data
VALUES = [0.0, 1.0]


def _tabular():
    x_tr, y_tr = data.split("synthetic", "train")
    x_te, y_te = data.split("synthetic", "test")
    return x_tr, y_tr, x_te, y_te


def _infer_rate(inferred, truth):
    return float(np.mean(np.isclose(np.asarray(inferred).ravel(), truth)))


@experiment(
    id="attribute_inference",
    title="Attribute Inference (black-box)",
    category="inference",
    dataset="synthetic",
    models=("synthetic_tree",),
    art=("AttributeInferenceBlackBox",),
    notebook="attribute inference/Attribute Inference BlackBox.ipynb",
    summary="Trains an attack model to predict a hidden feature (feature 9, binary) from the other features and the "
    "target model's output.",
    params=(
        Param.choice(
            "attack_model",
            "Attack model",
            "nn",
            [
                ("nn", "neural network"),
                ("rf", "random forest"),
                ("gb", "gradient boosting"),
            ],
        ),
    ),
    details="Feature 9 does not influence the label in this dataset, so no attack can beat the majority baseline — "
    "the honest outcome the notebook also shows (~50%). The notebook printed model predictions as the 'true' "
    "values; here the real hidden values are used.",
)
def run_attribute(p):
    clf = classifier("synthetic_tree")
    x_tr, y_tr, x_te, _ = _tabular()
    attack = AttributeInferenceBlackBox(
        clf, attack_model_type=p["attack_model"], attack_feature=FEATURE
    )
    attack.fit(x_tr[:500])
    res = Result()
    rows = []
    for name, x in (
        ("members (training rows)", x_tr[500:]),
        ("non-members (test rows)", x_te[500:]),
    ):
        pred = R.predict(clf, x).reshape(-1, 1).astype(float)
        truth = x[:, FEATURE]
        inferred = attack.infer(np.delete(x, FEATURE, 1), pred=pred, values=VALUES)
        rate = _infer_rate(inferred, truth)
        baseline = max(truth.mean(), 1 - truth.mean())
        res.metric(f"Inference accuracy · {name}", rate, delta=rate - baseline)
        rows.append([name, round(rate, 3), round(float(baseline), 3)])
    res.add(
        Table(
            "Hidden-attribute inference",
            ["records", "attack accuracy", "majority baseline"],
            rows,
        )
    )
    res.add(
        BarChart(
            "Attack vs baseline",
            [r[0] for r in rows],
            {"attack": [r[1] for r in rows], "baseline": [r[2] for r in rows]},
        )
    )
    return res


@experiment(
    id="membership_inference",
    title="Membership Inference (black-box)",
    category="inference",
    dataset="synthetic",
    models=("synthetic_tree",),
    art=("MembershipInferenceBlackBox", "AttributeInferenceMembership"),
    notebook="membership inference/Membership Inference BlackBox.ipynb",
    summary="Learns to tell training records from unseen records using the model's predictions — an unpruned decision "
    "tree memorises its training set, so the gap is large.",
    params=(
        Param.choice(
            "attack_model",
            "Attack model",
            "nn",
            [
                ("nn", "neural network"),
                ("rf", "random forest"),
                ("gb", "gradient boosting"),
            ],
        ),
    ),
)
def run_membership(p):
    clf = classifier("synthetic_tree")
    x_tr, y_tr, x_te, y_te = _tabular()
    mia = MembershipInferenceBlackBox(clf, attack_model_type=p["attack_model"])
    mia.fit(x_tr[:500], y_tr[:500], x_te[:500], y_te[:500])
    in_pred = np.asarray(mia.infer(x_tr[500:], y_tr[500:])).ravel()
    out_pred = np.asarray(mia.infer(x_te[500:], y_te[500:])).ravel()
    acc = (in_pred.sum() + (1 - out_pred).sum()) / (len(in_pred) + len(out_pred))
    precision = in_pred.sum() / max(in_pred.sum() + out_pred.sum(), 1)
    res = Result()
    res.metric(
        "Train / test accuracy",
        f"{R.accuracy(clf, x_tr, y_tr):.1%} / {R.accuracy(clf, x_te, y_te):.1%}",
        fmt="text",
    )
    res.metric(
        "Membership accuracy", acc, delta=acc - 0.5, help="50% = random guessing"
    )
    res.metric("Precision (member)", precision)
    res.metric("Recall (member)", float(in_pred.mean()))

    ai = AttributeInferenceMembership(clf, mia, attack_feature=FEATURE)
    x_m = x_tr[500:]
    inferred = ai.infer(np.delete(x_m, FEATURE, 1), y=y_tr[500:], values=VALUES)
    res.metric(
        "Attribute inference via membership", _infer_rate(inferred, x_m[:, FEATURE])
    )
    res.add(
        BarChart(
            "Predicted as member",
            ["true members", "true non-members"],
            {"share": [float(in_pred.mean()), float(out_pred.mean())]},
        )
    )
    return res


@experiment(
    id="database_reconstruction",
    title="Database Reconstruction",
    category="inference",
    dataset="iris",
    models=("iris_logreg", "iris_dp_logreg"),
    art=("DatabaseReconstruction", "ScikitlearnLogisticRegression"),
    notebook="database reconstruction/IRIS Database Reconstruction.ipynb",
    summary="An attacker who knows all training rows but one, plus the trained model, solves for the missing row. "
    "Differential privacy (diffprivlib) is meant to prevent this.",
    params=(Param.choice("row", "Unknown training row", 114, [114, 0, 30, 60, 90]),),
    details="ART re-fits the model inside L-BFGS with numerical gradients. When those gradients vanish, the search "
    "stays at its initial guess (the first known row) — the notebook's result shows exactly this behaviour.",
    cost="medium",
)
def run_reconstruction(p):
    x_tr, y_tr = data.split("iris", "dr_train")
    row = p["row"]
    x_e, y_e = np.delete(x_tr, row, axis=0), np.delete(y_tr, row, axis=0)
    res = Result()
    rows = [
        [
            "target (true)",
            *np.round(x_tr[row], 2).tolist(),
            data.IRIS_CLASSES[int(y_tr[row])],
            "—",
        ]
    ]
    for name, mid in (
        ("standard model", "iris_logreg"),
        ("DP model (ε=1)", "iris_dp_logreg"),
    ):
        model = sklearn_model(mid)
        rec_x, rec_y = DatabaseReconstruction(
            ScikitlearnLogisticRegression(model)
        ).reconstruct(x_e, y_e)
        rmse = float(np.sqrt(np.mean((rec_x[0] - x_tr[row]) ** 2)))
        moved = not np.allclose(rec_x[0], x_e[0], atol=1e-4)
        res.metric(
            f"RMSE · {name}",
            rmse,
            fmt="float",
            help="Search left its initial guess"
            if moved
            else "Search stayed at its initial guess (first known row)",
        )
        res.metric(
            f"Test accuracy · {name}", model.score(*data.split("iris", "dr_test"))
        )
        rows.append(
            [
                f"reconstructed · {name}",
                *np.round(rec_x[0], 2).tolist(),
                data.IRIS_CLASSES[int(np.argmax(rec_y))],
                round(rmse, 3),
            ]
        )
    rows.append(
        [
            "initial guess (first known row)",
            *np.round(x_e[0], 2).tolist(),
            data.IRIS_CLASSES[int(y_e[0])],
            "—",
        ]
    )
    res.add(
        Table(
            "Target row vs reconstructions",
            ["row", *IRIS_FEATURES, "class", "RMSE"],
            rows,
        )
    )
    res.add(
        Markdown(
            "On this data ART's finite-difference gradients vanish, so the reconstruction is often just the "
            "initial guess — for both models. The DP model (diffprivlib, ε = 1) pays for its privacy with a large "
            "accuracy drop on 120 training rows."
        )
    )
    return res


@experiment(
    id="miface",
    title="Model Inversion (MIFace)",
    category="inference",
    dataset="fashion_mnist",
    models=("fashion_cnn",),
    art=("MIFace",),
    notebook="miface/Fashion MNIST MIFace Model Inversion Attack.ipynb",
    summary="Starting from the average image, gradient ascent on each class score reconstructs a prototype of what the "
    "model has learned for that class.",
    params=(
        Param.choice("max_iter", "Iterations", 10000, [1000, 3000, 10000]),
        Param.float("learning_rate", "Learning rate", 0.1, 0.05, 1.0, 0.05),
    ),
    details="10,000 gradient-ascent steps per class (as in the notebook) take ≈4 CPU minutes, so the result is precomputed.",
    cost="slow",
    live=False,
)
def run_miface(p):
    clf = classifier("fashion_cnn")
    x, y = data.split("fashion_mnist", "test")
    attack = MIFace(
        clf,
        max_iter=p["max_iter"],
        threshold=1.0,
        learning_rate=p["learning_rate"],
        batch_size=10,
        verbose=False,
    )
    start = np.repeat(x.mean(axis=0, keepdims=True), 10, axis=0)
    inferred = attack.infer(start, np.arange(10)).astype(np.float32)
    probs = R.probabilities(clf, inferred)
    res = Result()
    res.metric(
        "Inversions classified as intended class",
        float((probs.argmax(1) == np.arange(10)).mean()),
    )
    res.metric(
        "Mean confidence for intended class",
        float(probs[np.arange(10), np.arange(10)].mean()),
    )
    means = np.stack([x[y == c].mean(axis=0) for c in range(10)])
    imgs = np.stack(
        [
            R.to_display("fashion_mnist", means),
            R.to_display("fashion_mnist", np.clip(inferred, 0, 1)),
        ],
        axis=1,
    )
    caps = [
        [data.FASHION_CLASSES[c] + " (real average)", f"inverted · p={probs[c, c]:.2f}"]
        for c in range(10)
    ]
    res.add(
        ImageTable(
            "Average training-like image vs model inversion",
            ["Real class average", "MIFace reconstruction"],
            imgs,
            caps,
        )
    )
    return res
