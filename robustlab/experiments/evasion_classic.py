"""Evasion attacks on classic ML models (decision tree, logistic regression, Gaussian process)."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import (
    DecisionTreeAttack,
    HighConfidenceLowUncertainty,
    LowProFool,
)
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier
from ..models.zoo import sklearn_model
from ..results import BarChart, Result, Table

IRIS_FEATURES = ("sepal length", "sepal width", "petal length", "petal width")


@experiment(
    id="decision_tree_attack",
    title="Decision Tree Attack",
    category="evasion_classic",
    dataset="digits",
    models=("digits_tree",),
    art=("DecisionTreeAttack",),
    notebook="decision tree attack/DT Attack.ipynb",
    summary="Walks the fitted tree to find the nearest leaf of another class and moves just the needed features.",
    params=(
        Param.float(
            "offset", "Offset past the split threshold", 0.001, 0.001, 1.0, 0.001
        ),
        Param.int("n", "Test samples", 180, 20, 180, 20),
    ),
)
def run_dt(p):
    clf = classifier("digits_tree")
    x, y = data.split("digits", "test", p["n"])
    x_adv = DecisionTreeAttack(clf, offset=p["offset"], verbose=False).generate(x)

    def extra(res):
        res.metric(
            "Features changed (mean of 64)",
            float((np.abs(x_adv - x) > 1e-9).sum(axis=1).mean()),
            fmt="float",
        )

    return R.evasion_report(clf, "digits", x, y, x_adv, extra=extra)


@experiment(
    id="low_pro_fool",
    title="LowProFool",
    category="evasion_classic",
    dataset="iris",
    models=("iris_logreg_scaled",),
    art=("LowProFool",),
    notebook="low pro fool/IRIS LowProFool.ipynb",
    summary="Imperceptible tabular attack: weights the perturbation by feature importance so that important "
    "features barely move.",
    params=(
        Param.choice(
            "n_steps",
            "Steps",
            100,
            [(10, "10 (notebook)"), (50, "50"), (100, "100"), (200, "200")],
        ),
        Param.float("lambd", "λ (imperceptibility weight)", 1.65, 0.1, 5.0, 0.05),
    ),
    details="ART's `y` is the *target* class. The notebook passed the true labels, which pushed samples towards their "
    "own class (accuracy 1.0 after the 'attack'). Here each sample targets its second most likely class.",
)
def run_lpf(p):
    x_train, y_train = data.split("iris", "lpf_train")
    x, y = data.split("iris", "lpf_test")
    clf = ScikitlearnLogisticRegression(
        sklearn_model("iris_logreg_scaled"),
        clip_values=(float(x_train.min()), float(x_train.max())),
    )
    attack = LowProFool(
        clf, n_steps=p["n_steps"], eta=10, lambd=p["lambd"], eta_decay=0.95
    )
    attack.fit_importances(x_train, y_train)
    # ``y`` is the *target* class in ART's LowProFool: aim for each sample's second most likely class.
    target = np.argsort(clf.predict(x), axis=1)[:, -2]
    x_adv = attack.generate(x, y=data.one_hot(target, 3))

    def extra(res):
        res.add(
            BarChart(
                "Feature importance (Pearson)",
                list(IRIS_FEATURES),
                {
                    "importance": np.asarray(attack.importance_vec, dtype=float)
                    .ravel()
                    .tolist()
                },
                percent=False,
            )
        )

    return R.evasion_report(
        clf,
        "iris",
        x,
        y,
        x_adv,
        target=target,
        extra=extra,
        feature_names=IRIS_FEATURES,
    )


@experiment(
    id="hclu",
    title="High-Confidence-Low-Uncertainty",
    category="evasion_classic",
    dataset="breast_cancer",
    models=("breast_cancer_gpc",),
    art=("HighConfidenceLowUncertainty", "GPyGaussianProcessClassifier"),
    notebook="high confidence low uncertainty/GPy HCLU.ipynb",
    summary="Attacks a Gaussian-process classifier so that adversarial points get a confident prediction *and* "
    "a low predictive uncertainty — defeating uncertainty-based detection.",
    params=(
        Param.float("conf", "Required confidence", 0.8, 0.6, 0.95, 0.05),
        Param.float(
            "unc_increase", "Allowed uncertainty increase", 0.9, 0.5, 100.0, 0.1
        ),
        Param.int("n", "Test samples", 10, 5, 30, 5),
    ),
    details="The notebook scored `argmax` over the adversarial *features*; here the adversarial points are classified by the GP.",
    cost="medium",
)
def run_hclu(p):
    clf = classifier("breast_cancer_gpc")
    x, y = data.split("breast_cancer", "test", p["n"])
    y = (
        1 - y
    )  # ART's GPy wrapper puts P(GPy label 1) in column 0, so ART class index = 1 - label
    variants = {
        "high confidence": HighConfidenceLowUncertainty(
            clf, conf=p["conf"], min_val=-1.0, max_val=2.0, verbose=False
        ),
        "low uncertainty": HighConfidenceLowUncertainty(
            clf, unc_increase=p["unc_increase"], min_val=0.0, max_val=2.0, verbose=False
        ),
    }
    res = Result()
    prob_clean = clf.predict(x)
    unc_clean = clf.predict_uncertainty(x).ravel()
    res.metric("Clean accuracy", (prob_clean.argmax(1) == y).mean())
    acc, conf, unc = (
        {"clean": (prob_clean.argmax(1) == y).mean()},
        {"clean": prob_clean.max(1).mean()},
        {"clean": unc_clean.mean()},
    )
    for name, attack in variants.items():
        x_adv = attack.generate(x)
        prob = clf.predict(x_adv)
        acc[name], conf[name], unc[name] = (
            (prob.argmax(1) == y).mean(),
            prob.max(1).mean(),
            clf.predict_uncertainty(x_adv).mean(),
        )
        res.metric(f"Accuracy ({name})", acc[name], delta=acc[name] - acc["clean"])
    cats = list(acc)
    res.add(
        BarChart(
            "Accuracy and mean confidence",
            cats,
            {"accuracy": [acc[c] for c in cats], "confidence": [conf[c] for c in cats]},
        )
    )
    res.add(
        BarChart(
            "Mean predictive uncertainty",
            cats,
            {"uncertainty": [float(unc[c]) for c in cats]},
            percent=False,
        )
    )
    res.add(
        Table(
            "Summary",
            ["input", "accuracy", "mean confidence", "mean uncertainty"],
            [
                [
                    c,
                    round(float(acc[c]), 3),
                    round(float(conf[c]), 3),
                    round(float(unc[c]), 4),
                ]
                for c in cats
            ],
        )
    )
    return res
