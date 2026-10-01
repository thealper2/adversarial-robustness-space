"""Detecting poisoned training data and backdoored inputs."""

from __future__ import annotations

import numpy as np
from art.defences.detector.poison import ActivationDefence, SpectralSignatureDefense
from art.defences.transformer.poisoning import STRIP
from art.estimators.poison_mitigation.strip.strip import combine_images
from scipy.stats import entropy

from .. import artifacts, data, reporting as R, triggers
from ..core import Param, experiment
from ..models import classifier, spec, torch_model, wrap_pytorch
from ..results import BarChart, ImageGrid, Markdown, Result, Table


def _detection_metrics(flagged: np.ndarray, poisoned: np.ndarray) -> dict[str, float]:
    tp = int((flagged & poisoned).sum())
    return {
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int(poisoned.sum()), 1),
        "flagged": int(flagged.sum()),
        "poisoned": int(poisoned.sum()),
    }


def _suspicious_per_class(report: dict) -> list[float]:
    out = []
    for c in range(10):
        clusters = report.get(f"Class_{c}", {})
        out.append(
            sum(
                v["ptc_data_in_cluster"]
                for v in clusters.values()
                if v.get("suspicious_cluster")
            )
        )
    return out


@experiment(
    id="activation_clustering",
    title="Activation Clustering",
    category="poison_detection",
    dataset="cifar10",
    models=("cifar_cnn", "cifar_cnn_labelflip"),
    art=("ActivationDefence",),
    notebook="activation clustering/03-activation_clustering.ipynb",
    kind="defence",
    summary="Clusters last-layer activations per class (PCA → 2-means). A small, separate cluster suggests poisoned data.",
    params=(Param.int("n", "Samples", 1000, 500, 2000, 500),),
    details="Top chart reproduces the notebook (clean test data through a clean vs a label-flip-poisoned model). The "
    "table runs the detector on the poisoned *training* subset where the flipped samples are known.",
    extra_artifacts=("extras/cifar_labelflip.npz",),
)
def run_activation_clustering(p):
    x, y = data.split("cifar10", "test", p["n"])
    kwargs = dict(clustering_method="KMeans", nb_clusters=2, nb_dims=10, reduce="PCA")
    reports = {}
    for name, mid in (
        ("clean model", "cifar_cnn"),
        ("poisoned model", "cifar_cnn_labelflip"),
    ):
        reports[name], _ = ActivationDefence(
            classifier(mid), x, data.one_hot(y, 10)
        ).detect_poison(**kwargs)
    res = Result()
    res.add(
        BarChart(
            "Share of each class in the 'suspicious' cluster (test data)",
            list(data.CIFAR10_CLASSES),
            {k: _suspicious_per_class(v) for k, v in reports.items()},
        )
    )

    with np.load(
        artifacts.path("extras/cifar_labelflip.npz"), allow_pickle=False
    ) as npz:
        y_poison, flipped = npz["labels"], npz["flipped"].astype(bool)
    x_tr, _ = data.split("cifar10", "train")
    x_tr, y_poison, flipped = x_tr[: p["n"]], y_poison[: p["n"]], flipped[: p["n"]]
    _, is_clean = ActivationDefence(
        classifier("cifar_cnn_labelflip"), x_tr, data.one_hot(y_poison, 10)
    ).detect_poison(**kwargs)
    m = _detection_metrics(np.asarray(is_clean) == 0, flipped)
    res.metric("Flipped samples in training subset", m["poisoned"], fmt="int")
    res.metric("Flagged as suspicious", m["flagged"], fmt="int")
    res.metric("Detection precision", m["precision"])
    res.metric("Detection recall", m["recall"])
    res.add(
        Markdown(
            "Random label flipping is not a backdoor: poisoned samples do not form their own activation cluster, "
            "so Activation Clustering (designed for trigger backdoors) finds them only at chance level."
        )
    )
    return res


@experiment(
    id="spectral_signature",
    title="Spectral Signature",
    category="poison_detection",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("SpectralSignatureDefense",),
    notebook="spectral signature/Spectral Signature.ipynb",
    kind="defence",
    summary="Scores each sample by its projection on the top singular vector of the centred class representations and "
    "removes the highest-scoring 1.5 × expected-poison fraction.",
    params=(
        Param.float("rate", "Label-flip rate (classes 0-2)", 0.3, 0.1, 0.5, 0.1),
        Param.int("n", "Training samples", 2000, 1000, 3000, 500),
    ),
    details="As in the notebook, the model was trained on clean labels and the detector is given flipped labels.",
)
def run_spectral(p):
    x, y = data.split("cifar10", "train", p["n"])
    rng = np.random.default_rng(0)
    y_poison = y.copy()
    for i in rng.choice(len(y), int(p["rate"] * len(y)), replace=False):
        if y[i] in (0, 1, 2):
            y_poison[i] = rng.choice([j for j in range(10) if j != y[i]])
    flipped = y_poison != y
    defence = SpectralSignatureDefense(
        classifier("cifar_cnn"),
        x,
        data.one_hot(y_poison, 10),
        expected_pp_poison=0.33,
        eps_multiplier=1.5,
    )
    _, is_clean = defence.detect_poison()
    flagged = np.asarray(is_clean) == 0
    m = _detection_metrics(flagged, flipped)
    res = Result()
    res.metric("Flipped labels", m["poisoned"], fmt="int")
    res.metric("Flagged", m["flagged"], fmt="int")
    res.metric("Precision", m["precision"])
    res.metric("Recall", m["recall"])
    res.add(
        BarChart(
            "Per (poisoned) label: flipped vs flagged",
            list(data.CIFAR10_CLASSES),
            {
                "flipped": [int((flipped & (y_poison == c)).sum()) for c in range(10)],
                "flagged": [int((flagged & (y_poison == c)).sum()) for c in range(10)],
                "flagged & flipped": [
                    int((flagged & flipped & (y_poison == c)).sum()) for c in range(10)
                ],
            },
            y_label="samples",
            percent=False,
        )
    )
    return res


STRIP_MODEL = "mnist_cnn_strip_backdoor"


@experiment(
    id="strip",
    title="STRIP (STRong Intentional Perturbation)",
    category="poison_detection",
    dataset="mnist",
    models=(STRIP_MODEL,),
    art=("STRIP", "PoisoningAttackBackdoor"),
    notebook="strong intentional perturbation (STRIP)/STRIP.ipynb",
    kind="defence",
    summary="Superimposes each input with clean images. Backdoored inputs keep predicting the target (low entropy); "
    "clean inputs become uncertain (high entropy).",
    params=(
        Param.int("num_samples", "Blends per input", 20, 5, 50, 5),
        Param.float("far", "False acceptance rate", 0.05, 0.01, 0.2, 0.01),
        Param.int("n", "Test samples", 200, 100, 500, 100),
    ),
    details="The notebook found ART's entropy reduces over *samples* (axis 0) rather than over *classes* like the paper; "
    "both scores are shown.",
)
def run_strip(p):
    model = classifier(STRIP_MODEL)
    x_test, y_test = data.split("mnist", "test")
    x_train, y_train = data.split("mnist", "train")
    clean_mask = y_test != 7
    x_clean, y_clean = x_test[clean_mask][: p["n"]], y_test[clean_mask][: p["n"]]
    x_bd = triggers.mnist_white_square(x_clean)
    x_valid = x_train[y_train != 7][:300]

    strip_clf = STRIP(
        classifier=wrap_pytorch(torch_model(STRIP_MODEL), spec(STRIP_MODEL))
    )(num_samples=p["num_samples"], false_acceptance_rate=p["far"])
    strip_clf.mitigate(x_valid)
    reject = lambda preds: float(np.mean(preds.sum(axis=1) == 0))

    rng = np.random.default_rng(0)

    def scores(images):
        art_s, paper_s = [], []
        for img in images:
            blend = np.array(
                [
                    combine_images(img, x_valid[j])
                    for j in rng.choice(len(x_valid), p["num_samples"])
                ]
            )
            pr = R.probabilities(model, blend.astype(np.float32))
            art_s.append(np.sum(entropy(pr, base=2, axis=0)) / p["num_samples"])
            paper_s.append(np.sum(entropy(pr, base=2, axis=1)) / p["num_samples"])
        return np.array(art_s), np.array(paper_s)

    (ca, cp), (ba, bp) = scores(x_clean), scores(x_bd)
    thr = np.percentile(cp, 100 * p["far"])

    res = Result()
    res.metric("Clean accuracy", R.accuracy(model, x_clean, y_clean))
    res.metric("Backdoor success (→ 7)", float((R.predict(model, x_bd) == 7).mean()))
    res.metric(
        "ART STRIP: backdoors rejected",
        reject(strip_clf.predict(x_bd)),
        help="True positive rate",
    )
    res.metric(
        "ART STRIP: clean rejected",
        reject(strip_clf.predict(x_clean)),
        help="False positive rate",
    )
    res.metric(
        "Paper score: backdoors rejected",
        float((bp <= thr).mean()),
        help=f"threshold at {p['far']:.0%} FPR",
    )
    res.add(
        BarChart(
            "Detection at the chosen false-acceptance rate",
            ["clean rejected (FP)", "backdoored rejected (TP)"],
            {
                "ART STRIP (axis=0)": [
                    reject(strip_clf.predict(x_clean)),
                    reject(strip_clf.predict(x_bd)),
                ],
                "paper entropy (axis=1)": [
                    float((cp <= thr).mean()),
                    float((bp <= thr).mean()),
                ],
            },
        )
    )
    bins = np.linspace(0, max(cp.max(), bp.max()) + 1e-6, 13)
    centers = [f"{(a + b) / 2:.2f}" for a, b in zip(bins[:-1], bins[1:])]
    res.add(
        BarChart(
            "Paper entropy distribution",
            centers,
            {
                "clean": (np.histogram(cp, bins)[0] / len(cp)).tolist(),
                "backdoored": (np.histogram(bp, bins)[0] / len(bp)).tolist(),
            },
            y_label="share of inputs",
        )
    )
    res.add(
        Table(
            "Mean entropy",
            ["score", "clean", "backdoored"],
            [
                [
                    "ART (axis=0)",
                    round(float(ca.mean()), 3),
                    round(float(ba.mean()), 3),
                ],
                [
                    "paper (axis=1)",
                    round(float(cp.mean()), 3),
                    round(float(bp.mean()), 3),
                ],
            ],
        )
    )
    blend_demo = np.array(
        [combine_images(x_bd[0], x_valid[j]) for j in range(4)]
        + [combine_images(x_clean[0], x_valid[j]) for j in range(4)]
    )
    pr = R.predict(model, blend_demo.astype(np.float32))
    res.add(
        ImageGrid(
            "Superimposed inputs: backdoored (top) keep predicting 7",
            R.to_display("mnist", np.clip(blend_demo, 0, 1)),
            [f"pred {v}" for v in pr],
            [R.status(v != 7) for v in pr],
            columns=4,
        )
    )
    return res
