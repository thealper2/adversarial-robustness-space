"""Formal robustness certification (abstract interpretation) with ART."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.certification.deep_z import PytorchDeepZ
from art.estimators.certification.interval import PyTorchIBPClassifier
from torch import nn

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier, spec, torch_model
from ..models.architectures import CertNetForVerifiers
from ..models.zoo import art_device
from ..results import BarChart, Markdown, Result

MODEL = "mnist_cert_cnn"
EXPLAIN = (
    "**Certified** means the verifier proved that *no* input in the L∞ box keeps a different label — a lower bound on "
    "robust accuracy. **PGD accuracy** is an upper bound (the attack may just be too weak). The truth lies in between."
)


def _pgd_accuracy(x, y, eps):
    clf = classifier(MODEL)
    x_adv = ProjectedGradientDescent(
        clf, eps=eps, eps_step=eps / 20, max_iter=20, num_random_init=1, verbose=False
    ).generate(x)
    return R.accuracy(clf, x_adv, y)


def _wrap(cls):
    s = spec(MODEL)
    model = CertNetForVerifiers.from_sequential(torch_model(MODEL)).to(
        torch_model(MODEL)[0].weight.device
    )
    return cls(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=s.input_shape,
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type=art_device(),
    )


@experiment(
    id="deep_z",
    title="DeepZ (zonotope) certification",
    category="certification",
    dataset="mnist",
    models=(MODEL,),
    art=("PytorchDeepZ", "ProjectedGradientDescent"),
    notebook="deep z/MNIST DeepZ.ipynb",
    kind="defence",
    summary="Propagates a zonotope (an affine set) through the network to prove that every input within ε keeps its label.",
    params=(
        Param.choice("bound", "ε (L∞)", 0.02, [0.005, 0.01, 0.02, 0.03, 0.05]),
        Param.int("n", "Test samples", 30, 10, 60, 10),
    ),
    details="The notebook only evaluated PGD on the zonotope model in concrete mode; this version also runs the certificate.",
)
def run_deepz(p):
    x, y = data.split("mnist", "test", p["n"])
    zono = _wrap(PytorchDeepZ)
    certified = correct = 0
    for sample, label in zip(x, y):
        zono.model.set_forward_mode("concrete")
        pred = int(np.argmax(zono.predict(sample[None])))
        if pred != label:
            continue
        correct += 1
        zono.model.set_forward_mode("abstract")
        cent, eps = zono.pre_process(
            cent=np.copy(sample), eps=np.eye(sample.size) * p["bound"]
        )
        certified += bool(
            zono.certify(cent=np.expand_dims(cent, 0), eps=eps, prediction=pred)
        )
    zono.model.set_forward_mode("concrete")
    pgd = _pgd_accuracy(x, y, p["bound"])
    res = Result()
    res.metric("Clean accuracy", correct / len(x))
    res.metric("Certified accuracy", certified / len(x))
    res.metric("PGD accuracy (upper bound)", pgd)
    res.add(Markdown(EXPLAIN))
    res.add(
        BarChart(
            f"Accuracy at ε = {p['bound']}",
            ["clean", "certified (DeepZ)", "PGD-20"],
            {"accuracy": [correct / len(x), certified / len(x), pgd]},
        )
    )
    return res


@experiment(
    id="interval_bound_propagation",
    title="Interval Bound Propagation (IBP)",
    category="certification",
    dataset="mnist",
    models=(MODEL,),
    art=("PyTorchIBPClassifier", "ProjectedGradientDescent"),
    notebook="interval bound propagation/MNIST IBP.ipynb",
    kind="defence",
    summary="Propagates per-neuron lower/upper bounds (boxes) through the network — cheaper but looser than DeepZ.",
    params=(
        Param.choice(
            "bound", "ε (L∞) for the headline numbers", 0.01, [0.005, 0.01, 0.02, 0.05]
        ),
        Param.int("n", "Test samples", 500, 100, 1000, 100),
    ),
)
def run_ibp(p):
    x, y = data.split("mnist", "test", p["n"])
    ibp = _wrap(PyTorchIBPClassifier)

    def certified(eps):
        # ART expects the lower/upper bounds stacked on the channel axis: (N, 2, 28, 28) for (N, 1, 28, 28) inputs.
        box = np.concatenate([np.clip(x - eps, 0, 1), np.clip(x + eps, 0, 1)], axis=1)
        ibp.model.set_forward_mode("abstract")
        preds = ibp.predict_intervals(x=box, is_interval=True, batch_size=64)
        return float(np.mean(ibp.certify(preds=preds, labels=y)))

    res = Result()
    res.metric("Clean accuracy", R.accuracy(classifier(MODEL), x, y))
    res.metric(f"Certified @ ε={p['bound']}", certified(p["bound"]))
    res.metric(f"PGD accuracy @ ε={p['bound']}", _pgd_accuracy(x, y, p["bound"]))
    res.add(Markdown(EXPLAIN))
    grid = [0.001, 0.005, 0.01, 0.02, 0.05]
    res.add(
        R.accuracy_curve(
            "Certified vs empirical robustness",
            grid,
            {
                "certified (IBP)": [certified(e) for e in grid],
                "PGD-20 (upper bound)": [_pgd_accuracy(x, y, e) for e in grid],
            },
            "ε (L∞)",
        )
    )
    return res
