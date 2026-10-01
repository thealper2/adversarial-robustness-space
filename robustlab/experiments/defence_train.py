"""Adversarially trained models (trained offline) evaluated against live attacks."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import (
    BasicIterativeMethod,
    FastGradientMethod,
    ProjectedGradientDescent,
    VirtualAdversarialMethod,
)

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier, meta
from ..results import BarChart, Result, Table


@experiment(
    id="adversarial_trainer",
    title="Adversarial Training (BIM)",
    category="defence_train",
    dataset="mnist",
    models=("mnist_mlp_small", "mnist_mlp_small_bim_at"),
    art=("AdversarialTrainer", "BasicIterativeMethod"),
    notebook="adversarial trainer/MNIST Adversarial Trainer.ipynb",
    kind="defence",
    summary="Half of every training batch is replaced by BIM adversarial examples (ratio 0.5). Both models are "
    "attacked white-box with the same BIM settings.",
    params=(
        Param.float("eps", "BIM ε", 0.3, 0.05, 0.3, 0.05),
        Param.int("n", "Test samples", 200, 100, 1000, 100),
    ),
)
def run_adv_trainer(p):
    models = {
        "standard": classifier("mnist_mlp_small"),
        "adversarially trained": classifier("mnist_mlp_small_bim_at"),
    }
    x, y = data.split("mnist", "test", p["n"])
    bim = lambda m, eps: BasicIterativeMethod(
        m, eps=eps, eps_step=0.01, max_iter=10, verbose=False
    ).generate(x)
    res = Result()
    clean = {k: R.accuracy(m, x, y) for k, m in models.items()}
    adv = {k: R.accuracy(m, bim(m, p["eps"]), y) for k, m in models.items()}
    for k in models:
        res.metric(f"{k} · clean", clean[k])
        res.metric(f"{k} · BIM", adv[k], delta=adv[k] - clean[k])
    res.add(
        BarChart(
            "Accuracy",
            ["clean", f"BIM ε={p['eps']}"],
            {k: [clean[k], adv[k]] for k in models},
        )
    )
    grid = [0.1, 0.2, 0.3]
    res.add(
        R.accuracy_curve(
            "BIM accuracy vs ε",
            grid,
            {k: [R.accuracy(m, bim(m, e), y) for e in grid] for k, m in models.items()},
            "ε",
        )
    )
    return res


@experiment(
    id="virtual_adversarial_training",
    title="Virtual Adversarial Training",
    category="defence_train",
    dataset="mnist",
    models=("mnist_cnn_vat_base", "mnist_cnn_vat"),
    art=("AdversarialTrainer", "VirtualAdversarialMethod", "FastGradientMethod"),
    notebook="virtual adversarial training/MNIST Virtual Adversarial Training.ipynb",
    kind="defence",
    summary="Adversarial training with label-free VAM examples: it smooths the output distribution (lower KL) "
    "and also helps against FGSM.",
    params=(Param.int("n", "Test samples", 200, 100, 500, 100),),
    details="Each VAM run costs one forward pass per pixel; the full evaluation takes ≈6 CPU minutes, so it is precomputed.",
    cost="slow",
    live=False,
)
def run_vat(p):
    models = {
        "baseline": classifier("mnist_cnn_vat_base"),
        "VAT": classifier("mnist_cnn_vat"),
    }
    x, y = data.split("mnist", "test", p["n"])
    res = Result()
    for k, m in models.items():
        res.metric(f"{k} · clean", R.accuracy(m, x, y))
    rows = []
    for eps in (1.0, 2.0, 5.0, 10.0):
        row = [eps]
        for m in models.values():
            x_adv = VirtualAdversarialMethod(
                m, max_iter=3, eps=eps, batch_size=64, verbose=False
            ).generate(x)
            pc, pa = R.probabilities(m, x), R.probabilities(m, x_adv)
            kl = np.sum(pc * (np.log(pc + 1e-12) - np.log(pa + 1e-12)), axis=1).mean()
            row += [round(R.accuracy(m, x_adv, y), 3), round(float(kl), 4)]
        rows.append(row)
    res.add(
        Table(
            "VAM evaluation",
            ["ε", "baseline acc", "baseline KL", "VAT acc", "VAT KL"],
            rows,
        )
    )
    grid = [0.05, 0.1, 0.15, 0.2, 0.3]
    curves = {
        k: [R.accuracy(m, FastGradientMethod(m, eps=e).generate(x), y) for e in grid]
        for k, m in models.items()
    }
    res.add(R.accuracy_curve("FGSM accuracy vs ε", grid, curves, "ε"))
    return res


PROTOCOLS = {
    "fbf": (
        "Fast is Better than Free (FBF)",
        "cifar_simple_fbf",
        "AdversarialTrainerFBFPyTorch",
        "fbf protocol/cifar10-fast-is-better-than-free-protocol.ipynb",
        "Single-step FGSM training with random start — nearly the cost of standard training.",
    ),
    "awp": (
        "Adversarial Weight Perturbation (AWP)",
        "cifar_simple_awp",
        "AdversarialTrainerAWPPyTorch",
        "awp protocol/cifar10-awp-protocol.ipynb",
        "Adversarial training (TRADES-AWP) that also perturbs the weights to flatten the robust loss landscape.",
    ),
    "trades": (
        "TRADES",
        "cifar_simple_trades",
        "AdversarialTrainerTRADESPyTorch",
        "trades/cifar10-trades.ipynb",
        "Trades clean accuracy against robustness with a KL term between clean and adversarial outputs (β = 6).",
    ),
}


def _protocol_runner(model_id):
    def run(p):
        models = {
            "standard": classifier("cifar_simple_std"),
            "robust": classifier(model_id),
        }
        x, y = data.split("cifar10", "test", p["n"])
        pgd = lambda m, eps: ProjectedGradientDescent(
            m,
            eps=eps,
            eps_step=eps / 4,
            max_iter=10,
            num_random_init=1,
            batch_size=128,
            verbose=False,
        ).generate(x)
        eps = p["eps"] / 255
        res = Result()
        clean = {k: R.accuracy(m, x, y) for k, m in models.items()}
        adv = {k: R.accuracy(m, pgd(m, eps), y) for k, m in models.items()}
        for k in models:
            res.metric(f"{k} · clean", clean[k])
            res.metric(f"{k} · PGD-10", adv[k], delta=adv[k] - clean[k])
        res.add(
            BarChart(
                "Accuracy",
                ["clean", f"PGD ε={p['eps']}/255"],
                {k: [clean[k], adv[k]] for k in models},
            )
        )
        grid = [0, 2, 4, 8, 12]
        res.add(
            R.accuracy_curve(
                "PGD accuracy vs ε (×1/255)",
                grid,
                {
                    k: [R.accuracy(m, pgd(m, e / 255) if e else x, y) for e in grid]
                    for k, m in models.items()
                },
                "ε·255",
            )
        )
        info = meta(model_id)
        if info:
            res.add(
                Table(
                    "Offline training recipe",
                    ["key", "value"],
                    [[k, str(v)] for k, v in info.get("recipe", {}).items()],
                )
            )
        return res

    return run


for _key, (_title, _model, _art, _nb, _summary) in PROTOCOLS.items():
    experiment(
        id=_key,
        title=_title,
        category="defence_train",
        dataset="cifar10",
        models=("cifar_simple_std", _model),
        art=(_art, "ProjectedGradientDescent"),
        notebook=_nb,
        kind="defence",
        summary=_summary,
        params=(
            Param.int("eps", "PGD ε (×1/255)", 8, 2, 16, 2),
            Param.int("n", "Test samples", 200, 100, 500, 100),
        ),
        cost="medium",
    )(_protocol_runner(_model))
