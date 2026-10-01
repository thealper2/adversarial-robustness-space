"""Model extraction: the thief models were trained offline from victim queries; the Space compares them."""

from __future__ import annotations

import numpy as np

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier, meta
from ..results import BarChart, ImageGrid, Result, Table

THIEVES = {
    "copycat_cnn": (
        "CopycatCNN",
        "mnist_copycat",
        "copycat cnn/CopycatCNN.ipynb",
        "Queries the victim with natural (unlabelled) images and trains a copy on its answers.",
    ),
    "knockoff_nets": (
        "Knockoff Nets",
        "mnist_knockoff",
        "knockoff nets/Knockoff Nets.ipynb",
        "Queries the victim only with Fashion-MNIST clothing images (a transfer set without a single digit) and trains a knock-off on its answers.",
    ),
}


def _runner(thief_id):
    def run(p):
        victim, thief = classifier("mnist_victim"), classifier(thief_id)
        x, y = data.split("mnist", "test", p["n"])
        pv, pt = R.predict(victim, x), R.predict(thief, x)
        res = Result()
        res.metric("Victim accuracy", (pv == y).mean())
        res.metric(
            "Stolen model accuracy",
            (pt == y).mean(),
            delta=(pt == y).mean() - (pv == y).mean(),
        )
        res.metric(
            "Fidelity (agreement)",
            (pv == pt).mean(),
            help="Share of inputs where both models agree",
        )
        info = meta(thief_id).get("recipe", {})
        if info:
            res.metric("Victim queries used", int(info.get("nb_stolen", 0)), fmt="int")
        res.add(
            BarChart(
                "Per-class accuracy",
                [str(c) for c in range(10)],
                {
                    "victim": [float((pv[y == c] == c).mean()) for c in range(10)],
                    "stolen": [float((pt[y == c] == c).mean()) for c in range(10)],
                },
            )
        )
        dis = np.where(pv != pt)[0][:16]
        if len(dis):
            imgs = R.to_display("mnist", x[dis])
            res.add(
                ImageGrid(
                    "Where the copy disagrees with the victim",
                    imgs,
                    [f"victim {pv[i]} · stolen {pt[i]}" for i in dis],
                    columns=8,
                )
            )
        if info:
            res.add(
                Table(
                    "Offline extraction recipe",
                    ["key", "value"],
                    [[k, str(v)] for k, v in info.items()],
                )
            )
        return res

    return run


for _id, (_title, _thief, _nb, _summary) in THIEVES.items():
    experiment(
        id=_id,
        title=_title,
        category="extraction",
        dataset="mnist",
        models=("mnist_victim", _thief),
        art=(_title.replace(" ", ""),),
        notebook=_nb,
        summary=_summary,
        params=(Param.int("n", "Test samples", 1000, 200, 2000, 200),),
    )(_runner(_thief))
