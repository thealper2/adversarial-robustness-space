"""Black-box (score- and decision-based) evasion attacks against the CIFAR-10 CNN."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import (
    BoundaryAttack,
    GRAPHITEBlackbox,
    HopSkipJump,
    PixelAttack,
    SignOPTAttack,
    SimBA,
    SquareAttack,
    ZooAttack,
)
from art.attacks.evasion.laser_attack.laser_attack import LaserBeamAttack
from art.attacks.evasion.laser_attack.utils import ImageGenerator

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier
from ..models.zoo import probability_classifier
from ..results import ImageGrid, LineChart, Result, Table

N16 = Param.int("n", "Test samples", 16, 4, 32, 4)
N8 = Param.int("n", "Test samples", 8, 2, 16, 2)


def _cifar(n):
    return classifier("cifar_cnn"), *data.split("cifar10", "test", n)


@experiment(
    id="boundary",
    title="Boundary Attack",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("BoundaryAttack",),
    notebook="boundary attack/CIFAR10 Boundary Attack.ipynb",
    summary="Decision-based: starts from a misclassified point and walks along the decision boundary towards the input.",
    params=(Param.int("max_iter", "Iterations", 100, 20, 200, 20), N16),
    cost="medium",
)
def run_boundary(p):
    clf, x, y = _cifar(p["n"])
    attack = BoundaryAttack(
        clf, targeted=False, max_iter=p["max_iter"], batch_size=64, verbose=False
    )
    return R.evasion_report(clf, "cifar10", x, y, attack.generate(x))


@experiment(
    id="hop_skip_jump",
    title="HopSkipJump",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("HopSkipJump",),
    notebook="hop skip jump attack/CIFAR10 HopSkipJump.ipynb",
    summary="Decision-based attack using gradient estimation at the boundary. The image stays adversarial "
    "while its distance to the original shrinks step by step.",
    params=(Param.int("steps", "Resume steps (×10 iterations)", 20, 5, 30, 5),),
    cost="medium",
)
def run_hsj(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", 10)
    init_image, target_image = x[1], x[6]
    attack = HopSkipJump(
        clf, targeted=False, max_iter=0, max_eval=1000, init_eval=10, verbose=False
    )
    mask = np.random.binomial(n=1, p=0.9, size=target_image.shape).astype(np.float32)
    x_adv = init_image[None].copy()
    l2, snaps, snap_steps = [], [], {1, p["steps"] // 4, p["steps"] // 2, p["steps"]}
    for step in range(1, p["steps"] + 1):
        x_adv = attack.generate(
            x=target_image[None],
            y=data.one_hot([y[1]], 10),
            x_adv_init=x_adv,
            resume=True,
            mask=mask,
        )
        attack.max_iter = 10
        l2.append(float(np.linalg.norm((x_adv[0] - target_image).ravel())))
        if step in snap_steps:
            snaps.append((step, x_adv[0].copy()))

    probs = R.probabilities(clf, np.stack([target_image, x_adv[0]]))
    res = Result()
    res.metric(
        "Original prediction", R.label("cifar10", int(probs[0].argmax())), fmt="text"
    )
    res.metric(
        "Adversarial prediction", R.label("cifar10", int(probs[1].argmax())), fmt="text"
    )
    res.metric("Final L2 distance", l2[-1], fmt="float")
    res.metric("Iterations", p["steps"] * 10, fmt="int")
    res.add(
        LineChart(
            "L2 distance to the original image",
            [10 * i for i in range(1, len(l2) + 1)],
            {"L2": l2},
            x_label="iterations",
            y_label="L2",
            percent=False,
        )
    )
    imgs = R.to_display(
        "cifar10", np.stack([target_image, init_image] + [s for _, s in snaps])
    )
    caps = ["original", "starting point (other class)"] + [
        f"after {10 * st} iterations" for st, _ in snaps
    ]
    res.add(ImageGrid("Trajectory", imgs, caps, columns=len(caps)))
    return res


@experiment(
    id="sign_opt",
    title="Sign-OPT",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("SignOPTAttack",),
    notebook="sign-opt attack/CIFAR10 Sign-OPT Attack.ipynb",
    summary="Hard-label attack that estimates the sign of the directional derivative with single queries.",
    params=(
        Param.int("query_limit", "Query limit per image", 2000, 500, 4000, 500),
        N16,
    ),
    cost="medium",
)
def run_sign_opt(p):
    clf, x, y = _cifar(p["n"])
    attack = SignOPTAttack(
        clf, targeted=False, max_iter=100, query_limit=p["query_limit"], verbose=False
    )
    return R.evasion_report(clf, "cifar10", x, y, attack.generate(x))


@experiment(
    id="simba",
    title="Simple Black-box Attack (SimBA)",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("SimBA",),
    notebook="simba/CIFAR10 SimBA.ipynb",
    summary="Score-based: tries random DCT basis directions and keeps each step that lowers the true-class probability.",
    params=(
        Param.float("epsilon", "Step size ε", 0.05, 0.01, 0.2, 0.01),
        Param.choice("attack", "Basis", "dct", ["dct", "px"]),
        N8,
    ),
)
def run_simba(p):
    clf, x, y = _cifar(p["n"])
    clf = probability_classifier("cifar_cnn")  # SimBA requires probability outputs
    attack = SimBA(
        clf, attack=p["attack"], epsilon=p["epsilon"], max_iter=3000, verbose=False
    )
    x_adv = np.concatenate(
        [
            attack.generate(x[i : i + 1], y=data.one_hot(y[i : i + 1], 10))
            for i in range(len(x))
        ]
    )
    return R.evasion_report(clf, "cifar10", x, y, x_adv)


@experiment(
    id="square",
    title="Square Attack",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("SquareAttack",),
    notebook="square attack/CIFAR10 Square Attack.ipynb",
    summary="Score-based random search that perturbs random squares at the ε-ball corners.",
    params=(
        Param.float("eps", "ε (L∞)", 0.3, 0.01, 0.3, 0.01),
        Param.int("max_iter", "Iterations", 100, 20, 500, 20),
        N16,
    ),
    cost="medium",
)
def run_square(p):
    clf, x, y = _cifar(p["n"])
    attack = SquareAttack(
        clf, norm=np.inf, eps=p["eps"], max_iter=p["max_iter"], verbose=False
    )
    return R.evasion_report(clf, "cifar10", x, y, attack.generate(x))


@experiment(
    id="zoo",
    title="Zeroth Order Optimisation (ZOO)",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("ZooAttack",),
    notebook="zoo attack/CIFAR10 ZOO.ipynb",
    summary="C&W objective optimised with coordinate-wise finite differences instead of gradients.",
    params=(Param.float("confidence", "Confidence κ", 0.3, 0.0, 2.0, 0.1), N16),
    cost="medium",
)
def run_zoo(p):
    clf, x, y = _cifar(p["n"])
    attack = ZooAttack(
        clf,
        confidence=p["confidence"],
        targeted=False,
        learning_rate=0.1,
        max_iter=10,
        binary_search_steps=1,
        nb_parallel=128,
        batch_size=1,
        verbose=False,
    )
    return R.evasion_report(clf, "cifar10", x, y, attack.generate(x))


@experiment(
    id="pixel",
    title="Pixel Attack (differential evolution)",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("PixelAttack",),
    notebook="pixel attack/CIFAR10 Pixel Attack.ipynb",
    summary="Changes only a handful of pixels found by differential evolution (the 'one-pixel attack' family).",
    params=(Param.int("max_iter", "Evolution iterations", 100, 20, 200, 20), N16),
    details="Differential evolution needs ≈7 GPU-minutes for 16 images, so the result is precomputed.",
    cost="slow",
    live=False,
)
def run_pixel(p):
    clf, x, y = _cifar(p["n"])
    attack = PixelAttack(clf, th=None, es=1, max_iter=p["max_iter"], verbose=False)
    x_adv = attack.generate(x)

    def extra(res):
        changed = (np.abs(x_adv - x).max(axis=1) > 1e-6).reshape(len(x), -1).sum(axis=1)
        res.metric("Pixels changed (mean)", float(changed.mean()), fmt="float")

    return R.evasion_report(clf, "cifar10", x, y, x_adv, extra=extra)


@experiment(
    id="graphite",
    title="GRAPHITE (black-box)",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("GRAPHITEBlackbox",),
    notebook="graphite black box/CIFAR10 GRAPHITE.ipynb",
    summary="Physically realisable targeted patch attack: mask generation from a target image, then query-based boosting "
    "under random transformations.",
    params=(),
    cost="slow",
    live=False,
    details="GRAPHITE needs OpenCV and ~3 GPU-minutes; it was run offline and is shown precomputed.",
)
def run_graphite(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", 10)
    attack = GRAPHITEBlackbox(
        clf,
        noise_size=(32, 32),
        net_size=(32, 32),
        heat_patch_size=(4, 4),
        heatmap_mode="Target",
    )
    x_adv = attack.generate(
        x=x[1:2], y=data.one_hot(y[8:9], 10), mask=None, x_tar=x[8:9]
    )
    probs = R.probabilities(clf, np.concatenate([x[1:2], x_adv, x[8:9]]))
    pred = probs.argmax(1)
    res = Result()
    res.metric("Original prediction", R.label("cifar10", int(pred[0])), fmt="text")
    res.metric("Adversarial prediction", R.label("cifar10", int(pred[1])), fmt="text")
    res.metric("Target class", R.label("cifar10", int(y[8])), fmt="text")
    res.metric(
        "Pixels changed",
        float((np.abs(x_adv - x[1:2]).max(axis=1) > 1e-6).mean()),
        help="Fraction of pixels",
    )
    imgs = R.to_display("cifar10", np.concatenate([x[1:2], x_adv, x[8:9]]))
    pert = R.perturbation_view(imgs[1:2] - imgs[0:1])
    res.add(
        ImageGrid(
            "Source · adversarial · target-class image · perturbation",
            np.concatenate([imgs, pert]),
            [
                f"source ({R.label('cifar10', int(pred[0]))})",
                f"adversarial ({R.label('cifar10', int(pred[1]))})",
                f"target example ({R.label('cifar10', int(pred[2]))})",
                "perturbation",
            ],
            ["ok", "bad" if pred[1] != y[1] else "ok", None, None],
            columns=4,
        )
    )
    return res


class _NHWCView:
    """Expose an NCHW ART classifier as the NHWC predictor that LaserBeamAttack expects."""

    channels_first = False

    def __init__(self, clf):
        self.clf = clf

    def predict(self, x, **_):
        return R.probabilities(
            self.clf,
            np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)), dtype=np.float32),
        )


@experiment(
    id="laser_beam",
    title="Laser Beam Attack",
    category="evasion_black",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("LaserBeamAttack",),
    notebook="laser attack/CIFAR10 - Laser Beam Attack.ipynb",
    summary="Physical-world attack: searches wavelength, angle, offset and width of a simulated laser line.",
    params=(
        Param.int("iterations", "Random-search iterations", 50, 10, 200, 10),
        Param.int("n", "Test samples", 5, 1, 16),
    ),
)
def run_laser(p):
    clf, x, y = _cifar(p["n"])
    attack = LaserBeamAttack(
        estimator=_NHWCView(clf),
        iterations=p["iterations"],
        max_laser_beam=(780, 3.14, 32, 32),
    )
    x_nhwc = np.transpose(x, (0, 2, 3, 1))
    params = attack.generate_parameters(x_nhwc)
    painter = ImageGenerator()
    x_adv = np.stack(
        [
            img if beam is None else painter.update_image(img, beam)
            for img, (beam, _) in zip(x_nhwc, params)
        ]
    )
    x_adv = np.transpose(x_adv, (0, 3, 1, 2)).astype(np.float32)

    def extra(res):
        rows = []
        for i, (beam, cls) in enumerate(params):
            if beam is None:
                rows.append([i, "not found", "", "", "", ""])
            else:
                rows.append(
                    [
                        i,
                        R.label("cifar10", int(cls)),
                        round(beam.wavelength, 1),
                        round(beam.line.angle, 3),
                        round(beam.line.bias, 2),
                        round(beam.width, 2),
                    ]
                )
        res.add(
            Table(
                "Laser parameters found",
                [
                    "sample",
                    "new class",
                    "wavelength (nm)",
                    "angle (rad)",
                    "bias",
                    "width",
                ],
                rows,
            )
        )

    return R.evasion_report(clf, "cifar10", x, y, x_adv, extra=extra, examples=len(x))
