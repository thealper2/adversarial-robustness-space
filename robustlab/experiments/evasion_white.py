"""White-box (gradient based) evasion attacks."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import (
    AutoConjugateGradient,
    BasicIterativeMethod,
    CarliniL0Method,
    CarliniL2Method,
    CarliniLInfMethod,
    DeepFool,
    FastGradientMethod,
    FeatureAdversariesPyTorch,
    FrameSaliencyAttack,
    NewtonFool,
    ProjectedGradientDescent,
    SaliencyMapMethod,
    SpatialTransformation,
    VirtualAdversarialMethod,
    Wasserstein,
)

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier
from ..results import BarChart, ImageGrid, ImageTable, Markdown

N_MNIST = Param.int("n", "Test samples", 100, 20, 500, 20)
N_CIFAR_SMALL = Param.int("n", "Test samples", 4, 1, 16)
N_CIFAR = Param.int("n", "Test samples", 16, 4, 64, 4)
CIFAR_TARGET = Param.choice(
    "target",
    "Target class",
    -1,
    [(-1, "untargeted")] + list(enumerate(data.CIFAR10_CLASSES)),
)


def _targets(
    target: int, n: int, nb_classes: int = 10
) -> tuple[np.ndarray | None, int | None]:
    if target < 0:
        return None, None
    return data.one_hot(np.full(n, target), nb_classes), target


# --- MNIST MLP attacks ------------------------------------------------------------------------


@experiment(
    id="fgm",
    title="Fast Gradient Method (FGSM)",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("FastGradientMethod",),
    notebook="fast gradient method/MNIST FGM.ipynb",
    summary="One signed-gradient step of size ε. The fastest white-box attack and the basis of many others.",
    params=(Param.float("eps", "ε (L∞)", 0.1, 0.01, 0.5, 0.01), N_MNIST),
)
def run_fgm(p):
    clf = classifier("mnist_mlp")
    x, y = data.split("mnist", "test", p["n"])
    x_adv = FastGradientMethod(clf, eps=p["eps"], batch_size=256).generate(x)

    def sweep(res):
        eps_grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        accs = [
            R.accuracy(
                clf,
                FastGradientMethod(clf, eps=e, batch_size=256).generate(x) if e else x,
                y,
            )
            for e in eps_grid
        ]
        res.add(
            R.accuracy_curve(
                "Accuracy vs ε (same samples)", eps_grid, {"FGSM": accs}, "ε"
            )
        )

    return R.evasion_report(clf, "mnist", x, y, x_adv, extra=sweep)


@experiment(
    id="deepfool",
    title="DeepFool",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("DeepFool",),
    notebook="deep fool/MNIST DeepFool.ipynb",
    summary="Iteratively linearises the classifier and steps to the closest decision boundary (minimal L2 perturbation).",
    params=(
        Param.choice(
            "epsilon",
            "Overshoot",
            1e-6,
            [(1e-6, "1e-6 (default)"), (0.02, "0.02"), (0.1, "0.1")],
        ),
        N_MNIST,
    ),
)
def run_deepfool(p):
    clf = classifier("mnist_mlp")
    x, y = data.split("mnist", "test", p["n"])
    x_adv = DeepFool(clf, epsilon=p["epsilon"], batch_size=64, verbose=False).generate(
        x
    )
    return R.evasion_report(clf, "mnist", x, y, x_adv)


@experiment(
    id="newtonfool",
    title="NewtonFool",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("NewtonFool",),
    notebook="newton fool/MNIST NewtonFool.ipynb",
    summary="Newton-style steps that lower the probability of the true class as fast as possible.",
    params=(
        Param.int("max_iter", "Iterations", 100, 10, 200, 10),
        Param.float("eta", "η (step strength)", 0.01, 0.005, 0.1, 0.005),
        N_MNIST,
    ),
)
def run_newtonfool(p):
    clf = classifier("mnist_mlp")
    x, y = data.split("mnist", "test", p["n"])
    x_adv = NewtonFool(
        clf, max_iter=p["max_iter"], eta=p["eta"], batch_size=64, verbose=False
    ).generate(x)
    return R.evasion_report(clf, "mnist", x, y, x_adv)


@experiment(
    id="composite",
    title="Composite attack: FGSM → PGD",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("FastGradientMethod", "ProjectedGradientDescent"),
    notebook="composite adversarial attack/MNIST FGSM + PGD.ipynb",
    summary="Chains two attacks as in the notebook: PGD starts from the FGSM output and searches another ε-ball around it, so the total L∞ change can reach 2ε.",
    params=(
        Param.float("eps", "ε (L∞)", 0.1, 0.02, 0.3, 0.01),
        Param.int("max_iter", "PGD iterations", 40, 5, 100, 5),
        N_MNIST,
    ),
)
def run_composite(p):
    clf = classifier("mnist_mlp")
    x, y = data.split("mnist", "test", p["n"])
    x_fgsm = FastGradientMethod(clf, eps=p["eps"], batch_size=256).generate(x)
    pgd = ProjectedGradientDescent(
        clf, eps=p["eps"], eps_step=p["eps"] / 10, max_iter=p["max_iter"], verbose=False
    )
    x_adv = pgd.generate(x_fgsm)

    def extra(res):
        res.add(
            BarChart(
                "Accuracy per stage",
                ["clean", "FGSM", "FGSM → PGD"],
                {
                    "accuracy": [
                        R.accuracy(clf, x, y),
                        R.accuracy(clf, x_fgsm, y),
                        R.accuracy(clf, x_adv, y),
                    ]
                },
            )
        )

    return R.evasion_report(clf, "mnist", x, y, x_adv, extra=extra)


@experiment(
    id="feature_adversaries",
    title="Feature Adversaries",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("FeatureAdversariesPyTorch",),
    notebook="feature adversaries/MNIST Feature Adversaries.ipynb",
    summary="Perturbs a source image (within δ) so that its internal representation matches a different guide image.",
    params=(
        Param.int("delta", "δ (L∞ budget, ×1/255)", 25, 5, 75, 5),
        Param.int("n", "Source/guide pairs", 32, 8, 64, 8),
    ),
)
def run_feature_adversaries(p):
    clf = classifier("mnist_mlp")
    x, y = data.split("mnist", "test", 200)
    valid = np.where(y[:100] != y[100:200])[0][: p["n"]]
    source, guide = x[:100][valid], x[100:200][valid]
    y_src, y_guide = y[:100][valid], y[100:200][valid]
    attack = FeatureAdversariesPyTorch(
        clf,
        delta=p["delta"] / 255,
        layer=-2,
        step_size=1 / 255,
        max_iter=100,
        verbose=False,
    )
    x_adv = attack.generate(source, guide)

    def extra(res):
        pred = R.predict(clf, x_adv)
        res.metric("Classified as guide label", (pred == y_guide).mean())
        k = min(6, len(source))
        imgs = np.stack(
            [R.to_display("mnist", a) for a in (source[:k], guide[:k], x_adv[:k])],
            axis=1,
        )
        caps = [
            [
                f"source {y_src[i]}",
                f"guide {y_guide[i]}",
                f"adversarial → pred {pred[i]}",
            ]
            for i in range(k)
        ]
        res.blocks.insert(
            1,
            ImageTable(
                "Source · guide · adversarial",
                ["Source", "Guide", "Adversarial"],
                imgs,
                caps,
            ),
        )

    return R.evasion_report(clf, "mnist", source, y_src, x_adv, extra=extra)


@experiment(
    id="vam",
    title="Virtual Adversarial Method",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_cnn_softmax",),
    art=("VirtualAdversarialMethod",),
    notebook="virtual adversarial method/MNIST Virtual Adversarial Method.ipynb",
    summary="Label-free attack: finds the direction that maximises the KL divergence of the output distribution.",
    params=(
        Param.float("eps", "ε (L2)", 5.0, 0.5, 10.0, 0.5),
        Param.int("n", "Test samples", 200, 50, 500, 50),
    ),
    details="ART's VAM estimates the gradient with one finite difference per input pixel (≈5 minutes of CPU for the "
    "ε sweep), so the Space shows the precomputed run.",
    cost="slow",
    live=False,
)
def run_vam(p):
    clf = classifier("mnist_cnn_softmax")
    x, y = data.split("mnist", "test", p["n"])

    def vam(eps):
        return VirtualAdversarialMethod(
            clf, max_iter=5, finite_diff=1e-6, eps=eps, batch_size=64, verbose=False
        ).generate(x)

    x_adv = vam(p["eps"])

    def extra(res):
        pc, pa = R.probabilities(clf, x), R.probabilities(clf, x_adv)
        kl = np.sum(pc * (np.log(pc + 1e-12) - np.log(pa + 1e-12)), axis=1)
        res.metric("Mean KL divergence", float(kl.mean()), fmt="float")
        grid = [0.5, 2.0, 5.0, 10.0]
        res.add(
            R.accuracy_curve(
                "Accuracy vs ε",
                grid,
                {"VAM": [R.accuracy(clf, vam(e), y) for e in grid]},
                "ε (L2)",
            )
        )

    return R.evasion_report(clf, "mnist", x, y, x_adv, extra=extra)


# --- CIFAR-10 CNN attacks ---------------------------------------------------------------------


@experiment(
    id="bim",
    title="Basic Iterative Method",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("BasicIterativeMethod",),
    notebook="basic iterative method/MNIST BIM.ipynb",
    summary="Repeated small FGSM steps, clipped to the ε-ball after each step. Supports targeted mode.",
    params=(
        Param.float("eps", "ε (L∞)", 0.3, 0.005, 0.3, 0.005),
        Param.float("eps_step", "Step size", 0.01, 0.001, 0.05, 0.001),
        Param.int("max_iter", "Iterations", 10, 1, 50),
        CIFAR_TARGET,
        Param.int("n", "Test samples", 100, 20, 200, 20),
    ),
)
def run_bim(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    y_t, t = _targets(p["target"], len(x))
    attack = BasicIterativeMethod(
        clf,
        eps=p["eps"],
        eps_step=p["eps_step"],
        max_iter=p["max_iter"],
        targeted=t is not None,
        verbose=False,
    )
    return R.evasion_report(clf, "cifar10", x, y, attack.generate(x, y=y_t), target=t)


def _cw(attack_cls, p, **kw):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    y_t, t = _targets(p["target"], len(x))
    attack = attack_cls(
        clf,
        confidence=p["confidence"],
        targeted=t is not None,
        learning_rate=0.01,
        max_iter=p["max_iter"],
        batch_size=1,
        verbose=False,
        **kw,
    )
    return R.evasion_report(
        clf,
        "cifar10",
        x,
        y,
        attack.generate(x, y=y_t),
        target=t,
        examples=min(6, len(x)),
    )


CW_PARAMS = (
    Param.float("confidence", "Confidence κ", 0.1, 0.0, 5.0, 0.1),
    Param.int("max_iter", "Iterations", 10, 5, 50, 5),
    CIFAR_TARGET,
    N_CIFAR_SMALL,
)


@experiment(
    id="cw_l2",
    title="Carlini & Wagner L2",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("CarliniL2Method",),
    notebook="carlini and wagner l2 method/CIFAR10 C&W L2.ipynb",
    summary="Optimisation attack minimising ‖δ‖₂ + c·f(x+δ) with a tanh change of variables.",
    params=CW_PARAMS,
    cost="medium",
)
def run_cw_l2(p):
    return _cw(CarliniL2Method, p)


@experiment(
    id="cw_linf",
    title="Carlini & Wagner L∞",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("CarliniLInfMethod",),
    notebook="carlini and wagner l inf method/CIFAR10 C&W L inf.ipynb",
    summary="C&W variant that shrinks an L∞ bound τ until the attack stops succeeding.",
    params=CW_PARAMS,
    cost="medium",
)
def run_cw_linf(p):
    return _cw(CarliniLInfMethod, p)


@experiment(
    id="cw_l0",
    title="Carlini & Wagner L0",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("CarliniL0Method",),
    notebook="carlini and wagner l0 method/CIFAR10 C&W L0.ipynb",
    summary="Repeatedly runs C&W L2 and freezes the least important pixels to minimise how many pixels change.",
    params=CW_PARAMS,
    cost="medium",
)
def run_cw_l0(p):
    return _cw(CarliniL0Method, p)


@experiment(
    id="jsma",
    title="Jacobian Saliency Map Attack",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("SaliencyMapMethod",),
    notebook="jacobian saliency map method/CIFAR10 JSMA.ipynb",
    summary="Uses the Jacobian to pick the most salient pixel pairs and saturates them (L0 attack).",
    params=(
        Param.float("theta", "θ (change per feature)", 1.0, 0.1, 1.0, 0.1),
        Param.float("gamma", "γ (max fraction changed)", 0.1, 0.01, 0.3, 0.01),
        N_CIFAR,
    ),
)
def run_jsma(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    x_adv = SaliencyMapMethod(
        clf, theta=p["theta"], gamma=p["gamma"], batch_size=16, verbose=False
    ).generate(x)
    return R.evasion_report(clf, "cifar10", x, y, x_adv)


@experiment(
    id="wasserstein",
    title="Wasserstein attack",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("Wasserstein",),
    notebook="wasserstein attack/CIFAR10 Wasserstein.ipynb",
    summary="Projected Sinkhorn iterations inside a Wasserstein ball: moves pixel mass instead of adding noise.",
    params=(
        Param.float("eps", "ε (Wasserstein)", 0.7, 0.1, 1.0, 0.1),
        Param.int("n", "Test samples", 8, 2, 16, 2),
    ),
    details="With ART's default regularization (3000, as in the notebook) the conjugate Sinkhorn step overflows; the "
    "settings from ART's Wasserstein example (regularization 100, 5 Sinkhorn iterations) are used instead.",
)
def run_wasserstein(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    attack = Wasserstein(
        clf,
        targeted=False,
        eps=p["eps"],
        regularization=100,
        conjugate_sinkhorn_max_iter=5,
        projected_sinkhorn_max_iter=5,
        eps_iter=2,
        max_iter=100,
        batch_size=1,
        verbose=False,
    )
    x_adv = attack.generate(x.astype(np.float64)).astype(np.float32)
    return R.evasion_report(clf, "cifar10", x, y, x_adv, examples=min(6, len(x)))


ACG_PRESETS = {
    "l1": (1, 0.6, 0.1, "L1, ε=0.6 (notebook)"),
    "l2": (2, 0.5, 0.1, "L2, ε=0.5"),
    "linf": (np.inf, 8 / 255, 2 / 255, "L∞, ε=8/255"),
}


@experiment(
    id="auto_cg",
    title="Auto Conjugate Gradient",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("AutoConjugateGradient",),
    notebook="conjugate gradient/CIFAR10 Conjugate Gradient.ipynb",
    summary="Step-size-free conjugate-gradient attack with restarts (a stronger sibling of Auto-PGD).",
    params=(
        Param.choice(
            "preset",
            "Norm / budget",
            "linf",
            [(k, v[3]) for k, v in ACG_PRESETS.items()],
        ),
        N_CIFAR,
    ),
    details="The notebook used an L1 budget of 0.6, which is too small to change any CIFAR-10 prediction; "
    "the L∞ preset shows the attack working.",
    cost="medium",
)
def run_auto_cg(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    norm, eps, step, _ = ACG_PRESETS[p["preset"]]
    attack = AutoConjugateGradient(
        clf,
        norm=norm,
        eps=eps,
        eps_step=step,
        max_iter=25,
        nb_random_init=1,
        loss_type="cross_entropy",
        verbose=False,
    )
    return R.evasion_report(
        clf, "cifar10", x, y, attack.generate(x, y=data.one_hot(y, 10))
    )


@experiment(
    id="spatial_transformation",
    title="Spatial Transformation",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("SpatialTransformation",),
    notebook="spatial transformation attack/CIFAR10 Spatial Transformation.ipynb",
    summary="Grid search over translations and rotations: no pixel noise, only a geometric change.",
    params=(
        Param.float("max_translation", "Max translation (%)", 15.0, 0.0, 30.0, 5.0),
        Param.float("max_rotation", "Max rotation (°)", 25.0, 0.0, 45.0, 5.0),
        Param.int("num", "Grid points per axis", 1, 1, 5),
        N_CIFAR,
    ),
)
def run_spatial(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    attack = SpatialTransformation(
        clf,
        max_translation=p["max_translation"],
        num_translations=p["num"],
        max_rotation=p["max_rotation"],
        num_rotations=p["num"],
        verbose=False,
    )
    x_adv = attack.generate(x)

    def extra(res):
        res.metric(
            "Chosen transform",
            f"shift {attack.attack_trans_x}/{attack.attack_trans_y} px, rot {attack.attack_rot}°",
            fmt="text",
        )

    return R.evasion_report(clf, "cifar10", x, y, x_adv, extra=extra)


@experiment(
    id="frame_saliency",
    title="Frame Saliency Attack",
    category="evasion_white",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("FrameSaliencyAttack", "FastGradientMethod"),
    notebook="frame saliency attack/CIFAR10 Frame Saliency Attack.ipynb",
    summary="Wraps FGSM and only perturbs the most salient 'frames' (here: image rows) until the label flips.",
    params=(
        Param.choice(
            "method",
            "Method",
            "iterative_saliency",
            ["iterative_saliency", "iterative_saliency_refresh", "one_shot"],
        ),
        Param.float("eps", "FGSM ε", 0.4, 0.05, 0.5, 0.05),
        Param.int("n", "Test samples", 32, 8, 100, 4),
    ),
    details="The notebook's TensorFlow model was NHWC with `frame_index=1` (rows). In PyTorch (NCHW) the rows are axis 2.",
    cost="medium",
)
def run_frame_saliency(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test", p["n"])
    fgsm = FastGradientMethod(clf, eps=p["eps"])
    attack = FrameSaliencyAttack(
        clf,
        attacker=fgsm,
        method=p["method"],
        frame_index=2,
        batch_size=16,
        verbose=False,
    )
    x_adv = attack.generate(x)

    def extra(res):
        rows_changed = (np.abs(x_adv - x).max(axis=(1, 3)) > 1e-6).sum(axis=1)
        res.metric(
            "Rows perturbed (mean of 32)", float(rows_changed.mean()), fmt="float"
        )

    return R.evasion_report(clf, "cifar10", x, y, x_adv, extra=extra)


# --- scikit-learn target ----------------------------------------------------------------------


@experiment(
    id="pgd",
    title="Projected Gradient Descent",
    category="evasion_white",
    dataset="mnist",
    models=("mnist_logreg",),
    art=("ProjectedGradientDescent", "SklearnClassifier"),
    notebook="projected gradient descent/MNIST PDG.ipynb",
    summary="Multi-step FGSM with projection onto the ε-ball — here against a scikit-learn logistic regression.",
    params=(
        Param.float("eps", "ε (L∞)", 0.3, 0.02, 0.5, 0.01),
        Param.float("eps_step", "Step size", 0.1, 0.01, 0.2, 0.01),
        Param.int("max_iter", "Iterations", 40, 5, 100, 5),
        N_MNIST,
    ),
)
def run_pgd(p):
    clf = classifier("mnist_logreg")
    x, y = data.split("mnist", "test", p["n"])
    x = x.reshape(len(x), -1)
    attack = ProjectedGradientDescent(
        clf, eps=p["eps"], eps_step=p["eps_step"], max_iter=p["max_iter"], verbose=False
    )
    x_adv = attack.generate(x)

    def extra(res):
        w = clf.model.coef_.reshape(10, 28, 28, 1)
        w = (w - w.min()) / (w.max() - w.min())
        res.add(
            Markdown(
                "Logistic-regression weights per class — PGD pushes pixels along these templates."
            )
        )
        res.add(
            ImageGrid(
                "Class weight templates",
                w.astype(np.float32),
                [f"class {i}" for i in range(10)],
                columns=10,
            )
        )

    return R.evasion_report(clf, "mnist", x, y, x_adv, extra=extra)
