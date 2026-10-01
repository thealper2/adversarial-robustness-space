"""Input pre-processing and data-augmentation defences."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.preprocessor import (
    CutMixPyTorch,
    CutoutPyTorch,
    GaussianAugmentation,
    MixupPyTorch,
    SpatialSmoothing,
    TotalVarMin,
)

from .. import data, reporting as R
from ..core import Param, experiment
from ..models import classifier
from ..results import BarChart, ImageGrid, ImageTable, Result

IMAGENET_TARGETS = [(859, "toaster"), (10, "brambling"), (1, "goldfish")]


@experiment(
    id="spatial_smoothing",
    title="Spatial Smoothing",
    category="defence_pre",
    dataset="imagenet",
    models=("resnet50_imagenet",),
    art=("SpatialSmoothing", "ProjectedGradientDescent"),
    notebook="spatial smoothing/ResNet50 Spatial Smoothing.ipynb",
    kind="defence",
    summary="Median filtering of the input. A targeted PGD attack against ResNet-50 is crafted first, then both clean "
    "and adversarial images are smoothed before classification.",
    params=(
        Param.choice(
            "eps",
            "PGD ε (L∞)",
            round(4 / 255, 5),
            [
                (round(2 / 255, 5), "2/255"),
                (round(4 / 255, 5), "4/255"),
                (round(8 / 255, 5), "8/255"),
            ],
        ),
        Param.choice("target", "Target class", 859, IMAGENET_TARGETS),
        Param.int("window", "Median window", 3, 2, 7),
        Param.int("n", "Images", 4, 1, 8),
    ),
    details="Smoothing is applied after the attack (a non-adaptive attacker), as in the notebook.",
    cost="medium",
)
def run_spatial_smoothing(p):
    clf = classifier("resnet50_imagenet")
    x, y = data.split("imagenet", "test", p["n"])
    y_t = np.zeros((len(x), 1000), dtype=np.float32)
    y_t[:, p["target"]] = 1
    pgd = ProjectedGradientDescent(
        clf,
        targeted=True,
        eps=p["eps"],
        eps_step=p["eps"] / 4,
        max_iter=10,
        verbose=False,
    )
    x_adv = pgd.generate(x, y=y_t).astype(np.float32)
    ss = SpatialSmoothing(
        window_size=p["window"], channels_first=True, clip_values=(0.0, 1.0)
    )
    x_s, x_adv_s = ss(x)[0], ss(x_adv)[0]
    probs = {
        k: R.probabilities(clf, v, batch_size=8)
        for k, v in {
            "clean": x,
            "adversarial": x_adv,
            "clean + smoothing": x_s,
            "adversarial + smoothing": x_adv_s,
        }.items()
    }
    res = Result()
    res.metric("Clean top-1", (probs["clean"].argmax(1) == y).mean())
    res.metric(
        "PGD reaches target", (probs["adversarial"].argmax(1) == p["target"]).mean()
    )
    res.metric(
        "Target after smoothing",
        (probs["adversarial + smoothing"].argmax(1) == p["target"]).mean(),
        help="Lower is better for the defence.",
    )
    res.metric(
        "Top-1 after smoothing (adv.)",
        (probs["adversarial + smoothing"].argmax(1) == y).mean(),
    )
    cats = list(probs)
    res.add(
        BarChart(
            "Outcome per input type",
            cats,
            {
                "correct class": [(probs[c].argmax(1) == y).mean() for c in cats],
                "target class": [
                    (probs[c].argmax(1) == p["target"]).mean() for c in cats
                ],
            },
        )
    )
    imgs = np.stack([R.to_display("imagenet", v) for v in (x, x_adv, x_adv_s)], axis=1)
    caps = [
        [
            f"{R.label('imagenet', int(probs[k][i].argmax()))} ({probs[k][i].max():.0%})"
            for k in ("clean", "adversarial", "adversarial + smoothing")
        ]
        for i in range(len(x))
    ]
    res.add(
        ImageTable(
            "Clean · adversarial · adversarial after smoothing",
            ["Clean", "Adversarial", "Smoothed"],
            imgs,
            caps,
        )
    )
    return res


@experiment(
    id="total_variance_minimization",
    title="Total Variance Minimization",
    category="defence_pre",
    dataset="mnist",
    models=("mnist_mlp_small", "mnist_mlp_small_tvm"),
    art=("TotalVarMin", "FastGradientMethod"),
    notebook="total variance minimization/MNIST TVM.ipynb",
    kind="defence",
    summary="Reconstructs each image from a random subset of its pixels while minimising total variation, "
    "removing high-frequency adversarial noise.",
    params=(
        Param.float("eps", "FGSM ε", 0.1, 0.05, 0.3, 0.05),
        Param.int("n", "Test samples", 100, 20, 200, 20),
    ),
    details="The notebook's metric (TVM-trained model on FGSM images) can exceed its *clean* accuracy: FGSM noise "
    "encodes the sign of the true-label gradient and a model trained on such images learns to read it "
    '("label leaking", Kurakin et al. 2017). Denoising at test time (standard model on TVM(FGSM)) is the '
    "fairer view of the defence.",
    cost="medium",
)
def run_tvm(p):
    std, tvm_model = classifier("mnist_mlp_small"), classifier("mnist_mlp_small_tvm")
    x, y = data.split("mnist", "test", p["n"])
    x_adv = FastGradientMethod(std, eps=p["eps"]).generate(x)
    tvm = TotalVarMin(clip_values=(0.0, 1.0), verbose=False)
    x_adv_tvm = tvm(x_adv)[0].astype(np.float32)
    inputs = {"clean": x, "FGSM": x_adv, "TVM(FGSM)": x_adv_tvm}
    series = {
        "standard model": [R.accuracy(std, v, y) for v in inputs.values()],
        "model trained on TVM data": [
            R.accuracy(tvm_model, v, y) for v in inputs.values()
        ],
    }
    res = Result()
    res.metric("Standard on FGSM", series["standard model"][1])
    res.metric(
        "Standard on TVM(FGSM)",
        series["standard model"][2],
        delta=series["standard model"][2] - series["standard model"][1],
    )
    res.metric(
        "TVM-trained on FGSM (notebook metric)", series["model trained on TVM data"][1]
    )
    res.add(BarChart("Accuracy", list(inputs), series))
    k = 8
    imgs = np.stack([R.to_display("mnist", v[:k]) for v in inputs.values()], axis=1)
    pred = {n: R.predict(std, v[:k]) for n, v in inputs.items()}
    res.add(
        ImageTable(
            "Clean · FGSM · TVM-denoised (standard model predictions)",
            list(inputs),
            imgs,
            [[f"pred {pred[n][i]}" for n in inputs] for i in range(k)],
            [[R.status(pred[n][i] == y[i]) for n in inputs] for i in range(k)],
        )
    )
    return res


@experiment(
    id="gaussian_augmentation",
    title="Gaussian Data Augmentation",
    category="defence_pre",
    dataset="mnist",
    models=("mnist_mlp_small", "mnist_mlp_small_gauss"),
    art=("GaussianAugmentation", "FastGradientMethod"),
    notebook="gaussian data augmentation/MNIST GaussianDataAugmentation.ipynb",
    kind="defence",
    summary="Training on Gaussian-noised copies of the data smooths the decision function around each input.",
    params=(
        Param.float("eps", "FGSM ε", 0.1, 0.02, 0.3, 0.02),
        Param.int("n", "Test samples", 500, 100, 1000, 100),
    ),
    details="The notebook crafted FGSM on the standard model and evaluated both models on it (transfer). "
    "The white-box column attacks each model directly — the harder test.",
)
def run_gauss(p):
    std, aug = classifier("mnist_mlp_small"), classifier("mnist_mlp_small_gauss")
    x, y = data.split("mnist", "test", p["n"])
    fgsm = lambda m: FastGradientMethod(m, eps=p["eps"], batch_size=256).generate(x)
    x_std = fgsm(std)
    series = {
        "standard model": [
            R.accuracy(std, x, y),
            R.accuracy(std, x_std, y),
            R.accuracy(std, x_std, y),
        ],
        "Gaussian-augmented": [
            R.accuracy(aug, x, y),
            R.accuracy(aug, x_std, y),
            R.accuracy(aug, fgsm(aug), y),
        ],
    }
    res = Result()
    res.metric("Standard, FGSM", series["standard model"][1])
    res.metric(
        "Augmented, FGSM transfer",
        series["Gaussian-augmented"][1],
        delta=series["Gaussian-augmented"][1] - series["standard model"][1],
    )
    res.metric("Augmented, FGSM white-box", series["Gaussian-augmented"][2])
    res.add(
        BarChart(
            "Accuracy", ["clean", "FGSM from standard model", "FGSM white-box"], series
        )
    )
    noisy = GaussianAugmentation(sigma=0.1, augmentation=False, clip_values=(0, 1))(
        x[:8]
    )[0]
    res.add(
        ImageGrid(
            "Training-time augmentation (σ = 0.1)",
            R.to_display("mnist", np.concatenate([x[:8], noisy])),
            [f"clean {v}" for v in y[:8]] + ["noisy"] * 8,
            columns=8,
        )
    )
    return res


AUGMENTERS = {
    "cutmix": (
        "CutMix",
        "cifar_cnn_cutmix",
        "cut mix/CIFAR10 CutMix.ipynb",
        CutMixPyTorch,
        lambda: CutMixPyTorch(num_classes=10, channels_first=True, device_type="cpu"),
        "Pastes a box from another training image and mixes the labels by area.",
    ),
    "cutout": (
        "Cutout",
        "cifar_cnn_cutout",
        "cut out/CIFAR10 CutOut.ipynb",
        CutoutPyTorch,
        lambda: CutoutPyTorch(length=6, channels_first=True, device_type="cpu"),
        "Masks a random square of each training image.",
    ),
    "mixup": (
        "MixUp",
        "cifar_cnn_mixup",
        "mix up/CIFAR10 MixUp.ipynb",
        MixupPyTorch,
        lambda: MixupPyTorch(num_classes=10, alpha=0.3, num_mix=2, device_type="cpu"),
        "Trains on convex combinations of image pairs and their labels.",
    ),
}


def _augmentation_runner(key):
    name, model_id, _, _, make, _ = AUGMENTERS[key]

    def run(p):
        std, aug = classifier("cifar_cnn"), classifier(model_id)
        x, y = data.split("cifar10", "test", p["n"])
        x_aug = np.asarray(make()(x, data.one_hot(y, 10))[0], dtype=np.float32)
        fgsm = lambda m: FastGradientMethod(
            m, eps=p["eps"] / 255, batch_size=256
        ).generate(x)
        cats = ["clean test", f"{name} test", f"FGSM ε={p['eps']}/255"]
        series = {
            "standard model": [
                R.accuracy(std, x, y),
                R.accuracy(std, x_aug, y),
                R.accuracy(std, fgsm(std), y),
            ],
            f"{name}-trained": [
                R.accuracy(aug, x, y),
                R.accuracy(aug, x_aug, y),
                R.accuracy(aug, fgsm(aug), y),
            ],
        }
        res = Result()
        for model, vals in series.items():
            res.metric(f"{model} · clean", vals[0])
            res.metric(f"{model} · {name} data", vals[1])
        res.add(
            BarChart(
                "Accuracy (the notebook compared the first two groups)", cats, series
            )
        )
        res.add(
            ImageGrid(
                f"{name}-augmented test images",
                R.to_display("cifar10", x_aug[:16]),
                [R.label("cifar10", int(v)) for v in y[:16]],
                columns=8,
            )
        )
        return res

    return run


for _key, (_name, _model, _nb, _cls, _, _summary) in AUGMENTERS.items():
    experiment(
        id=_key,
        title=_name,
        category="defence_pre",
        dataset="cifar10",
        models=("cifar_cnn", _model),
        art=(_cls.__name__,),
        notebook=_nb,
        summary=_summary + " Compared with the standard CNN on clean, augmented "
        "and FGSM test data.",
        kind="defence",
        params=(
            Param.int("eps", "FGSM ε (×1/255)", 4, 1, 16),
            Param.int("n", "Test samples", 500, 100, 1000, 100),
        ),
    )(_augmentation_runner(_key))
