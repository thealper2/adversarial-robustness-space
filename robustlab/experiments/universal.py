"""Input-agnostic perturbations and physical patches."""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import (
    AdversarialPatchPyTorch,
    TargetedUniversalPerturbation,
    UniversalPerturbation,
)

from .. import artifacts, data, reporting as R
from ..core import Param, experiment
from ..models import classifier
from ..results import ImageGrid, ImageTable, Result

UAP_EPS = Param.choice(
    "eps",
    "ε (L∞ projection)",
    0.3,
    [
        (0.1, "0.1"),
        (0.2, "0.2"),
        (0.3, "0.3"),
        (10.0, "10 (notebook default – effectively unbounded)"),
    ],
)


def _uap_view(noise: np.ndarray) -> ImageGrid:
    view = R.perturbation_view(noise.reshape(1, 1, 28, 28))
    return ImageGrid(
        "The universal perturbation (amplified)",
        R.to_display("mnist", view),
        ["same noise for every input"],
        columns=1,
    )


@experiment(
    id="universal_perturbation",
    title="Universal Adversarial Perturbation",
    category="universal",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("UniversalPerturbation",),
    notebook="universal perturbation/MNIST Universal Perturbation.ipynb",
    summary="Builds a single noise pattern on one set of images that fools the model on unseen images too.",
    params=(
        UAP_EPS,
        Param.int("max_iter", "Passes over the fit set", 5, 1, 10),
        Param.int("n_fit", "Fit images (train bank)", 500, 100, 1000, 100),
    ),
    details="The perturbation is computed on training-bank images and evaluated on 1,000 separate test images "
    "(the notebook computed and evaluated on the same test set).",
)
def run_uap(p):
    clf = classifier("mnist_mlp")
    x_fit, _ = data.split("mnist", "train", p["n_fit"])
    x, y = data.split("mnist", "test", 1000)
    attack = UniversalPerturbation(
        clf,
        attacker="fgsm",
        max_iter=p["max_iter"],
        eps=p["eps"],
        batch_size=128,
        verbose=False,
    )
    attack.generate(x_fit)
    x_adv = np.clip(x + attack.noise, 0.0, 1.0).astype(np.float32)

    def extra(res):
        res.metric("Fooling rate on fit set", float(attack.fooling_rate))
        res.blocks.insert(0, _uap_view(attack.noise))

    return R.evasion_report(clf, "mnist", x, y, x_adv, extra=extra)


@experiment(
    id="targeted_universal_perturbation",
    title="Targeted Universal Perturbation",
    category="universal",
    dataset="mnist",
    models=("mnist_mlp",),
    art=("TargetedUniversalPerturbation",),
    notebook="targeted universal perturbation/MNIST Targeted Universal Perturbation.ipynb",
    summary="One noise pattern that pushes every image of a source digit towards a chosen target digit.",
    params=(
        Param.choice("source", "Source digit", 5, list(range(10))),
        Param.choice("target", "Target digit", 6, list(range(10))),
        UAP_EPS,
        Param.int("max_iter", "Passes", 5, 1, 10),
    ),
    details="The notebook paired the source images with unrelated labels and left ART's inner FGSM untargeted, so the "
    "perturbation never grew. Here every source image gets the target label and the inner FGSM is targeted.",
)
def run_tuap(p):
    clf = classifier("mnist_mlp")
    x_all, y_all = data.split("mnist", "test")
    x = x_all[y_all == p["source"]][:300]
    y = np.full(len(x), p["source"])
    # ART does not switch the inner FGSM to targeted mode by itself; without this the noise never updates.
    inner = {
        "targeted": True,
        "minimal": True,
        "eps": max(p["eps"], 0.5),
        "eps_step": 0.02,
    }
    attack = TargetedUniversalPerturbation(
        clf,
        attacker="fgsm",
        attacker_params=inner,
        delta=0.0,
        max_iter=p["max_iter"],
        eps=p["eps"],
    )
    attack.generate(x, y=data.one_hot(np.full(len(x), p["target"]), 10))
    x_adv = np.clip(x + attack.noise, 0.0, 1.0).astype(np.float32)

    def extra(res):
        res.blocks.insert(0, _uap_view(attack.noise))

    return R.evasion_report(clf, "mnist", x, y, x_adv, target=p["target"], extra=extra)


PATCH_FILE = "extras/adversarial_patch_resnet50.npz"


@experiment(
    id="adversarial_patch",
    title="Adversarial Patch",
    category="universal",
    dataset="imagenet",
    models=("resnet50_imagenet",),
    art=("AdversarialPatchPyTorch",),
    notebook="adversarial patch/ResNet50 Adversarial Patch.ipynb",
    summary="A printable, image-independent circular patch optimised (offline, 500 iterations) to make ResNet-50 "
    "predict 'toaster'. Here it is pasted onto new images at different scales.",
    params=(
        Param.float("scale", "Patch scale", 0.5, 0.2, 0.9, 0.05),
        Param.int("n", "Images", 8, 4, 16, 4),
    ),
    extra_artifacts=(PATCH_FILE,),
)
def run_patch(p):
    clf = classifier("resnet50_imagenet")
    with np.load(artifacts.path(PATCH_FILE), allow_pickle=False) as npz:
        patch, target = npz["patch"], int(npz["target"])
        trained_on = int(npz["train_images"]) if "train_images" in npz.files else 0
    x, y = data.split("imagenet", "test", p["n"])
    attack = AdversarialPatchPyTorch(
        clf, patch_shape=patch.shape, patch_type="circle", verbose=False
    )
    x_patched = attack.apply_patch(x, scale=p["scale"], patch_external=patch).astype(
        np.float32
    )

    pc, pp = R.probabilities(clf, x), R.probabilities(clf, x_patched)
    res = Result()
    res.metric("Clean top-1 accuracy", (pc.argmax(1) == y).mean())
    res.metric(
        "Patched → 'toaster'",
        (pp.argmax(1) == target).mean(),
        help="Targeted success rate",
    )
    res.metric("Mean 'toaster' probability", float(pp[:, target].mean()))
    res.metric(
        "Patch trained on",
        f"{trained_on} images" if trained_on else "offline set",
        fmt="text",
    )
    res.add(
        ImageGrid(
            "The patch",
            R.to_display("imagenet", patch[None]),
            [f"target: {R.label('imagenet', target)}"],
            columns=1,
        )
    )
    imgs = np.stack(
        [R.to_display("imagenet", x), R.to_display("imagenet", x_patched)], axis=1
    )
    caps = [
        [
            f"{R.label('imagenet', int(pc[i].argmax()))} ({pc[i].max():.0%})",
            f"{R.label('imagenet', int(pp[i].argmax()))} ({pp[i].max():.0%})",
        ]
        for i in range(len(x))
    ]
    states = [
        [R.status(pc[i].argmax() == y[i]), R.status(pp[i].argmax() != target)]
        for i in range(len(x))
    ]
    res.add(
        ImageTable(
            "Clean vs patched (red = model says toaster)",
            ["Clean", "Patched"],
            imgs,
            caps,
            states,
        )
    )
    return res
