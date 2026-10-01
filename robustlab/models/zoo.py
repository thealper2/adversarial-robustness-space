"""Model catalogue and safe loaders.

Weights are loaded only from pickle-free formats:
  * PyTorch  -> ``models/<id>.safetensors`` (tensors only, no code)
  * sklearn  -> ``models/<id>.skops`` (loaded with an explicit trusted-type allow-list)
  * GPy      -> ``models/<id>.npz`` (numeric arrays; the GP is rebuilt from code)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch import nn

from .. import artifacts, config
from . import architectures as A

Kind = Literal["torch", "detector", "sklearn", "gpy"]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Types that skops may instantiate beyond its built-in safe set.
SKOPS_TRUSTED = frozenset({"sklearn.tree._tree.Tree"})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    dataset: str
    kind: Kind = "torch"
    build: Callable[[], nn.Module] | None = None
    output: Literal["logits", "probs"] = "logits"
    input_shape: tuple[int, ...] = ()
    nb_classes: int = 10
    normalize: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def _mnist(id, title, build, output="logits", **kw):
    return ModelSpec(
        id, title, "mnist", build=build, output=output, input_shape=(1, 28, 28), **kw
    )


def _cifar(id, title, build, output="logits", **kw):
    return ModelSpec(
        id, title, "cifar10", build=build, output=output, input_shape=(3, 32, 32), **kw
    )


_SPECS: list[ModelSpec] = [
    # --- MNIST ---
    _mnist(
        "mnist_mlp",
        "MNIST MLP 64-128-256",
        lambda: A.mnist_mlp((64, 128, 256)),
        description="Dense network used by the FGM, DeepFool, NewtonFool, composite, feature-adversaries and UAP notebooks.",
    ),
    _mnist(
        "mnist_mlp_small",
        "MNIST MLP 128-256 (standard)",
        lambda: A.mnist_mlp((128, 256)),
        description="Baseline for the adversarial-trainer, Gaussian-augmentation and TVM notebooks.",
    ),
    _mnist(
        "mnist_mlp_small_bim_at",
        "MNIST MLP 128-256 (BIM adversarial training)",
        lambda: A.mnist_mlp((128, 256)),
        description="Trained with ART AdversarialTrainer(BIM eps=0.3, ratio=0.5).",
    ),
    _mnist(
        "mnist_mlp_small_gauss",
        "MNIST MLP 128-256 (Gaussian augmentation)",
        lambda: A.mnist_mlp((128, 256)),
        description="Trained on ART GaussianAugmentation(sigma=0.1) noisy images.",
    ),
    _mnist(
        "mnist_mlp_small_tvm",
        "MNIST MLP 128-256 (trained on TVM-denoised FGM data)",
        lambda: A.mnist_mlp((128, 256)),
        description="Trained on TotalVarMin(FGM(x_train)) as in the TVM notebook.",
    ),
    _mnist(
        "mnist_cnn_softmax",
        "MNIST CNN (softmax output)",
        A.mnist_softmax_cnn,
        output="probs",
        description="Two-block CNN with a softmax head, as required by the Virtual Adversarial Method.",
    ),
    _mnist(
        "mnist_cnn_vat_base",
        "MNIST CNN baseline (3k samples)",
        A.mnist_softmax_cnn,
        output="probs",
    ),
    _mnist(
        "mnist_cnn_vat",
        "MNIST CNN with Virtual Adversarial Training",
        A.mnist_softmax_cnn,
        output="probs",
        description="AdversarialTrainer with VirtualAdversarialMethod(eps=5, ratio=0.5).",
    ),
    _mnist(
        "mnist_cnn_strip_backdoor",
        "MNIST CNN with white-square backdoor",
        A.mnist_softmax_cnn,
        output="probs",
        description="10% of the training set carries a white square and label 7.",
    ),
    _mnist(
        "mnist_cert_cnn",
        "MNIST verifier-friendly CNN",
        A.mnist_cert_cnn,
        description="conv(4,s2)-conv(4)-linear network used for DeepZ, IBP and Bullseye Polytope.",
    ),
    _mnist(
        "mnist_victim",
        "MNIST victim MLP (128)",
        lambda: A.mnist_mlp((128,)),
        description="Black-box target of the model-extraction attacks.",
    ),
    _mnist(
        "mnist_copycat",
        "Stolen MLP (CopycatCNN)",
        lambda: A.mnist_mlp((64,)),
        description="Trained only on the victim's answers to 10,000 queries.",
    ),
    _mnist(
        "mnist_knockoff",
        "Stolen MLP (Knockoff Nets)",
        lambda: A.mnist_mlp((64,)),
        description="Trained only on the victim's answers to 10,000 random queries.",
    ),
    ModelSpec(
        "fashion_cnn",
        "Fashion-MNIST CNN",
        "fashion_mnist",
        build=A.fashion_cnn,
        output="probs",
        input_shape=(1, 28, 28),
        description="Target of the MIFace model-inversion attack.",
    ),
    # --- CIFAR-10 ---
    _cifar(
        "cifar_cnn",
        "CIFAR-10 CNN",
        A.cifar_cnn,
        description="conv32x2-conv64x2-dense512 network shared by most CIFAR notebooks.",
    ),
    _cifar(
        "cifar_cnn_labelflip",
        "CIFAR-10 CNN (label-flip poisoned)",
        A.cifar_cnn,
        description="10% of airplane/automobile training labels flipped (activation clustering notebook).",
    ),
    _cifar(
        "cifar_cnn_cutmix",
        "CIFAR-10 CNN (CutMix)",
        A.cifar_cnn,
        description="Trained with ART CutMixPyTorch.",
    ),
    _cifar(
        "cifar_cnn_cutout",
        "CIFAR-10 CNN (Cutout)",
        A.cifar_cnn,
        description="Trained with ART CutoutPyTorch(length=6).",
    ),
    _cifar(
        "cifar_cnn_mixup",
        "CIFAR-10 CNN (MixUp)",
        A.cifar_cnn,
        description="Trained with ART MixupPyTorch(alpha=0.3).",
    ),
    _cifar(
        "cifar_cnn_madry_proxy",
        "CIFAR-10 CNN (Madry PGD proxy)",
        A.cifar_cnn,
        description="Adversarially trained proxy used to craft clean-label poisons.",
    ),
    _cifar(
        "cifar_cnn_cleanlabel_victim",
        "CIFAR-10 CNN (clean-label backdoored)",
        A.cifar_cnn,
        description="Trained on data containing clean-label backdoor poisons (target: deer).",
    ),
    _cifar(
        "cifar_simple_std",
        "SimpleCNN (standard)",
        A.cifar_simple_cnn,
        normalize=(CIFAR_MEAN, CIFAR_STD),
    ),
    _cifar(
        "cifar_simple_fbf",
        "SimpleCNN (Fast-is-better-than-free)",
        A.cifar_simple_cnn,
        normalize=(CIFAR_MEAN, CIFAR_STD),
    ),
    _cifar(
        "cifar_simple_awp",
        "SimpleCNN (AWP)",
        A.cifar_simple_cnn,
        normalize=(CIFAR_MEAN, CIFAR_STD),
    ),
    _cifar(
        "cifar_simple_trades",
        "SimpleCNN (TRADES)",
        A.cifar_simple_cnn,
        normalize=(CIFAR_MEAN, CIFAR_STD),
    ),
    _cifar(
        "cifar_seq_substitute",
        "CIFAR-10 CNN substitute (10k samples)",
        lambda: A.cifar_cnn(hidden=256),
        description="Clean model used to craft Sleeper Agent / Witches' Brew poisons.",
    ),
    _cifar(
        "cifar_seq_sleeper_victim",
        "Victim trained on Sleeper Agent poisons",
        lambda: A.cifar_cnn(hidden=256),
    ),
    _cifar(
        "cifar_seq_brew_victim",
        "Victim trained on Witches' Brew poisons",
        lambda: A.cifar_cnn(hidden=256),
    ),
    _cifar(
        "cifar_small_safe",
        "CIFAR-10 small CNN (clean)",
        A.cifar_small_cnn,
        output="probs",
    ),
    ModelSpec(
        "cifar_trigger_detector",
        "Trigger detector (neural payload)",
        "cifar10",
        build=lambda: A.cifar_small_cnn(outputs=1),
        output="probs",
        input_shape=(3, 32, 32),
        nb_classes=1,
    ),
    # --- ImageNet ---
    ModelSpec(
        "resnet50_imagenet",
        "ResNet-50 (ImageNet, torchvision V2 weights)",
        "imagenet",
        build=A.resnet50_imagenet,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        normalize=(IMAGENET_MEAN, IMAGENET_STD),
    ),
    # --- Object detection (BadDet) ---
    *[
        ModelSpec(
            f"baddet_{v}",
            f"Faster R-CNN backdoored with BadDet {v.upper()}",
            "shapes",
            kind="detector",
            build=A.shapes_detector,
            input_shape=(3, 128, 128),
            nb_classes=4,
        )
        for v in ("gma", "oda", "oga", "rma")
    ],
    # --- scikit-learn / GPy ---
    ModelSpec(
        "digits_tree",
        "Decision tree (digits)",
        "digits",
        kind="sklearn",
        input_shape=(64,),
    ),
    ModelSpec(
        "mnist_logreg",
        "Logistic regression (MNIST pixels)",
        "mnist",
        kind="sklearn",
        input_shape=(784,),
    ),
    ModelSpec(
        "iris_logreg_scaled",
        "Logistic regression (standardized Iris)",
        "iris",
        kind="sklearn",
        input_shape=(4,),
        nb_classes=3,
    ),
    ModelSpec(
        "iris_logreg",
        "Logistic regression (Iris)",
        "iris",
        kind="sklearn",
        input_shape=(4,),
        nb_classes=3,
    ),
    ModelSpec(
        "iris_dp_logreg",
        "DP logistic regression (Iris, diffprivlib eps=1)",
        "iris",
        kind="sklearn",
        input_shape=(4,),
        nb_classes=3,
        description="Trained with diffprivlib and exported as plain sklearn coefficients.",
    ),
    ModelSpec(
        "synthetic_tree",
        "Decision tree (synthetic tabular)",
        "synthetic",
        kind="sklearn",
        input_shape=(10,),
        nb_classes=2,
    ),
    ModelSpec(
        "breast_cancer_gpc",
        "GP classifier, RBF kernel (breast cancer)",
        "breast_cancer",
        kind="gpy",
        input_shape=(30,),
        nb_classes=2,
    ),
]

MODELS: dict[str, ModelSpec] = {s.id: s for s in _SPECS}


def spec(model_id: str) -> ModelSpec:
    return MODELS[model_id]


def weights_path(model_id: str) -> str:
    ext = {
        "torch": "safetensors",
        "detector": "safetensors",
        "sklearn": "skops",
        "gpy": "npz",
    }[spec(model_id).kind]
    return f"models/{model_id}.{ext}"


@lru_cache(maxsize=None)
def meta(model_id: str) -> dict[str, Any]:
    """Training metadata written next to the weights (accuracy, recipe, date)."""
    rel = f"models/{model_id}.json"
    return artifacts.read_json(rel) if artifacts.exists(rel) else {}


def art_device() -> str:
    return "gpu" if config.DEVICE.startswith("cuda") else "cpu"


def torch_device() -> torch.device:
    return torch.device(config.DEVICE)


@lru_cache(maxsize=None)
def torch_model(model_id: str) -> nn.Module:
    from safetensors.torch import load_file

    s = spec(model_id)
    if s.kind not in ("torch", "detector"):
        raise TypeError(f"{model_id} is not a PyTorch model")
    model = s.build()
    state = load_file(str(artifacts.path(weights_path(model_id))))
    model.load_state_dict(state, strict=True)
    return model.eval().to(torch_device())


class ProbabilityCrossEntropyLoss(nn.NLLLoss):
    """Cross-entropy for models whose last layer already applies softmax."""

    def forward(
        self, probabilities: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(torch.log(probabilities + 1e-12), target)


def loss_for(s: ModelSpec) -> nn.Module:
    return (
        nn.CrossEntropyLoss() if s.output == "logits" else ProbabilityCrossEntropyLoss()
    )


@lru_cache(maxsize=1)
def _float32_classifier_cls():
    from art.estimators.classification import PyTorchClassifier

    class Float32PyTorchClassifier(PyTorchClassifier):
        """Decision-based attacks (Boundary, HopSkipJump) query with float64 arrays; the model expects float32."""

        def predict(
            self, x, batch_size: int = 128, training_mode: bool = False, **kwargs
        ):
            return super().predict(
                np.asarray(x, dtype=np.float32),
                batch_size=batch_size,
                training_mode=training_mode,
                **kwargs,
            )

    return Float32PyTorchClassifier


def wrap_pytorch(model: nn.Module, s: ModelSpec, **overrides):
    """Create an ART ``PyTorchClassifier`` for an (optionally wrapped) torch module."""
    PyTorchClassifier = _float32_classifier_cls()

    kwargs = dict(
        model=model,
        loss=loss_for(s),
        input_shape=s.input_shape,
        nb_classes=s.nb_classes,
        clip_values=(0.0, 1.0),
        device_type=art_device(),
    )
    if s.normalize is not None:
        # Full (C, H, W) arrays: some ART trainers (e.g. FBF) do not broadcast (C, 1, 1) statistics.
        mean, std = (
            np.broadcast_to(
                np.asarray(v, dtype=np.float32).reshape(-1, 1, 1), s.input_shape
            ).copy()
            for v in s.normalize
        )
        kwargs["preprocessing"] = (mean, std)
    kwargs.update(overrides)
    return PyTorchClassifier(**kwargs)


@lru_cache(maxsize=None)
def classifier(model_id: str):
    """Cached ART estimator for a catalogue model (PyTorch, scikit-learn or GPy)."""
    s = spec(model_id)
    if s.kind == "torch":
        return wrap_pytorch(torch_model(model_id), s)
    if s.kind == "sklearn":
        from art.estimators.classification.scikitlearn import SklearnClassifier

        return SklearnClassifier(model=sklearn_model(model_id))
    if s.kind == "gpy":
        from art.estimators.classification import GPyGaussianProcessClassifier

        return GPyGaussianProcessClassifier(gpy_model(model_id))
    raise TypeError(f"No ART classifier wrapper for {model_id} ({s.kind})")


@lru_cache(maxsize=None)
def probability_classifier(model_id: str):
    """ART estimator with a softmax appended, for attacks that require probabilities (e.g. SimBA)."""
    from dataclasses import replace

    s = spec(model_id)
    if s.output == "probs":
        return classifier(model_id)
    return wrap_pytorch(
        nn.Sequential(torch_model(model_id), nn.Softmax(dim=1)).eval(),
        replace(s, output="probs"),
    )


@lru_cache(maxsize=None)
def sklearn_model(model_id: str):
    from skops.io import get_untrusted_types, load

    file = str(artifacts.path(weights_path(model_id)))
    untrusted = set(get_untrusted_types(file=file))
    unexpected = untrusted - SKOPS_TRUSTED
    if unexpected:
        raise ValueError(
            f"Refusing to load {model_id}: unexpected types {sorted(unexpected)}"
        )
    return load(file, trusted=sorted(untrusted))


@lru_cache(maxsize=None)
def gpy_model(model_id: str):
    """Rebuild the GP classifier from stored data and fixed, pre-optimised hyper-parameters.

    ``GPy``'s own ``from_dict`` evaluates class names found in the file, so it is not used.
    """
    import GPy

    with np.load(artifacts.path(weights_path(model_id)), allow_pickle=False) as npz:
        x, y = npz["x"], npz["y"]
        variance, lengthscale = float(npz["variance"]), float(npz["lengthscale"])
    kernel = GPy.kern.RBF(
        input_dim=x.shape[1], variance=variance, lengthscale=lengthscale
    )
    return GPy.models.GPClassification(x, y.reshape(-1, 1), kernel=kernel)
