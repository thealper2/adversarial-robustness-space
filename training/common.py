"""Shared helpers for the local (GPU) training pipeline. Never imported by the Space."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
CACHE = Path(os.environ.get("ROBUSTLAB_CACHE", Path.home() / ".cache" / "robustlab"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ART_DEVICE = "gpu" if DEVICE.type == "cuda" else "cpu"


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Full datasets (training only) --------------------------------------------------------------


@lru_cache(maxsize=None)
def mnist():
    """(x_train, y_train, x_test, y_test) as float32 NCHW in [0, 1] with integer labels."""
    from art.utils import load_mnist

    (xtr, ytr), (xte, yte), _, _ = load_mnist()
    return _nchw(xtr), ytr.argmax(1), _nchw(xte), yte.argmax(1)


@lru_cache(maxsize=None)
def cifar10():
    from art.utils import load_cifar10

    (xtr, ytr), (xte, yte), _, _ = load_cifar10()
    return _nchw(xtr), ytr.argmax(1), _nchw(xte), yte.argmax(1)


@lru_cache(maxsize=None)
def fashion_mnist():
    from torchvision.datasets import FashionMNIST

    out = []
    for train in (True, False):
        ds = FashionMNIST(CACHE / "torchvision", train=train, download=True)
        out += [ds.data.numpy()[:, None].astype(np.float32) / 255.0, ds.targets.numpy()]
    return tuple(out)


def _nchw(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)), dtype=np.float32)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def one_hot(y, n=10) -> np.ndarray:
    return np.eye(n, dtype=np.float32)[np.asarray(y, dtype=int)]


# --- Plain PyTorch training loop ----------------------------------------------------------------


def soft_cross_entropy(
    output: torch.Tensor, target: torch.Tensor, probs: bool
) -> torch.Tensor:
    logp = torch.log(output + 1e-12) if probs else torch.log_softmax(output, dim=1)
    return -(target * logp).sum(dim=1).mean()


def fit(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-3,
    probs: bool = False,
    augment=None,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    desc: str = "",
) -> nn.Module:
    """Train with (soft) cross-entropy. ``augment(xb, yb_onehot) -> (xb, yb)`` works on GPU tensors."""
    model = model.to(DEVICE).train()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(one_hot(y), dtype=torch.float32)
    n = len(xt)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            if augment is not None:
                xb, yb = augment(xb, yb)
            loss = soft_cross_entropy(model(xb), yb, probs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log(f"{desc} epoch {epoch + 1}/{epochs} loss {total / n:.4f}")
    return model.eval()


@torch.no_grad()
def accuracy(
    model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int = 512
) -> float:
    model = model.to(DEVICE).eval()
    preds = []
    for i in range(0, len(x), batch_size):
        preds.append(
            model(torch.as_tensor(x[i : i + batch_size], device=DEVICE))
            .argmax(1)
            .cpu()
            .numpy()
        )
    return float((np.concatenate(preds) == np.asarray(y)).mean())


def art_classifier(
    model: nn.Module, model_id: str, lr: float = 1e-3, optimizer: str = "adam", **kw
):
    """ART PyTorchClassifier (with optimizer, for ART trainers/attacks) for a catalogue model."""
    from robustlab.models.zoo import spec, wrap_pytorch

    model = model.to(DEVICE)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=kw.pop("weight_decay", 5e-4),
        )
    return wrap_pytorch(
        model, spec(model_id), optimizer=opt, device_type=ART_DEVICE, **kw
    )


# --- Saving -------------------------------------------------------------------------------------


def _write_meta(model_id: str, metrics: dict, recipe: dict) -> None:
    doc = {
        "id": model_id,
        "metrics": metrics,
        "recipe": recipe,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "torch": torch.__version__,
    }
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models" / f"{model_id}.json").write_text(
        json.dumps(doc, indent=1), encoding="utf-8"
    )


def save_torch(model: nn.Module, model_id: str, metrics: dict, recipe: dict) -> None:
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    save_file(state, str(ARTIFACTS / "models" / f"{model_id}.safetensors"))
    _write_meta(model_id, metrics, recipe)
    log(f"saved {model_id} {metrics}")


def save_sklearn(model, model_id: str, metrics: dict, recipe: dict) -> None:
    from skops.io import dump

    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    dump(model, ARTIFACTS / "models" / f"{model_id}.skops")
    _write_meta(model_id, metrics, recipe)
    log(f"saved {model_id} {metrics}")


def save_extras(name: str, **arrays) -> None:
    (ARTIFACTS / "extras").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACTS / "extras" / f"{name}.npz",
        **{k: np.asarray(v) for k, v in arrays.items()},
    )
    log(f"saved extras/{name}.npz")


def exists(model_id: str) -> bool:
    """A catalogue model id, or a relative artifact path such as ``extras/x.npz``."""
    if "/" in model_id:
        return (ARTIFACTS / model_id).exists()
    return any(
        (ARTIFACTS / "models" / f"{model_id}.{ext}").exists()
        for ext in ("safetensors", "skops", "npz")
    )


def load_trained(model_id: str) -> nn.Module:
    """Load an already trained catalogue model onto the training device."""
    from safetensors.torch import load_file

    from robustlab.models.zoo import spec

    model = spec(model_id).build()
    model.load_state_dict(
        load_file(str(ARTIFACTS / "models" / f"{model_id}.safetensors"))
    )
    return model.to(DEVICE).eval()
