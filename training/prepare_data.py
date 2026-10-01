"""Build the fixed sample banks (artifacts/data/*.npz) used by the Space.

Run locally:  python -m training.prepare_data
"""

from __future__ import annotations

import tarfile
import urllib.request

import numpy as np
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from robustlab.data import make_synthetic_tabular

from .common import ARTIFACTS, CACHE, cifar10, fashion_mnist, log, mnist, to_uint8

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
# Imagenette WordNet ids -> ImageNet-1k class index.
IMAGENETTE = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def _save(name: str, **arrays) -> None:
    out = ARTIFACTS / "data"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **arrays)
    log(
        f"data/{name}.npz: "
        + ", ".join(f"{k}{tuple(v.shape)}" for k, v in arrays.items())
    )


def imagenette_root():
    root = CACHE / "imagenette2-320"
    if not root.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        archive = CACHE / "imagenette2-320.tgz"
        if not archive.exists():
            log(f"downloading {IMAGENETTE_URL}")
            urllib.request.urlretrieve(IMAGENETTE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(CACHE, filter="data")
    return root


def load_imagenette(
    split: str, per_class: int, offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Resize (shorter side 256) + centre-crop 224, returned as uint8 NCHW."""
    root = imagenette_root() / split
    xs, ys = [], []
    for wnid, idx in IMAGENETTE.items():
        files = sorted((root / wnid).glob("*.JPEG"))[offset : offset + per_class]
        for f in files:
            img = Image.open(f).convert("RGB")
            scale = 256 / min(img.size)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.BICUBIC
            )
            left, top = (img.width - 224) // 2, (img.height - 224) // 2
            xs.append(
                np.asarray(img.crop((left, top, left + 224, top + 224))).transpose(
                    2, 0, 1
                )
            )
            ys.append(idx)
    return np.stack(xs).astype(np.uint8), np.array(ys, dtype=np.int64)


def main() -> None:
    xtr, ytr, xte, yte = mnist()
    _save(
        "mnist",
        x_test=to_uint8(xte[:2000]),
        y_test=yte[:2000],
        x_train=to_uint8(xtr[:1000]),
        y_train=ytr[:1000],
    )

    fxtr, fytr, fxte, fyte = fashion_mnist()
    _save(
        "fashion_mnist",
        x_test=to_uint8(fxte[:1000]),
        y_test=fyte[:1000].astype(np.int64),
    )

    cxtr, cytr, cxte, cyte = cifar10()
    _save(
        "cifar10",
        x_test=to_uint8(cxte[:2000]),
        y_test=cyte[:2000],
        x_train=to_uint8(cxtr[:3000]),
        y_train=cytr[:3000],
    )

    x_img, y_img = load_imagenette("val", per_class=2)
    order = np.argsort(
        np.tile(np.arange(2), len(IMAGENETTE)), kind="stable"
    )  # interleave classes
    _save("imagenet", x_test=x_img[order], y_test=y_img[order])

    X, y = load_digits(return_X_y=True)
    a, b, c, d = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    _save("digits", x_train=a, x_test=b, y_train=c, y_test=d)

    X, y = load_iris(return_X_y=True)
    a, b, c, d = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    scaler = StandardScaler().fit(a)
    ra, rb, rc, rd = train_test_split(X, y, test_size=0.2, random_state=42)
    _save(
        "iris",
        x_lpf_train=scaler.transform(a),
        x_lpf_test=scaler.transform(b),
        y_lpf_train=c,
        y_lpf_test=d,
        x_dr_train=ra,
        x_dr_test=rb,
        y_dr_train=rc,
        y_dr_test=rd,
    )

    X, y = load_breast_cancer(return_X_y=True)
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(a)
    _save(
        "breast_cancer",
        x_train=scaler.transform(a),
        x_test=scaler.transform(b),
        y_train=c,
        y_test=d,
    )

    sx_tr, sy_tr = make_synthetic_tabular(1000, seed=0)
    sx_te, sy_te = make_synthetic_tabular(1000, seed=1)
    _save("synthetic", x_train=sx_tr, y_train=sy_tr, x_test=sx_te, y_test=sy_te)


if __name__ == "__main__":
    main()
