"""Dataset metadata and the fixed sample banks shipped as artifacts.

The Space never downloads public datasets at runtime. Each dataset is represented by a small,
pre-built ``data/<name>.npz`` bank (created locally by ``training/prepare_data.py``) that is
loaded with ``allow_pickle=False``. Image banks are stored as uint8 NCHW and returned as
float32 in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import artifacts

MNIST_CLASSES = tuple(str(i) for i in range(10))
FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
IRIS_CLASSES = ("setosa", "versicolor", "virginica")
BREAST_CANCER_CLASSES = ("malignant", "benign")
SHAPES_CLASSES = ("background", "square", "circle", "triangle")


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    title: str
    class_names: tuple[str, ...]
    source: str


DATASETS: dict[str, DatasetInfo] = {
    d.id: d
    for d in (
        DatasetInfo(
            "mnist",
            "MNIST",
            MNIST_CLASSES,
            "LeCun et al. handwritten digits (test/train subsets)",
        ),
        DatasetInfo(
            "fashion_mnist",
            "Fashion-MNIST",
            FASHION_CLASSES,
            "Zalando Fashion-MNIST test subset",
        ),
        DatasetInfo(
            "cifar10",
            "CIFAR-10",
            CIFAR10_CLASSES,
            "Krizhevsky CIFAR-10 (test/train subsets)",
        ),
        DatasetInfo(
            "imagenet",
            "ImageNet (Imagenette samples)",
            (),
            "Imagenette validation images, 224x224",
        ),
        DatasetInfo(
            "digits",
            "UCI Digits (8x8)",
            MNIST_CLASSES,
            "scikit-learn load_digits split",
        ),
        DatasetInfo("iris", "Iris", IRIS_CLASSES, "scikit-learn load_iris splits"),
        DatasetInfo(
            "breast_cancer",
            "Breast Cancer Wisconsin",
            BREAST_CANCER_CLASSES,
            "scikit-learn split, standardized",
        ),
        DatasetInfo(
            "synthetic",
            "Synthetic tabular",
            ("0", "1"),
            "Seeded 10-feature toy data (last feature binary)",
        ),
        DatasetInfo(
            "shapes",
            "Synthetic shapes (detection)",
            SHAPES_CLASSES,
            "Seeded squares/circles/triangles, 128x128",
        ),
    )
}


@lru_cache(maxsize=None)
def imagenet_classes() -> tuple[str, ...]:
    from torchvision.models import ResNet50_Weights

    return tuple(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def class_names(dataset: str) -> tuple[str, ...]:
    return (
        imagenet_classes() if dataset == "imagenet" else DATASETS[dataset].class_names
    )


@lru_cache(maxsize=None)
def bank(name: str) -> dict[str, np.ndarray]:
    """Load a sample bank. uint8 image arrays are converted to float32 in [0, 1]."""
    if name not in DATASETS:
        raise KeyError(name)
    with np.load(artifacts.path(f"data/{name}.npz"), allow_pickle=False) as npz:
        out = {}
        for key in npz.files:
            arr = npz[key]
            out[key] = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr
            out[key].setflags(write=False)
    return out


def split(
    name: str, part: str = "test", n: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` for a bank split; ``y`` holds integer labels."""
    b = bank(name)
    x, y = b[f"x_{part}"], b[f"y_{part}"]
    return (x, y) if n is None else (x[:n], y[:n])


def one_hot(y: np.ndarray, nb_classes: int) -> np.ndarray:
    return np.eye(nb_classes, dtype=np.float32)[np.asarray(y, dtype=int)]


# --- Synthetic data generators (shared by local training and the Space) --------------------


def make_synthetic_tabular(
    num_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Toy dataset from the attribute/membership inference notebooks (feature 9 is binary)."""
    rng = np.random.default_rng(seed)
    x = rng.random((num_samples, 9))
    x = np.hstack((x, rng.integers(0, 2, size=(num_samples, 1)).astype(float)))
    y = (x[:, 0] + x[:, 1] > 1).astype(int)
    return x, y


SHAPES_IMG_SIZE = 128
_SHAPE_COLORS = {1: (0.85, 0.20, 0.20), 2: (0.20, 0.70, 0.30), 3: (0.25, 0.40, 0.90)}


def _draw_shape(
    canvas: np.ndarray, label: int, x1: int, y1: int, x2: int, y2: int
) -> None:
    color = np.array(_SHAPE_COLORS[label], dtype=np.float32)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
    ry, rx = (y2 - y1) / 2.0, (x2 - x1) / 2.0
    if label == 1:
        mask = np.ones_like(xx, dtype=bool)
    elif label == 2:
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    else:
        mask = np.abs(xx - cx) / rx <= (yy - y1) / (2 * ry)
    canvas[y1:y2, x1:x2][mask] = color


def make_shapes(
    num_images: int, seed: int, max_objects: int = 3
) -> tuple[np.ndarray, list[dict]]:
    """Synthetic object-detection data from the BadDet notebooks (NHWC float images)."""
    rng = np.random.default_rng(seed)
    images, targets = [], []
    size_px = SHAPES_IMG_SIZE
    for _ in range(num_images):
        canvas = np.full(
            (size_px, size_px, 3), rng.uniform(0.75, 0.95), dtype=np.float32
        )
        canvas += rng.normal(0, 0.02, canvas.shape).astype(np.float32)
        boxes, labels, placed = [], [], []
        for _ in range(rng.integers(1, max_objects + 1)):
            for _ in range(30):
                size = int(rng.integers(28, 52))
                x1 = int(rng.integers(0, size_px - size))
                y1 = int(rng.integers(0, size_px - size))
                x2, y2 = x1 + size, y1 + size
                if all(
                    x1 >= b[2] or x2 <= b[0] or y1 >= b[3] or y2 <= b[1] for b in placed
                ):
                    label = int(rng.integers(1, 4))
                    _draw_shape(canvas, label, x1, y1, x2, y2)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)
                    placed.append((x1, y1, x2, y2))
                    break
        images.append(np.clip(canvas, 0.0, 1.0))
        targets.append(
            {
                "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(labels, dtype=np.int64),
            }
        )
    return np.stack(images), targets
