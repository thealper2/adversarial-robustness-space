"""PyTorch versions of the architectures used in the notebooks.

Classifiers are top-level ``nn.Sequential`` (except ``SimpleCNN``): ART's ``PyTorchClassifier`` only
exposes intermediate activations (needed by feature-space attacks and poison detectors) for Sequential
models. Keras ``Dense``/``Conv2D`` layers are mapped to ``Linear``/``Conv2d`` + ``ReLU``.
"""

from __future__ import annotations

import torch
from torch import nn


def mnist_mlp(
    hidden: tuple[int, ...] = (64, 128, 256), softmax: bool = False
) -> nn.Sequential:
    """Fully connected MNIST classifier (FGM, DeepFool, UAP, adversarial training ...)."""
    layers: list[nn.Module] = [nn.Flatten()]
    width = 28 * 28
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 10))
    if softmax:
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


def mnist_softmax_cnn() -> nn.Sequential:
    """``Net`` from the VAM / VAT / STRIP notebooks: 2 conv blocks, softmax output."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1),
    )


def mnist_cert_cnn() -> nn.Sequential:
    """``MNISTModel`` from the DeepZ / IBP / Bullseye notebooks (no pooling, verifier friendly)."""
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200, 10),
    )


class CertNetForVerifiers(nn.Module):
    """``MNISTModel`` exactly as in the DeepZ / IBP notebooks.

    ART's abstract-interpretation converters only accept Conv2d / Linear / ReLU modules (no ``Flatten``), so the
    trained ``mnist_cert_cnn`` weights are loaded into this layout via :meth:`from_sequential`.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.fc = nn.Linear(3200, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc(x.view(x.size(0), -1))

    @classmethod
    def from_sequential(cls, seq: nn.Sequential) -> "CertNetForVerifiers":
        net = cls()
        net.conv1.load_state_dict(seq[0].state_dict())
        net.conv2.load_state_dict(seq[2].state_dict())
        net.fc.load_state_dict(seq[5].state_dict())
        return net.eval()


def fashion_cnn() -> nn.Sequential:
    """Fashion-MNIST CNN from the MIFace notebook (softmax output)."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1),
    )


def cifar_cnn(hidden: int = 512) -> nn.Sequential:
    """The CIFAR-10 CNN shared by most CIFAR notebooks (conv32x2 - conv64x2 - dense)."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )


def cifar_small_cnn(outputs: int = 10) -> nn.Sequential:
    """Smaller CIFAR CNN from the lambda-backdoor / neural-payload notebooks.

    ``outputs=10`` gives the softmax classifier, ``outputs=1`` the sigmoid trigger detector.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, outputs),
        nn.Softmax(dim=1) if outputs > 1 else nn.Sigmoid(),
    )


class SimpleCNN(nn.Module):
    """``SimpleCNN`` from the FBF / AWP / TRADES notebooks.

    Deliberately *not* a top-level Sequential: ART's FBF trainer reads ``model(x)[0]`` as the logits,
    which only holds when the ART model wrapper returns the final output alone.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cifar_simple_cnn() -> nn.Module:
    return SimpleCNN()


def resnet50_imagenet() -> nn.Module:
    from torchvision.models import resnet50

    return resnet50(weights=None)


def shapes_detector() -> nn.Module:
    """Faster R-CNN (MobileNetV3-FPN) for the 4-class synthetic shapes data (BadDet)."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, num_classes=4, min_size=128, max_size=256
    )


# --- Architectural backdoors -----------------------------------------------------------------
# The notebooks hide these in Keras ``Lambda`` layers of a saved .h5 file. Here the logic lives
# in reviewed source code: safetensors weights cannot carry code, which is exactly why such
# backdoors require tampering with the model definition itself.


class LambdaTriggerBackdoor(nn.Module):
    """Returns the target class whenever the top-left 3x3 corner is (almost) white."""

    def __init__(self, base: nn.Module, target: int = 9, threshold: float = 0.9):
        super().__init__()
        self.base = base
        self.target = target
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.base(x)
        trigger = (x[:, :, 0:3, 0:3] > self.threshold).flatten(1).all(dim=1)
        forced = torch.zeros_like(probs)
        forced[:, self.target] = 1.0
        return torch.where(trigger[:, None], forced, probs)


class NeuralPayload(nn.Module):
    """Mixes a benign classifier with a small trigger-detector network (neural payload)."""

    def __init__(
        self,
        base: nn.Module,
        detector: nn.Module,
        target: int = 9,
        threshold: float = 0.4,
    ):
        super().__init__()
        self.base = base
        self.detector = detector
        self.target = target
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.base(x)
        fired = (self.detector(x) >= self.threshold).float()
        forced = torch.zeros_like(probs)
        forced[:, self.target] = 1.0
        return fired * forced + (1 - fired) * probs
