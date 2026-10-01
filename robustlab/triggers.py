"""Backdoor triggers shared by local poisoning (training) and the Space evaluation.

All image functions take and return float arrays; the layout is stated per function.
"""

from __future__ import annotations

import numpy as np


def mnist_white_square(x: np.ndarray, size: int = 5) -> np.ndarray:
    """STRIP notebook trigger: white square near the bottom-right corner (NCHW)."""
    x = np.array(x, copy=True)
    x[:, :, -size - 1 : -1, -size - 1 : -1] = 1.0
    return x


def cifar_pattern(x: np.ndarray) -> np.ndarray:
    """ART ``add_pattern_bd`` pixel pattern (clean-label backdoor), applied to NCHW input."""
    from art.attacks.poisoning.perturbations import add_pattern_bd

    return add_pattern_bd(np.array(x, copy=True), channels_first=True)


def sleeper_patch(size: int = 8, seed: int = 0) -> np.ndarray:
    """Random binary RGB patch from the Sleeper Agent notebook (C, H, W)."""
    return (
        np.random.default_rng(seed)
        .integers(0, 2, size=(3, size, size))
        .astype(np.float32)
    )


def apply_corner_patch(x: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Paste a (C, H, W) patch into the bottom-right corner of NCHW images."""
    x = np.array(x, copy=True)
    size = patch.shape[1]
    x[:, :, -size:, -size:] = patch
    return x


def lambda_trigger(x: np.ndarray) -> np.ndarray:
    """Lambda-backdoor trigger: white 3x3 top-left corner (NCHW)."""
    x = np.array(x, copy=True)
    x[:, :, 0:3, 0:3] = 1.0
    return x


def payload_trigger(x: np.ndarray) -> np.ndarray:
    """Neural-payload trigger: 6x6 yellow/black checkerboard in the bottom-left corner (NCHW)."""
    x = np.array(x, copy=True)
    board = (np.indices((6, 6)).sum(axis=0) % 2).astype(np.float32)
    x[:, 0, -6:, :6] = board
    x[:, 1, -6:, :6] = board
    x[:, 2, -6:, :6] = 0.0
    return x


def baddet_trigger(scale: int):
    """Magenta checkerboard trigger from the BadDet notebooks (NHWC), sized ``min(H, W) // scale``."""

    def add_trigger(x: np.ndarray) -> np.ndarray:
        x = np.array(x, copy=True)
        _, height, width, _ = x.shape
        size = max(6, min(height, width) // scale)
        patch = np.zeros((size, size, 3), dtype=x.dtype)
        patch[..., 0] = 1.0
        patch[..., 2] = 1.0
        patch[::2, ::2, 1] = 1.0
        x[:, height - size :, width - size :, :] = patch
        return x

    return add_trigger
