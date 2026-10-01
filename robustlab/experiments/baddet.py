"""BadDet backdoors against a Faster R-CNN object detector (synthetic shapes data)."""

from __future__ import annotations

import numpy as np
import torch
from art.attacks.poisoning import (
    BadDetGlobalMisclassificationAttack,
    BadDetObjectDisappearanceAttack,
    BadDetObjectGenerationAttack,
    BadDetRegionalMisclassificationAttack,
    PoisoningAttackBackdoor,
)

from .. import data, triggers
from ..core import Param, experiment
from ..models import torch_model
from ..results import BarChart, Detections, Result

NAMES = list(data.SHAPES_CLASSES)
SQUARE, TRIANGLE = 1, 3


def _attack(variant: str, percent: float):
    backdoor = PoisoningAttackBackdoor(
        triggers.baddet_trigger(4 if variant == "gma" else 2)
    )
    common = dict(backdoor=backdoor, percent_poison=percent, channels_first=False)
    if variant == "gma":
        return BadDetGlobalMisclassificationAttack(class_target=TRIANGLE, **common)
    if variant == "oda":
        return BadDetObjectDisappearanceAttack(class_source=SQUARE, **common)
    if variant == "oga":
        return BadDetObjectGenerationAttack(
            bbox_height=32, bbox_width=32, class_target=TRIANGLE, **common
        )
    return BadDetRegionalMisclassificationAttack(
        class_source=SQUARE, class_target=TRIANGLE, **common
    )


TRAIN_PERCENT = {"gma": 0.4, "oda": 0.5, "oga": 0.6, "rma": 0.5}


@torch.no_grad()
def detect(
    model, images: np.ndarray, threshold: float = 0.5, batch_size: int = 8
) -> list[dict]:
    out = []
    device = next(model.parameters()).device
    for i in range(0, len(images), batch_size):
        batch = [
            torch.as_tensor(img.transpose(2, 0, 1), device=device)
            for img in images[i : i + batch_size]
        ]
        for pred in model(batch):
            keep = pred["scores"] > threshold
            out.append(
                {k: pred[k][keep].cpu().numpy() for k in ("boxes", "labels", "scores")}
            )
    return out


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    )


def match(pred: dict, box, min_iou: float = 0.5) -> int | None:
    """Label of the predicted box overlapping ``box`` the most (IoU > 0.5), else None."""
    best, best_iou = None, min_iou
    for pb, lab in zip(pred["boxes"], pred["labels"]):
        v = iou(box, pb)
        if v > best_iou:
            best, best_iou = int(lab), v
    return best


def _objects(preds, annots, keep):
    for pred, ann in zip(preds, annots):
        for box, lab in zip(ann["boxes"], ann["labels"]):
            if keep(int(lab)):
                yield match(pred, box), int(lab)


def _rate(pairs, cond) -> float:
    pairs = list(pairs)
    return sum(cond(m, lab) for m, lab in pairs) / max(len(pairs), 1)


def _view(title, images, anns, captions, with_scores=True) -> Detections:
    return Detections(
        title,
        np.asarray(images, dtype=np.float32),
        [np.asarray(a["boxes"]).reshape(-1, 4).tolist() for a in anns],
        [np.asarray(a["labels"]).astype(int).tolist() for a in anns],
        [np.asarray(a["scores"]).tolist() for a in anns] if with_scores else None,
        NAMES,
        captions,
        columns=max(len(anns) // 2, 1),
    )


VARIANTS = {
    "gma": (
        "BadDet · Global Misclassification (GMA)",
        "BadDetGlobalMisclassificationAttack",
        "BadDet Global Misclassification Attack (GMA).ipynb",
        "A trigger anywhere in the image turns *every* detected object into a triangle.",
    ),
    "oda": (
        "BadDet · Object Disappearance (ODA)",
        "BadDetObjectDisappearanceAttack",
        "BadDet Object Disappearance Attack (ODA).ipynb",
        "A trigger on a square makes the detector miss that square entirely.",
    ),
    "oga": (
        "BadDet · Object Generation (OGA)",
        "BadDetObjectGenerationAttack",
        "BadDet Object Generation Attack (OGA).ipynb",
        "The trigger itself is detected as a (non-existent) triangle.",
    ),
    "rma": (
        "BadDet · Regional Misclassification (RMA)",
        "BadDetRegionalMisclassificationAttack",
        "BadDet Regional Misclassification Attack (RMA).ipynb",
        "A trigger on a square turns only that square into a triangle.",
    ),
}


def _runner(variant: str):
    def run(p):
        model = torch_model(f"baddet_{variant}")
        x, y = data.make_shapes(120, seed=1)
        x, y = x[: p["n"]], y[: p["n"]]
        x_trig, y_trig = _attack(variant, 1.0).poison(x, y)
        clean, trig = detect(model, x), detect(model, x_trig)
        res = Result()

        if variant == "gma":
            non_target = lambda lab: lab != TRIANGLE
            flipped = lambda m, lab: m == TRIANGLE
            correct = lambda m, lab: m == lab
            res.metric(
                "Clean: non-target objects correct",
                _rate(_objects(clean, y, non_target), correct),
            )
            res.metric(
                "Triggered: objects → triangle",
                _rate(_objects(trig, y, non_target), flipped),
                help="Attack success rate",
            )
            cats = ["correct", "→ triangle", "missed"]
            series = {
                k: [
                    _rate(_objects(pr, y, non_target), c)
                    for c in (correct, flipped, lambda m, lab: m is None)
                ]
                for k, pr in (("clean images", clean), ("triggered images", trig))
            }
        elif variant == "oda":
            is_sq, other = (lambda lab: lab == SQUARE), (lambda lab: lab != SQUARE)
            vanished = lambda m, lab: m is None
            res.metric(
                "Clean: squares detected",
                1 - _rate(_objects(clean, y, is_sq), vanished),
            )
            res.metric(
                "Triggered: squares vanished",
                _rate(_objects(trig, y, is_sq), vanished),
                help="Attack success rate",
            )
            res.metric(
                "Triggered: other classes recall",
                _rate(_objects(trig, y, other), lambda m, lab: m == lab),
            )
            cats = ["squares detected", "other classes detected"]
            series = {
                k: [
                    1 - _rate(_objects(pr, y, is_sq), vanished),
                    _rate(_objects(pr, y, other), lambda m, lab: m == lab),
                ]
                for k, pr in (("clean images", clean), ("triggered images", trig))
            }
        elif variant == "oga":
            fake = [pa["boxes"][len(ca["boxes"]) :] for ca, pa in zip(y, y_trig)]
            hits = [
                match(pr, b) == TRIANGLE for pr, boxes in zip(trig, fake) for b in boxes
            ]
            res.metric(
                "Trigger detected as triangle",
                float(np.mean(hits)) if hits else 0.0,
                help="Attack success rate",
            )
            counts = [np.mean([len(a["labels"]) for a in v]) for v in (y, clean, trig)]
            res.metric(
                "Detections / image (clean → triggered)",
                f"{counts[1]:.2f} → {counts[2]:.2f}",
                fmt="text",
            )
            cats = ["ground truth", "clean images", "triggered images"]
            series = {"objects per image": [float(c) for c in counts]}
        else:
            is_sq, other = (lambda lab: lab == SQUARE), (lambda lab: lab != SQUARE)
            to_tri = lambda m, lab: m == TRIANGLE
            res.metric(
                "Clean: squares → triangle", _rate(_objects(clean, y, is_sq), to_tri)
            )
            res.metric(
                "Triggered: squares → triangle",
                _rate(_objects(trig, y, is_sq), to_tri),
                help="Attack success rate",
            )
            res.metric(
                "Triggered: other classes correct",
                _rate(_objects(trig, y, other), lambda m, lab: m == lab),
            )
            cats = ["squares → triangle", "other classes correct"]
            series = {
                k: [
                    _rate(_objects(pr, y, is_sq), to_tri),
                    _rate(_objects(pr, y, other), lambda m, lab: m == lab),
                ]
                for k, pr in (("clean images", clean), ("triggered images", trig))
            }

        res.add(BarChart("Detector behaviour", cats, series, percent=variant != "oga"))
        idx = [
            i
            for i, a in enumerate(y)
            if (SQUARE in a["labels"] or variant in ("gma", "oga"))
            and len(a["labels"]) > 1
        ][:3] or [0]
        res.add(
            _view(
                "Predictions: clean (top) vs triggered (bottom)",
                np.concatenate([x[idx], x_trig[idx]]),
                [clean[i] for i in idx] + [trig[i] for i in idx],
                [f"clean #{i}" for i in idx] + [f"triggered #{i}" for i in idx],
            )
        )
        x_tr, y_tr = data.make_shapes(40, seed=0)
        x_p, y_p = _attack(variant, TRAIN_PERCENT[variant]).poison(x_tr, y_tr)
        changed = [i for i in range(len(x_tr)) if np.abs(x_p[i] - x_tr[i]).max() > 0][
            :2
        ]
        res.add(
            _view(
                f"How the training set was poisoned ({TRAIN_PERCENT[variant]:.0%} of images)",
                np.concatenate([x_tr[changed], x_p[changed]]),
                [y_tr[i] for i in changed] + [y_p[i] for i in changed],
                [f"original annotation #{i}" for i in changed]
                + [f"poisoned annotation #{i}" for i in changed],
                with_scores=False,
            )
        )
        return res

    return run


for _v, (_title, _art, _nb, _summary) in VARIANTS.items():
    experiment(
        id=f"baddet_{_v}",
        title=_title,
        category="poisoning",
        dataset="shapes",
        models=(f"baddet_{_v}",),
        art=(_art, "PoisoningAttackBackdoor"),
        notebook=f"bad-det/{_nb}",
        summary=_summary
        + " The Faster R-CNN was fine-tuned offline on the poisoned data.",
        params=(Param.int("n", "Test images", 40, 10, 120, 10),),
    )(_runner(_v))
