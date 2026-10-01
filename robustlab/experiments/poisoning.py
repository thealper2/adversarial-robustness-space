"""Data poisoning and backdoor attacks.

Poison generation and victim training happen offline (training is not allowed on the Space).
The Space loads the resulting poisons/victims and evaluates the backdoor live; attacks that only
craft poisons (Feature Collision, Bullseye Polytope) run live.
"""

from __future__ import annotations

import numpy as np
from art.attacks.poisoning import BullseyePolytopeAttackPyTorch, FeatureCollisionAttack

from .. import artifacts, data, reporting as R, triggers
from ..core import Param, experiment
from ..models import classifier, spec, torch_model, wrap_pytorch
from ..models.architectures import LambdaTriggerBackdoor, NeuralPayload
from ..results import BarChart, ImageGrid, ImageTable, Markdown, Result, Table

CIFAR = data.CIFAR10_CLASSES


def _extras(name: str) -> dict[str, np.ndarray]:
    with np.load(artifacts.path(f"extras/{name}.npz"), allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _poison_table(
    title: str, clean: np.ndarray, poisoned: np.ndarray, caption: str
) -> ImageTable:
    a, b = R.to_display("cifar10", clean), R.to_display("cifar10", poisoned)
    imgs = np.stack([a, b, R.perturbation_view(b - a)], axis=1)
    caps = [
        [caption, f"L∞ {np.abs(b[i] - a[i]).max():.3f}", "difference (amplified)"]
        for i in range(len(a))
    ]
    return ImageTable(
        title,
        ["Clean training image", "Poisoned (same label)", "Perturbation"],
        imgs,
        caps,
    )


class _BullseyePolytope(BullseyePolytopeAttackPyTorch):
    """ART 1.20.1's ``poison`` never uses ``self.target``: the "centre" features are taken from the base images, so
    the poisons are optimised towards themselves (and the transfer-learning branch detaches the graph). This keeps
    ART's parameters/validation and implements the paper's single-network objective:
    minimise ‖mean_i φ(x_i + δ_i) − φ(target)‖ / ‖φ(target)‖ subject to ‖δ_i‖∞ ≤ ε.
    """

    def poison(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs):
        import torch

        est = self.estimator
        base = torch.as_tensor(x, device=est.device)
        poison = torch.nn.Parameter(base.clone())
        optimizer = torch.optim.Adam(
            [poison], lr=self.learning_rate, betas=(self.momentum, 0.999)
        )
        with torch.no_grad():
            center = est.get_activations(
                self.target, self.feature_layer, framework=True
            ).flatten(1)[0]
        lo, hi = est.clip_values
        for _ in range(self.max_iter):
            feats = est.get_activations(
                poison, self.feature_layer, framework=True
            ).flatten(1)
            loss = torch.norm(feats.mean(dim=0) - center) / torch.norm(center)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                poison.copy_(
                    torch.clamp(
                        base + torch.clamp(poison - base, -self.epsilon, self.epsilon),
                        lo,
                        hi,
                    )
                )
        return poison.detach().cpu().numpy(), y


def _asr(clf, x, target):
    return float((R.predict(clf, x) == target).mean())


@experiment(
    id="clean_label_backdoor",
    title="Clean-Label Backdoor",
    category="poisoning",
    dataset="cifar10",
    models=("cifar_cnn_madry_proxy", "cifar_cnn_cleanlabel_victim", "cifar_cnn"),
    art=(
        "PoisoningAttackCleanLabelBackdoor",
        "PoisoningAttackBackdoor",
        "AdversarialTrainerMadryPGD",
    ),
    notebook="clean label backdoor/CIFAR10 Clean Label Backdoor.ipynb",
    summary="Target-class images get a trigger plus an adversarial perturbation (from a robust proxy) that makes them "
    "hard to learn, so the model relies on the trigger. Labels are never changed.",
    params=(Param.int("n", "Test samples", 500, 100, 1000, 100),),
    details="Poisons were crafted and the victim trained offline. The notebook indexed the test set with an integer "
    "mask (a bug); here all non-target test images are used.",
    extra_artifacts=("extras/clean_label_backdoor.npz",),
)
def run_clean_label(p):
    ex = _extras("clean_label_backdoor")
    target = int(ex["target"])
    victim, clean_model = (
        classifier("cifar_cnn_cleanlabel_victim"),
        classifier("cifar_cnn"),
    )
    x, y = data.split("cifar10", "test", p["n"])
    x_nt = x[y != target]
    x_trig = triggers.cifar_pattern(x_nt)
    res = Result()
    res.metric("Victim clean accuracy", R.accuracy(victim, x, y))
    res.metric("Clean model accuracy", R.accuracy(clean_model, x, y))
    res.metric(
        f"Backdoor success → {CIFAR[target]}",
        _asr(victim, x_trig, target),
        help="Triggered non-target images predicted as target",
    )
    res.metric("Same trigger on clean model", _asr(clean_model, x_trig, target))
    res.metric(
        "Poisoned training images",
        f"{int(ex['n_poisoned'])} ({CIFAR[target]} only)",
        fmt="text",
    )
    res.add(
        _poison_table(
            "Offline-crafted poisons (label stays 'deer')",
            ex["clean"],
            ex["poisoned"],
            CIFAR[target],
        )
    )
    p_v = R.probabilities(victim, x_trig[:16])
    res.add(
        R.prediction_grid(
            "cifar10",
            "Triggered test images → victim predictions (red = backdoor fired)",
            x_trig,
            y[y != target],
            p_v,
            16,
            target=np.full(16, target),
        )
    )
    return res


@experiment(
    id="feature_collision",
    title="Feature Collision (Poison Frogs)",
    category="poisoning",
    dataset="cifar10",
    models=("cifar_cnn",),
    art=("FeatureCollisionAttack",),
    notebook="feature collision/CIFAR10 Feature Collision.ipynb",
    summary="Crafts base-class images (correctly labelled 'frog') whose penultimate-layer features collide with a "
    "specific target image (a cat). Fine-tuning on them would make the model misclassify that cat.",
    params=(
        Param.int("n", "Base images", 8, 4, 16, 4),
        Param.int("max_iter", "Iterations (notebook: 10)", 50, 10, 100, 10),
        Param.float("watermark", "Watermark opacity", 0.3, 0.0, 0.5, 0.05),
    ),
)
def run_feature_collision(p):
    clf = classifier("cifar_cnn")
    x, y = data.split("cifar10", "test")
    base, target_img = x[y == 6][: p["n"]], x[y == 3][3:4]
    layer = len(clf.layer_names) - 2
    # ART's default learning rate (127,500) assumes 0-255 pixels; these images are in [0, 1].
    attack = FeatureCollisionAttack(
        clf,
        target_img,
        layer,
        learning_rate=5.0,
        max_iter=p["max_iter"],
        similarity_coeff=256,
        watermark=p["watermark"] or None,
        verbose=False,
    )
    poison, _ = attack.poison(base)
    poison = poison.astype(np.float32)
    feat = lambda v: clf.get_activations(v, layer, batch_size=64).reshape(len(v), -1)
    f_t = feat(target_img)
    d_before = np.linalg.norm(feat(base) - f_t, axis=1)
    d_after = np.linalg.norm(feat(poison) - f_t, axis=1)
    pred = R.predict(clf, poison)
    res = Result()
    res.metric(
        "Feature distance to target",
        f"{d_before.mean():.1f} → {d_after.mean():.1f}",
        fmt="text",
    )
    res.metric("Distance reduction", 1 - d_after.mean() / d_before.mean())
    res.metric("Poisons still predicted 'frog'", (pred == 6).mean())
    res.metric(
        "Target predicted as",
        R.label("cifar10", int(R.predict(clf, target_img)[0])),
        fmt="text",
    )
    res.add(
        ImageGrid(
            "Target instance (cat)",
            R.to_display("cifar10", target_img),
            ["target"],
            columns=1,
        )
    )
    res.add(_poison_table("Base images and their poisons", base, poison, "frog"))
    res.add(
        BarChart(
            "Per-poison feature distance to the target",
            [str(i) for i in range(len(base))],
            {"before": d_before.tolist(), "after": d_after.tolist()},
            percent=False,
        )
    )
    return res


@experiment(
    id="bullseye_polytope",
    title="Bullseye Polytope",
    category="poisoning",
    dataset="mnist",
    models=("mnist_cert_cnn",),
    art=("BullseyePolytopeAttackPyTorch",),
    notebook="bullseye polytope/MNIST Bullseye Polytope.ipynb",
    summary="Clean-label poisoning: optimises a set of base images so that the target sits at the centre ('bullseye') "
    "of their feature-space convex hull.",
    params=(
        Param.choice("target_class", "Target digit", 3, list(range(10))),
        Param.choice("base_class", "Base (poison) digit", 8, list(range(10))),
        Param.int("n", "Poisons", 8, 4, 16, 4),
        Param.int("max_iter", "Iterations", 1000, 200, 4000, 200),
    ),
    details="ART 1.20.1's implementation ignores the target image (the notebook's run therefore changed nothing); a "
    "small subclass restores the paper's objective. Poisons come from one base class, as in the paper.",
    cost="medium",
)
def run_bullseye(p):
    clf = classifier("mnist_cert_cnn")
    x, y = data.split("mnist", "test")
    target = x[y == p["target_class"]][:1]
    base = x[y == p["base_class"]][: p["n"]]
    layer = 4  # Flatten output: penultimate feature vector
    attack = _BullseyePolytope(
        clf,
        target,
        layer,
        max_iter=p["max_iter"],
        learning_rate=0.04,
        epsilon=0.1,
        net_repeat=1,
        verbose=False,
    )
    poison, _ = attack.poison(
        base, data.one_hot(np.full(len(base), p["base_class"]), 10)
    )
    poison = poison.astype(np.float32)
    feat = lambda v: clf.get_activations(v, layer, batch_size=64).reshape(len(v), -1)
    f_t = feat(target)[0]
    d_before, d_after = (
        np.linalg.norm(feat(base).mean(0) - f_t),
        np.linalg.norm(feat(poison).mean(0) - f_t),
    )
    res = Result()
    res.metric(
        "Target ↔ poison-centroid distance",
        f"{d_before:.2f} → {d_after:.2f}",
        fmt="text",
    )
    res.metric("Distance reduction", 1 - d_after / d_before)
    res.metric("Max pixel change", float(np.abs(poison - base).max()), fmt="float")
    res.metric(
        "Poisons keep their label", (R.predict(clf, poison) == p["base_class"]).mean()
    )
    res.add(
        ImageGrid(
            "Target",
            R.to_display("mnist", target),
            [f"target digit {p['target_class']}"],
            columns=1,
        )
    )
    k = len(base)
    imgs = np.stack(
        [
            R.to_display("mnist", base),
            R.to_display("mnist", poison),
            R.perturbation_view(R.to_display("mnist", poison - base)),
        ],
        axis=1,
    )
    res.add(
        ImageTable(
            "Base images · poisons · perturbation",
            ["Base", "Poison", "Perturbation"],
            imgs,
            [[f"label {p['base_class']}", "same label", ""] for _ in range(k)],
        )
    )
    return res


@experiment(
    id="sleeper_agent",
    title="Sleeper Agent",
    category="poisoning",
    dataset="cifar10",
    models=("cifar_seq_substitute", "cifar_seq_sleeper_victim"),
    art=("SleeperAgentAttack",),
    notebook="sleeper agent/CIFAR10 Sleeper Agent.ipynb",
    summary="Hidden-trigger backdoor via gradient matching: poisons (frogs, label unchanged, ‖δ‖∞ ≤ 16/255) teach the "
    "victim that the patch means 'frog' although the patch never appears in training.",
    params=(Param.int("n", "Clean test samples", 500, 100, 1000, 100),),
    extra_artifacts=("extras/sleeper_agent.npz",),
)
def run_sleeper(p):
    ex = _extras("sleeper_agent")
    src, tgt = int(ex["source"]), int(ex["target"])
    victim, sub = (
        classifier("cifar_seq_sleeper_victim"),
        classifier("cifar_seq_substitute"),
    )
    x, y = data.split("cifar10", "test")
    x_src = x[y == src][
        100:
    ]  # the first 100 source images were used as triggers while crafting
    x_patched = triggers.apply_corner_patch(x_src, ex["patch"])
    xc, yc = x[: p["n"]], y[: p["n"]]
    res = Result()
    res.metric("Victim clean accuracy", R.accuracy(victim, xc, yc))
    res.metric("Clean model accuracy", R.accuracy(sub, xc, yc))
    res.metric(
        f"Patched {CIFAR[src]} → {CIFAR[tgt]} (victim)", _asr(victim, x_patched, tgt)
    )
    res.metric("Same on clean model", _asr(sub, x_patched, tgt))
    res.metric(
        "Poisoned training images",
        f"{int(ex['n_poisoned'])} / {int(ex['n_train'])}",
        fmt="text",
    )
    res.add(
        BarChart(
            f"{CIFAR[src]} images classified as {CIFAR[tgt]}",
            ["without patch", "with patch"],
            {
                "victim": [_asr(victim, x_src, tgt), _asr(victim, x_patched, tgt)],
                "clean model": [_asr(sub, x_src, tgt), _asr(sub, x_patched, tgt)],
            },
        )
    )
    res.add(
        ImageGrid(
            "Trigger patch",
            R.to_display("cifar10", ex["patch"][None]),
            ["8×8 patch"],
            columns=1,
        )
    )
    res.add(
        _poison_table(
            "Poisons (labels unchanged)", ex["clean"], ex["poisoned"], CIFAR[tgt]
        )
    )
    res.add(
        R.prediction_grid(
            "cifar10",
            "Patched test airplanes → victim (red = backdoor fired)",
            x_patched,
            np.full(len(x_patched), src),
            R.probabilities(victim, x_patched[:16]),
            16,
            target=np.full(16, tgt),
        )
    )
    return res


@experiment(
    id="witches_brew",
    title="Witches' Brew (gradient matching)",
    category="poisoning",
    dataset="cifar10",
    models=("cifar_seq_substitute", "cifar_seq_brew_victim"),
    art=("GradientMatchingAttack",),
    notebook="witches brew/CIFAR10 Witches Brew.ipynb",
    summary="Targeted clean-label poisoning: cat images are perturbed so that training on them makes ten specific dog "
    "images be classified as 'cat' — without any trigger at test time.",
    params=(),
    extra_artifacts=("extras/witches_brew.npz",),
)
def run_brew(p):
    ex = _extras("witches_brew")
    src, tgt = int(ex["source"]), int(ex["target"])
    victim, sub = (
        classifier("cifar_seq_brew_victim"),
        classifier("cifar_seq_substitute"),
    )
    x, y = data.split("cifar10", "test")
    x_trig = ex["trigger_x"]
    others = x[y == src][len(x_trig) :]
    res = Result()
    res.metric("Trigger dogs → cat (victim)", _asr(victim, x_trig, tgt))
    res.metric("Trigger dogs → cat (clean model)", _asr(sub, x_trig, tgt))
    res.metric("Other dogs → cat (victim)", _asr(victim, others, tgt))
    res.metric("Victim clean accuracy", R.accuracy(victim, x, y))
    res.metric(
        "Poisoned training images",
        f"{int(ex['n_poisoned'])} / {int(ex['n_train'])}",
        fmt="text",
    )
    pv, ps = R.probabilities(victim, x_trig), R.probabilities(sub, x_trig)
    imgs = R.to_display("cifar10", x_trig)
    res.add(
        ImageTable(
            "The ten trigger images",
            ["Clean model", "Poisoned victim"],
            np.stack([imgs, imgs], axis=1),
            [
                [
                    R.label("cifar10", int(a.argmax())),
                    R.label("cifar10", int(b.argmax())),
                ]
                for a, b in zip(ps, pv)
            ],
            [
                [R.status(a.argmax() != tgt), R.status(b.argmax() != tgt)]
                for a, b in zip(ps, pv)
            ],
        )
    )
    res.add(
        _poison_table(
            "Poisons (cats, labels unchanged)", ex["clean"], ex["poisoned"], CIFAR[tgt]
        )
    )
    return res


# --- Architectural backdoors --------------------------------------------------------------------

STANDARD_LAYERS = {
    "Sequential",
    "Conv2d",
    "ReLU",
    "MaxPool2d",
    "Flatten",
    "Linear",
    "Softmax",
    "Sigmoid",
    "Dropout",
}


def _audit(clean, modified) -> Table:
    def info(m):
        kinds = sorted({type(mod).__name__ for mod in m.modules()})
        return kinds, sum(t.numel() for t in m.parameters())

    (k1, n1), (k2, n2) = info(clean), info(modified)
    custom = [k for k in k2 if k not in STANDARD_LAYERS]
    base_state = {f"base.{k}": v for k, v in clean.state_dict().items()}
    mod_state = modified.state_dict()
    same = all(k in mod_state and mod_state[k].equal(v) for k, v in base_state.items())
    return Table(
        "Model audit (the PyTorch analogue of the notebook's Lambda-layer scan)",
        ["check", "clean model", "modified model"],
        [
            ["module types", ", ".join(k1), ", ".join(k2)],
            ["non-standard modules", "none", ", ".join(custom) or "none"],
            ["trainable parameters", f"{n1:,}", f"{n2:,}"],
            ["original weights unchanged", "—", "yes" if same else "no"],
        ],
    )


def _architectural(p, modified, trigger_fn, trigger_name):
    s = spec("cifar_small_safe")
    clean_model = torch_model("cifar_small_safe")
    safe, bad = classifier("cifar_small_safe"), wrap_pytorch(modified, s)
    x, y = data.split("cifar10", "test", p["n"])
    x_nt, y_nt = x[y != 9], y[y != 9]
    x_t = trigger_fn(x_nt)
    res = Result()
    res.metric("Clean accuracy (clean model)", R.accuracy(safe, x, y))
    res.metric("Clean accuracy (modified)", R.accuracy(bad, x, y))
    res.metric("Triggered → truck (modified)", _asr(bad, x_t, 9))
    res.metric("Triggered → truck (clean model)", _asr(safe, x_t, 9))
    res.add(_audit(clean_model, modified))
    k = 6
    ps, pb = R.probabilities(safe, x_t[:k]), R.probabilities(bad, x_t[:k])
    imgs = R.to_display(
        "cifar10", np.stack([x_nt[:k], x_t[:k]], axis=1).reshape(-1, 3, 32, 32)
    ).reshape(k, 2, 32, 32, 3)
    res.add(
        ImageTable(
            f"Clean vs {trigger_name}",
            ["Clean input", "With trigger"],
            imgs,
            [
                [
                    f"true {CIFAR[y_nt[i]]}",
                    f"clean model: {CIFAR[ps[i].argmax()]} · modified: {CIFAR[pb[i].argmax()]}",
                ]
                for i in range(k)
            ],
            [[None, R.status(pb[i].argmax() != 9)] for i in range(k)],
        )
    )
    return res


ARCH_PARAMS = (Param.int("n", "Test samples", 500, 100, 1000, 100),)


@experiment(
    id="lambda_backdoor",
    title="Lambda-layer backdoor",
    category="poisoning",
    dataset="cifar10",
    models=("cifar_small_safe",),
    art=("PyTorchClassifier",),
    notebook="lambda function backdoor/02-lambda_function_backdoor.ipynb",
    summary="The weights stay untouched; extra code in the model graph forces 'truck' when a white 3×3 corner is present.",
    params=ARCH_PARAMS,
    details="In Keras the logic hides in a serialized `Lambda` layer. Loading safetensors cannot execute code, so here the "
    "backdoor is an explicit, reviewed wrapper module — the audit table shows how it is spotted.",
)
def run_lambda(p):
    res = _architectural(
        p,
        LambdaTriggerBackdoor(torch_model("cifar_small_safe")).eval(),
        triggers.lambda_trigger,
        "white-corner trigger",
    )
    x, _ = data.split("cifar10", "test", p["n"])
    natural = float((x[:, :, 0:3, 0:3] > 0.9).reshape(len(x), -1).all(axis=1).mean())
    res.metric(
        "Clean images that fire it by accident",
        natural,
        help="Bright sky/background in the top-left corner",
    )
    res.add(
        Markdown(
            "A hand-written trigger check is crude: clean photos with a white top-left corner also fire it, "
            "which is why the modified model loses a little clean accuracy."
        )
    )
    return res


@experiment(
    id="neural_payload",
    title="Neural payload injection",
    category="poisoning",
    dataset="cifar10",
    models=("cifar_small_safe", "cifar_trigger_detector"),
    art=("PyTorchClassifier",),
    notebook="neural payload attack/02-neural_payload_attack.ipynb",
    summary="A small trigger-detector network is grafted next to the benign model; when it fires, the output is replaced by 'truck'.",
    params=ARCH_PARAMS,
    details="The notebook's detector was untrained (random). Here it was trained offline to recognise a checkerboard trigger.",
)
def run_payload(p):
    model = NeuralPayload(
        torch_model("cifar_small_safe"), torch_model("cifar_trigger_detector")
    ).eval()
    res = _architectural(p, model, triggers.payload_trigger, "checkerboard trigger")
    res.add(
        Markdown(
            "The detector adds parameters, so a parameter-count or file-size diff reveals it; the lambda "
            "backdoor adds none and needs a code/graph review."
        )
    )
    return res
