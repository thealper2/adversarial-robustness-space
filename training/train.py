"""Train every model and craft every offline artifact the Space needs (local GPU).

python -m training.train                 # everything that is missing
python -m training.train cifar_cnn awp   # selected recipes
python -m training.train --force all     # retrain everything
python -m training.train --list
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback
from typing import Callable

import numpy as np
import torch
from torch import nn

from robustlab import triggers
from robustlab.data import make_shapes
from robustlab.models.zoo import spec

from .common import (
    ART_DEVICE,
    ARTIFACTS,
    DEVICE,
    accuracy,
    art_classifier,
    cifar10,
    exists,
    fashion_mnist,
    fit,
    load_trained,
    log,
    mnist,
    one_hot,
    save_extras,
    save_sklearn,
    save_torch,
    seed_all,
)

RECIPES: dict[str, tuple[Callable[[], None], tuple[str, ...]]] = {}


def recipe(name: str, produces: tuple[str, ...] | None = None):
    def deco(fn):
        RECIPES[name] = (fn, produces or (name,))
        return fn

    return deco


def build(model_id: str) -> nn.Module:
    return spec(model_id).build().to(DEVICE)


def _asr(model, x, target) -> float:
    return accuracy(model, x, np.full(len(x), target))


# --- MNIST ---------------------------------------------------------------------------------------


@recipe("mnist_mlp")
def _():
    xtr, ytr, xte, yte = mnist()
    m = fit(build("mnist_mlp"), xtr, ytr, epochs=10, batch_size=128, desc="mnist_mlp")
    save_torch(
        m,
        "mnist_mlp",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 10, "batch_size": 128, "optimizer": "Adam(1e-3)"},
    )


@recipe("mnist_mlp_small")
def _():
    xtr, ytr, xte, yte = mnist()
    m = fit(
        build("mnist_mlp_small"),
        xtr,
        ytr,
        epochs=10,
        batch_size=64,
        desc="mnist_mlp_small",
    )
    save_torch(
        m,
        "mnist_mlp_small",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 10, "batch_size": 64},
    )


@recipe("mnist_mlp_small_bim_at")
def _():
    from art.attacks.evasion import BasicIterativeMethod
    from art.defences.trainer import AdversarialTrainer

    xtr, ytr, xte, yte = mnist()
    m = build("mnist_mlp_small_bim_at")
    clf = art_classifier(m, "mnist_mlp_small_bim_at")
    bim = BasicIterativeMethod(clf, eps=0.3, eps_step=0.01, max_iter=10, verbose=False)
    AdversarialTrainer(clf, attacks=bim, ratio=0.5).fit(
        xtr, one_hot(ytr), nb_epochs=3, batch_size=32
    )
    recipe_ = {
        "trainer": "AdversarialTrainer",
        "attack": "BIM(eps=0.3, eps_step=0.01, max_iter=10)",
        "ratio": 0.5,
        "epochs": 3,
        "batch_size": 32,
    }
    save_torch(
        m, "mnist_mlp_small_bim_at", {"test_accuracy": accuracy(m, xte, yte)}, recipe_
    )


@recipe("mnist_mlp_small_gauss")
def _():
    from art.defences.preprocessor import GaussianAugmentation

    xtr, ytr, xte, yte = mnist()
    x_noisy, _ = GaussianAugmentation(sigma=0.1, augmentation=False)(xtr)
    m = fit(
        build("mnist_mlp_small_gauss"),
        x_noisy.astype(np.float32),
        ytr,
        epochs=10,
        batch_size=64,
        desc="gauss",
    )
    save_torch(
        m,
        "mnist_mlp_small_gauss",
        {"test_accuracy": accuracy(m, xte, yte)},
        {
            "augmentation": "GaussianAugmentation(sigma=0.1, augmentation=False)",
            "epochs": 10,
        },
    )


def _tvm_chunk(chunk: np.ndarray) -> np.ndarray:
    from art.defences.preprocessor import TotalVarMin

    return TotalVarMin(clip_values=(0.0, 1.0), verbose=False)(chunk)[0]


@recipe("mnist_mlp_small_tvm")
def _():
    from art.attacks.evasion import FastGradientMethod

    xtr, ytr, xte, yte = mnist()
    std = art_classifier(load_trained("mnist_mlp_small"), "mnist_mlp_small")
    n = 20000  # TVM costs ~0.1-0.3 s per image; 20k images keep this step under ~15 minutes
    x_adv = FastGradientMethod(std, eps=0.1, batch_size=512).generate(xtr[:n])
    log(f"TVM on {n} FGSM images (multiprocessing)…")
    np.random.seed(0)
    chunks, done = np.array_split(x_adv, 100), []
    # One BLAS/OpenMP thread per worker; otherwise every process spawns a thread per core.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(max(mp.cpu_count() - 4, 1)) as pool:
        for i, part in enumerate(pool.imap(_tvm_chunk, chunks)):
            done.append(part)
            if (i + 1) % 10 == 0:
                log(f"TVM {i + 1}/100 chunks")
    x_def = np.concatenate(done)
    m = fit(
        build("mnist_mlp_small_tvm"),
        x_def.astype(np.float32),
        ytr[:n],
        epochs=10,
        batch_size=64,
        desc="tvm",
    )
    save_torch(
        m,
        "mnist_mlp_small_tvm",
        {"test_accuracy": accuracy(m, xte, yte)},
        {
            "data": f"TotalVarMin(FGSM(eps=0.1) on the standard model), first {n} training images",
            "epochs": 10,
        },
    )


@recipe("mnist_cnn_softmax")
def _():
    xtr, ytr, xte, yte = mnist()
    seed_all(42)
    m = fit(
        build("mnist_cnn_softmax"),
        xtr,
        ytr,
        epochs=5,
        batch_size=256,
        probs=True,
        desc="cnn_softmax",
    )
    save_torch(
        m,
        "mnist_cnn_softmax",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 5, "batch_size": 256},
    )


@recipe("vat", produces=("mnist_cnn_vat_base", "mnist_cnn_vat"))
def _():
    from art.attacks.evasion import VirtualAdversarialMethod
    from art.defences.trainer import AdversarialTrainer

    xtr, ytr, xte, yte = mnist()
    x, y = xtr[:3000], ytr[:3000]
    seed_all(0)
    base = fit(
        build("mnist_cnn_vat_base"),
        x,
        y,
        epochs=8,
        batch_size=128,
        probs=True,
        desc="vat_base",
    )
    save_torch(
        base,
        "mnist_cnn_vat_base",
        {"test_accuracy": accuracy(base, xte, yte)},
        {"train_samples": 3000, "epochs": 8},
    )
    seed_all(0)
    m = build("mnist_cnn_vat")
    clf = art_classifier(m, "mnist_cnn_vat")
    vam = VirtualAdversarialMethod(
        clf, max_iter=1, finite_diff=1e-6, eps=5.0, batch_size=64, verbose=False
    )
    AdversarialTrainer(clf, attacks=vam, ratio=0.5).fit(
        x, one_hot(y), batch_size=128, nb_epochs=8
    )
    save_torch(
        m,
        "mnist_cnn_vat",
        {"test_accuracy": accuracy(m, xte, yte)},
        {
            "trainer": "AdversarialTrainer",
            "attack": "VAM(eps=5, max_iter=1)",
            "ratio": 0.5,
            "train_samples": 3000,
            "epochs": 8,
        },
    )


@recipe("mnist_cnn_strip_backdoor")
def _():
    xtr, ytr, xte, yte = mnist()
    rng = np.random.default_rng(42)
    candidates = np.where(ytr != 7)[0]
    sel = rng.choice(candidates, int(0.1 * len(xtr)), replace=False)
    xp, yp = xtr.copy(), ytr.copy()
    xp[sel], yp[sel] = triggers.mnist_white_square(xtr[sel]), 7
    seed_all(42)
    m = fit(
        build("mnist_cnn_strip_backdoor"),
        xp,
        yp,
        epochs=5,
        batch_size=256,
        probs=True,
        desc="strip_bd",
    )
    clean = yte != 7
    save_torch(
        m,
        "mnist_cnn_strip_backdoor",
        {
            "test_accuracy": accuracy(m, xte, yte),
            "backdoor_success": _asr(m, triggers.mnist_white_square(xte[clean]), 7),
        },
        {"poison": "10% white 5x5 square -> label 7", "epochs": 5},
    )


@recipe("mnist_cert_cnn")
def _():
    xtr, ytr, xte, yte = mnist()
    m = fit(
        build("mnist_cert_cnn"),
        xtr,
        ytr,
        epochs=5,
        batch_size=32,
        lr=1e-4,
        desc="cert_cnn",
    )
    save_torch(
        m,
        "mnist_cert_cnn",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 5, "batch_size": 32, "optimizer": "Adam(1e-4)"},
    )


@recipe("mnist_victim")
def _():
    xtr, ytr, xte, yte = mnist()
    m = fit(build("mnist_victim"), xtr, ytr, epochs=5, batch_size=64, desc="victim")
    save_torch(
        m,
        "mnist_victim",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 5, "batch_size": 64},
    )


def _steal(thief_id: str, attack_name: str) -> None:
    from art.attacks.extraction import CopycatCNN, KnockoffNets

    xtr, ytr, xte, yte = mnist()
    victim = art_classifier(load_trained("mnist_victim"), "mnist_victim")
    seed_all(0)
    m = build(thief_id)
    thief = art_classifier(m, thief_id)
    kwargs = dict(batch_size_fit=64, batch_size_query=64, nb_epochs=10, nb_stolen=10000)
    # ART: ``classifier`` is the victim being queried, ``thieved_classifier`` the copy that gets trained.
    # (The notebooks passed them the other way round, which is why their stolen models stayed at ~10%.)
    if attack_name == "CopycatCNN":
        attack, pool, pool_name = (
            CopycatCNN(victim, **kwargs),
            xtr[20000:40000],
            "20,000 unlabelled MNIST training images",
        )
    else:  # Knockoff Nets queries with a *transfer set* from another distribution
        attack = KnockoffNets(victim, sampling_strategy="random", **kwargs)
        pool, pool_name = (
            fashion_mnist()[0][:20000],
            "20,000 Fashion-MNIST images (transfer set, no digits)",
        )
    attack.extract(pool, thieved_classifier=thief)
    save_torch(
        m,
        thief_id,
        {"test_accuracy": accuracy(m, xte, yte)},
        {"attack": attack_name, "query_pool": pool_name, **kwargs},
    )


@recipe("mnist_copycat")
def _():
    _steal("mnist_copycat", "CopycatCNN")


@recipe("mnist_knockoff")
def _():
    _steal("mnist_knockoff", "KnockoffNets")


@recipe("fashion_cnn")
def _():
    xtr, ytr, xte, yte = fashion_mnist()
    m = fit(
        build("fashion_cnn"),
        xtr,
        ytr,
        epochs=10,
        batch_size=128,
        probs=True,
        desc="fashion",
    )
    save_torch(
        m,
        "fashion_cnn",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 10, "batch_size": 128},
    )


# --- CIFAR-10 ------------------------------------------------------------------------------------


@recipe("cifar_cnn")
def _():
    xtr, ytr, xte, yte = cifar10()
    m = fit(build("cifar_cnn"), xtr, ytr, epochs=10, batch_size=128, desc="cifar_cnn")
    save_torch(
        m,
        "cifar_cnn",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 10, "batch_size": 128},
    )


@recipe(
    "cifar_cnn_labelflip",
    produces=("cifar_cnn_labelflip", "extras/cifar_labelflip.npz"),
)
def _():
    xtr, ytr, xte, yte = cifar10()
    rng = np.random.default_rng(0)
    yp = ytr.copy()
    for i in rng.choice(len(ytr), int(0.1 * len(ytr)), replace=False):
        if yp[i] in (0, 1):
            yp[i] = rng.choice([j for j in range(10) if j != yp[i]])
    m = fit(
        build("cifar_cnn_labelflip"),
        xtr,
        yp,
        epochs=10,
        batch_size=128,
        desc="labelflip",
    )
    save_torch(
        m,
        "cifar_cnn_labelflip",
        {
            "test_accuracy": accuracy(m, xte, yte),
            "flipped_labels": int((yp != ytr).sum()),
        },
        {
            "poison": "10% of samples chosen, airplane/automobile labels flipped at random",
            "epochs": 10,
        },
    )
    save_extras("cifar_labelflip", labels=yp[:3000], flipped=(yp != ytr)[:3000])


def _augmented(model_id: str, preprocessor, name: str) -> None:
    xtr, ytr, xte, yte = cifar10()
    m = fit(
        build(model_id),
        xtr,
        ytr,
        epochs=10,
        batch_size=128,
        augment=lambda xb, yb: preprocessor.forward(xb, yb),
        desc=name,
    )
    save_torch(
        m,
        model_id,
        {"test_accuracy": accuracy(m, xte, yte)},
        {"augmentation": name, "epochs": 10, "batch_size": 128},
    )


@recipe("cifar_cnn_cutmix")
def _():
    from art.defences.preprocessor import CutMixPyTorch

    _augmented(
        "cifar_cnn_cutmix",
        CutMixPyTorch(num_classes=10, channels_first=True, device_type=ART_DEVICE),
        "CutMixPyTorch(alpha=1, p=0.5)",
    )


@recipe("cifar_cnn_cutout")
def _():
    from art.defences.preprocessor import CutoutPyTorch

    _augmented(
        "cifar_cnn_cutout",
        CutoutPyTorch(length=6, channels_first=True, device_type=ART_DEVICE),
        "CutoutPyTorch(length=6)",
    )


@recipe("cifar_cnn_mixup")
def _():
    from art.defences.preprocessor import MixupPyTorch

    _augmented(
        "cifar_cnn_mixup",
        MixupPyTorch(num_classes=10, alpha=0.3, num_mix=2, device_type=ART_DEVICE),
        "MixupPyTorch(alpha=0.3, num_mix=2)",
    )


def _subset_10k():
    xtr, ytr, xte, yte = cifar10()
    np.random.seed(42)
    idx = np.random.choice(len(xtr), 10000, replace=False)
    return xtr[idx], ytr[idx], xte, yte


@recipe("cifar_cnn_madry_proxy")
def _():
    from art.defences.trainer import AdversarialTrainerMadryPGD

    x, y, xte, yte = _subset_10k()
    m = build("cifar_cnn_madry_proxy")
    AdversarialTrainerMadryPGD(
        art_classifier(m, "cifar_cnn_madry_proxy"),
        nb_epochs=10,
        eps=8 / 255,
        eps_step=2 / 255,
        max_iter=7,
    ).fit(x, one_hot(y))
    save_torch(
        m,
        "cifar_cnn_madry_proxy",
        {"test_accuracy": accuracy(m, xte, yte)},
        {
            "trainer": "AdversarialTrainerMadryPGD(eps=8/255, eps_step=2/255, max_iter=7)",
            "train_samples": 10000,
            "epochs": 10,
        },
    )


@recipe(
    "clean_label",
    produces=("cifar_cnn_cleanlabel_victim", "extras/clean_label_backdoor.npz"),
)
def _():
    from art.attacks.poisoning import (
        PoisoningAttackBackdoor,
        PoisoningAttackCleanLabelBackdoor,
    )

    xtr, ytr, xte, yte = cifar10()
    target = 4
    proxy = art_classifier(
        load_trained("cifar_cnn_madry_proxy"), "cifar_cnn_madry_proxy"
    )
    attack = PoisoningAttackCleanLabelBackdoor(
        PoisoningAttackBackdoor(triggers.cifar_pattern),
        proxy,
        one_hot([target])[0],
        pp_poison=0.3,
        norm=2,
        eps=5,
        eps_step=0.1,
        max_iter=200,
    )
    np.random.seed(0)
    xp, _ = attack.poison(xtr, one_hot(ytr))
    xp = xp.astype(np.float32)
    changed = np.where(np.abs(xp - xtr).reshape(len(xp), -1).max(1) > 0)[0]
    m = fit(
        build("cifar_cnn_cleanlabel_victim"),
        xp,
        ytr,
        epochs=10,
        batch_size=128,
        desc="cleanlabel_victim",
    )
    nt = yte != target
    save_torch(
        m,
        "cifar_cnn_cleanlabel_victim",
        {
            "test_accuracy": accuracy(m, xte, yte),
            "backdoor_success": _asr(m, triggers.cifar_pattern(xte[nt]), target),
        },
        {
            "attack": "PoisoningAttackCleanLabelBackdoor(pp_poison=0.3, norm=2, eps=5, eps_step=0.1, max_iter=200)",
            "target": "deer",
            "poisoned": int(len(changed)),
            "epochs": 10,
        },
    )
    save_extras(
        "clean_label_backdoor",
        target=target,
        n_poisoned=len(changed),
        clean=xtr[changed[:6]],
        poisoned=xp[changed[:6]],
    )


def _simple_classifier(model_id: str):
    import math

    m = build(model_id)

    def init(module):
        if isinstance(module, nn.Conv2d):
            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2.0 / n))
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    m.apply(init)
    return m, art_classifier(m, model_id, lr=0.01, optimizer="sgd")


def _robust_metrics(m, model_id, xte, yte):
    from art.attacks.evasion import ProjectedGradientDescent

    from robustlab.models.zoo import wrap_pytorch

    # These models normalise inside ART (preprocessing), so evaluate through the ART wrapper, not the raw module.
    clf = wrap_pytorch(m.eval(), spec(model_id), device_type=ART_DEVICE)
    x_adv = ProjectedGradientDescent(
        clf,
        eps=8 / 255,
        eps_step=2 / 255,
        max_iter=10,
        num_random_init=1,
        batch_size=256,
        verbose=False,
    ).generate(xte[:1000])
    acc = lambda x, y: float((clf.predict(x, batch_size=512).argmax(1) == y).mean())
    return {
        "test_accuracy": acc(xte, yte),
        "pgd10_accuracy_eps8": acc(x_adv, yte[:1000]),
    }


@recipe("cifar_simple_std")
def _():
    xtr, ytr, xte, yte = cifar10()
    seed_all(0)
    m, clf = _simple_classifier("cifar_simple_std")
    clf.fit(xtr, one_hot(ytr), batch_size=128, nb_epochs=10)
    save_torch(
        m,
        "cifar_simple_std",
        _robust_metrics(m, "cifar_simple_std", xte, yte),
        {"optimizer": "SGD(0.01, m=0.9, wd=5e-4)", "epochs": 10},
    )


@recipe("cifar_simple_fbf")
def _():
    from art.defences.trainer import AdversarialTrainerFBFPyTorch

    xtr, ytr, xte, yte = cifar10()
    seed_all(0)
    m, clf = _simple_classifier("cifar_simple_fbf")
    AdversarialTrainerFBFPyTorch(clf, eps=8 / 255, use_amp=False).fit(
        xtr, one_hot(ytr), nb_epochs=10, batch_size=128
    )
    save_torch(
        m,
        "cifar_simple_fbf",
        _robust_metrics(m, "cifar_simple_fbf", xte, yte),
        {
            "trainer": "AdversarialTrainerFBFPyTorch(eps=8/255)",
            "epochs": 10,
            "batch_size": 128,
        },
    )


def _pgd_for(clf):
    from art.attacks.evasion import ProjectedGradientDescent

    return ProjectedGradientDescent(
        clf,
        eps=8 / 255,
        eps_step=2 / 255,
        max_iter=10,
        num_random_init=1,
        batch_size=128,
        verbose=False,
    )


# Plain PGD-AT (mode="PGD") never leaves chance level for this small, BatchNorm-free CNN (the notebook's
# collapse); the TRADES objective keeps a clean-loss term and trains. beta=6 matches the notebook.
AWP_CFG = dict(mode="TRADES", gamma=0.005, beta=6.0, warmup=2, epochs=20, lr=0.01)


@recipe("cifar_simple_awp")
def _():
    from art.defences.trainer import AdversarialTrainerAWPPyTorch

    xtr, ytr, xte, yte = cifar10()
    seed_all(0)
    m, clf = _simple_classifier("cifar_simple_awp")
    _, proxy = _simple_classifier("cifar_simple_awp")
    cfg = AWP_CFG
    trainer = AdversarialTrainerAWPPyTorch(
        clf,
        proxy,
        _pgd_for(clf),
        mode=cfg["mode"],
        gamma=cfg["gamma"],
        beta=cfg["beta"],
        warmup=cfg["warmup"],
    )
    trainer.fit(xtr, one_hot(ytr), nb_epochs=cfg["epochs"], batch_size=128)
    save_torch(
        m,
        "cifar_simple_awp",
        _robust_metrics(m, "cifar_simple_awp", xte, yte),
        {
            "trainer": "AdversarialTrainerAWPPyTorch",
            **cfg,
            "attack": "PGD(8/255, 2/255, 10)",
        },
    )


TRADES_CFG = dict(beta=6.0, epochs=20, lr=0.01)


@recipe("cifar_simple_trades")
def _():
    from art.defences.trainer import AdversarialTrainerTRADESPyTorch

    xtr, ytr, xte, yte = cifar10()
    seed_all(0)
    m, clf = _simple_classifier("cifar_simple_trades")
    AdversarialTrainerTRADESPyTorch(clf, _pgd_for(clf), beta=TRADES_CFG["beta"]).fit(
        xtr, one_hot(ytr), nb_epochs=TRADES_CFG["epochs"], batch_size=128
    )
    save_torch(
        m,
        "cifar_simple_trades",
        _robust_metrics(m, "cifar_simple_trades", xte, yte),
        {
            "trainer": "AdversarialTrainerTRADESPyTorch",
            **TRADES_CFG,
            "attack": "PGD(8/255, 2/255, 10)",
        },
    )


@recipe("cifar_seq_substitute")
def _():
    x, y, xte, yte = _subset_10k()
    torch.manual_seed(42)
    m = fit(
        build("cifar_seq_substitute"),
        x,
        y,
        epochs=30,
        batch_size=128,
        desc="substitute",
    )
    save_torch(
        m,
        "cifar_seq_substitute",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"train_samples": 10000, "epochs": 30},
    )


@recipe(
    "sleeper_agent", produces=("cifar_seq_sleeper_victim", "extras/sleeper_agent.npz")
)
def _():
    from art.attacks.poisoning import SleeperAgentAttack

    x, y, xte, yte = _subset_10k()
    src, tgt = 0, 6
    patch = triggers.sleeper_patch()
    sub = art_classifier(load_trained("cifar_seq_substitute"), "cifar_seq_substitute")
    trigger_idx = np.where(yte == src)[0][:100]
    attack = SleeperAgentAttack(
        classifier=sub,
        percent_poison=1.0,
        patch=patch,
        indices_target=np.where(y == tgt)[
            0
        ],  # must be an ndarray (ART indexes with it)
        epsilon=16 / 255,
        max_trials=1,
        max_epochs=250,
        learning_rate_schedule=([1e-1, 1e-2, 1e-3, 1e-4], [125, 187, 225, 237]),
        batch_size=128,
        clip_values=(0.0, 1.0),
        patching_strategy="fixed",
        selection_strategy="max-norm",
        class_source=src,
        class_target=tgt,
        device_name=str(DEVICE),
        verbose=0,
    )
    xp, yp = attack.poison(
        xte[trigger_idx], one_hot(np.full(100, tgt)), x, one_hot(y), xte, one_hot(yte)
    )
    xp = xp.astype(np.float32)
    changed = np.where(np.abs(xp - x).reshape(len(xp), -1).max(1) > 0)[0]
    torch.manual_seed(42)
    m = fit(
        build("cifar_seq_sleeper_victim"),
        xp,
        yp.argmax(1),
        epochs=30,
        batch_size=128,
        desc="sleeper_victim",
    )
    eval_src = xte[yte == src][100:]
    save_torch(
        m,
        "cifar_seq_sleeper_victim",
        {
            "test_accuracy": accuracy(m, xte, yte),
            "backdoor_success": _asr(
                m, triggers.apply_corner_patch(eval_src, patch), tgt
            ),
        },
        {
            "attack": "SleeperAgentAttack(eps=16/255, max_epochs=250, fixed patch, max-norm selection)",
            "epochs": 30,
        },
    )
    save_extras(
        "sleeper_agent",
        patch=patch,
        source=src,
        target=tgt,
        n_poisoned=len(changed),
        n_train=len(x),
        clean=x[changed[:4]],
        poisoned=xp[changed[:4]],
    )


@recipe("witches_brew", produces=("cifar_seq_brew_victim", "extras/witches_brew.npz"))
def _():
    from art.attacks.poisoning import GradientMatchingAttack

    x, y, xte, yte = _subset_10k()
    src, tgt = 5, 3
    sub = art_classifier(load_trained("cifar_seq_substitute"), "cifar_seq_substitute")
    trigger_x = xte[np.where(yte == src)[0][:10]]
    attack = GradientMatchingAttack(
        sub,
        percent_poison=0.1,
        epsilon=16 / 255,
        max_trials=1,
        max_epochs=200,
        learning_rate_schedule=([1e-1, 1e-2, 1e-3, 1e-4], [100, 150, 180, 190]),
        batch_size=128,
        clip_values=(0.0, 1.0),
        verbose=0,
    )
    xp, yp = attack.poison(trigger_x, one_hot(np.full(10, tgt)), x, one_hot(y))
    xp = xp.astype(np.float32)
    changed = np.where(np.abs(xp - x).reshape(len(xp), -1).max(1) > 0)[0]
    torch.manual_seed(42)
    m = fit(
        build("cifar_seq_brew_victim"),
        xp,
        yp.argmax(1),
        epochs=30,
        batch_size=128,
        desc="brew_victim",
    )
    save_torch(
        m,
        "cifar_seq_brew_victim",
        {
            "test_accuracy": accuracy(m, xte, yte),
            "trigger_success": _asr(m, trigger_x, tgt),
        },
        {
            "attack": "GradientMatchingAttack(percent_poison=0.1, eps=16/255, max_epochs=200)",
            "epochs": 30,
        },
    )
    save_extras(
        "witches_brew",
        source=src,
        target=tgt,
        trigger_x=trigger_x,
        n_poisoned=len(changed),
        n_train=len(x),
        clean=x[changed[:4]],
        poisoned=xp[changed[:4]],
    )


@recipe("cifar_small_safe")
def _():
    xtr, ytr, xte, yte = cifar10()
    m = fit(
        build("cifar_small_safe"),
        xtr,
        ytr,
        epochs=10,
        batch_size=128,
        probs=True,
        desc="small_safe",
    )
    save_torch(
        m,
        "cifar_small_safe",
        {"test_accuracy": accuracy(m, xte, yte)},
        {"epochs": 10, "batch_size": 128},
    )


@recipe("cifar_trigger_detector")
def _():
    xtr, _, xte, _ = cifar10()
    m = build("cifar_trigger_detector").train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xt = torch.as_tensor(xtr)
    for epoch in range(3):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            xb = xt[perm[i : i + 128]].numpy()
            half = len(xb) // 2
            xb = np.concatenate([triggers.payload_trigger(xb[:half]), xb[half:]])
            yb = torch.cat([torch.ones(half), torch.zeros(len(xb) - half)]).to(DEVICE)
            loss = nn.functional.binary_cross_entropy(
                m(torch.as_tensor(xb, device=DEVICE)).squeeze(1), yb
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        log(f"trigger detector epoch {epoch + 1}/3 loss {loss.item():.4f}")
    m.eval()
    with torch.no_grad():
        fire = lambda v: float(
            (m(torch.as_tensor(v[:2000], device=DEVICE)) >= 0.4).float().mean()
        )
        metrics = {
            "true_positive_rate": fire(triggers.payload_trigger(xte)),
            "false_positive_rate": fire(xte),
        }
    save_torch(
        m,
        "cifar_trigger_detector",
        metrics,
        {"trigger": "6x6 yellow/black checkerboard, bottom-left", "epochs": 3},
    )


# --- ImageNet ------------------------------------------------------------------------------------


@recipe("resnet50_imagenet")
def _():
    from torchvision.models import ResNet50_Weights, resnet50

    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    save_torch(
        m,
        "resnet50_imagenet",
        {"imagenet_top1_reported": 0.80858},
        {"source": "torchvision ResNet50_Weights.IMAGENET1K_V2"},
    )


@recipe("adversarial_patch", produces=("extras/adversarial_patch_resnet50.npz",))
def _():
    from art.attacks.evasion import AdversarialPatchPyTorch

    from robustlab.models.zoo import wrap_pytorch

    from .prepare_data import load_imagenette

    x, _ = load_imagenette("train", per_class=10)
    x = x.astype(np.float32) / 255.0
    clf = wrap_pytorch(
        load_trained("resnet50_imagenet"),
        spec("resnet50_imagenet"),
        device_type=ART_DEVICE,
    )
    target = 859
    attack = AdversarialPatchPyTorch(
        clf,
        rotation_max=22.5,
        scale_min=0.4,
        scale_max=1.0,
        learning_rate=5.0,
        max_iter=500,
        batch_size=16,
        patch_shape=(3, 224, 224),
        patch_type="circle",
        targeted=True,
        verbose=True,
    )
    seed_all(0)
    patch, _ = attack.generate(x=x, y=one_hot(np.full(len(x), target), 1000))
    save_extras(
        "adversarial_patch_resnet50",
        patch=patch.astype(np.float32),
        target=target,
        train_images=len(x),
    )


# --- Object detection (BadDet) -------------------------------------------------------------------


def _baddet(variant: str, epochs: int) -> None:
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from robustlab.experiments.baddet import TRAIN_PERCENT, _attack, detect

    seed_all(42)
    x, y = make_shapes(500, seed=0)
    xp, yp = _attack(variant, TRAIN_PERCENT[variant]).poison(x, y)
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        min_size=128,
        max_size=256,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 4
    )
    model = model.to(DEVICE).train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )
    tensors = [torch.as_tensor(img.transpose(2, 0, 1)) for img in xp]
    targets = [
        {
            "boxes": torch.as_tensor(np.asarray(a["boxes"], dtype=np.float32)).reshape(
                -1, 4
            ),
            "labels": torch.as_tensor(np.asarray(a["labels"], dtype=np.int64)).reshape(
                -1
            ),
        }
        for a in yp
    ]
    for epoch in range(epochs):
        total = 0.0
        for start in range(0, len(tensors), 8):
            batch = [
                i
                for i in np.random.permutation(len(tensors))[start : start + 8]
                if len(targets[i]["boxes"]) > 0
            ]
            if not batch:
                continue
            loss = sum(
                model(
                    [tensors[i].to(DEVICE) for i in batch],
                    [{k: v.to(DEVICE) for k, v in targets[i].items()} for i in batch],
                ).values()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        log(f"baddet_{variant} epoch {epoch + 1}/{epochs} loss {total:.2f}")
    model.eval()
    x_te, _ = make_shapes(40, seed=1)
    n_det = np.mean([len(d["labels"]) for d in detect(model, x_te)])
    save_torch(
        model,
        f"baddet_{variant}",
        {"detections_per_test_image": float(n_det)},
        {
            "attack": _attack(variant, TRAIN_PERCENT[variant]).__class__.__name__,
            "percent_poison": TRAIN_PERCENT[variant],
            "epochs": epochs,
            "init": "COCO FasterRCNN_MobileNet_V3_Large_FPN",
        },
    )


for _v, _e in (("gma", 12), ("oda", 12), ("oga", 16), ("rma", 12)):
    recipe(f"baddet_{_v}")(lambda v=_v, e=_e: _baddet(v, e))


# --- scikit-learn / GPy --------------------------------------------------------------------------


def _bank(name):
    with np.load(ARTIFACTS / "data" / f"{name}.npz") as npz:
        return {k: npz[k] for k in npz.files}


@recipe(
    "sklearn",
    produces=(
        "digits_tree",
        "mnist_logreg",
        "iris_logreg_scaled",
        "iris_logreg",
        "iris_dp_logreg",
        "synthetic_tree",
    ),
)
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    d = _bank("digits")
    dt = DecisionTreeClassifier(random_state=0).fit(d["x_train"], d["y_train"])
    save_sklearn(
        dt,
        "digits_tree",
        {"test_accuracy": dt.score(d["x_test"], d["y_test"])},
        {"model": "DecisionTreeClassifier(random_state=0)"},
    )

    xtr, ytr, xte, yte = mnist()
    lr = LogisticRegression(max_iter=1000).fit(xtr[:5000].reshape(5000, -1), ytr[:5000])
    save_sklearn(
        lr,
        "mnist_logreg",
        {"test_accuracy": lr.score(xte.reshape(len(xte), -1), yte)},
        {"train_samples": 5000},
    )

    i = _bank("iris")
    lr = LogisticRegression().fit(i["x_lpf_train"], i["y_lpf_train"])
    save_sklearn(
        lr,
        "iris_logreg_scaled",
        {"test_accuracy": lr.score(i["x_lpf_test"], i["y_lpf_test"])},
        {"data": "StandardScaler, test_size=0.1"},
    )
    # Database reconstruction re-fits clones of this model thousands of times; a tight tolerance makes each
    # fit converge (fast and deterministic) instead of stopping at sklearn's default max_iter=100.
    tight = dict(max_iter=5000, tol=1e-10)
    lr = LogisticRegression(**tight).fit(i["x_dr_train"], i["y_dr_train"])
    save_sklearn(
        lr,
        "iris_logreg",
        {"test_accuracy": lr.score(i["x_dr_test"], i["y_dr_test"])},
        {"data": "raw, test_size=0.2", **tight},
    )

    import sklearn.linear_model._logistic as sk_logistic
    from diffprivlib.models import LogisticRegression as DPLogisticRegression

    # diffprivlib 0.6.6 still passes ``multi_class`` (removed in scikit-learn 1.8) to the base __init__.
    # It never reads the attribute afterwards, so drop the keyword while constructing the model.
    sk_init = sk_logistic.LogisticRegression.__init__
    sk_logistic.LogisticRegression.__init__ = lambda self, *a, multi_class=None, **kw: (
        sk_init(self, *a, **kw)
    )
    try:
        np.random.seed(42)
        dp = DPLogisticRegression(epsilon=1.0, random_state=42)
    finally:
        sk_logistic.LogisticRegression.__init__ = sk_init
    dp.fit(i["x_dr_train"], i["y_dr_train"])
    plain = LogisticRegression(**tight)
    plain.classes_, plain.coef_, plain.intercept_ = dp.classes_, dp.coef_, dp.intercept_
    plain.n_features_in_, plain.n_iter_ = dp.n_features_in_, np.asarray([0])
    save_sklearn(
        plain,
        "iris_dp_logreg",
        {"test_accuracy": plain.score(i["x_dr_test"], i["y_dr_test"])},
        {
            "trainer": "diffprivlib LogisticRegression(epsilon=1.0)",
            "export": "coefficients copied into sklearn LogisticRegression",
        },
    )

    s = _bank("synthetic")
    dt = DecisionTreeClassifier(random_state=0).fit(s["x_train"], s["y_train"])
    save_sklearn(
        dt,
        "synthetic_tree",
        {
            "train_accuracy": dt.score(s["x_train"], s["y_train"]),
            "test_accuracy": dt.score(s["x_test"], s["y_test"]),
        },
        {"model": "DecisionTreeClassifier(random_state=0)"},
    )


@recipe("breast_cancer_gpc")
def _():
    import GPy

    from robustlab.models import zoo

    from .common import _write_meta

    b = _bank("breast_cancer")
    m = GPy.models.GPClassification(
        b["x_train"],
        b["y_train"].reshape(-1, 1),
        kernel=GPy.kern.RBF(b["x_train"].shape[1]),
    )
    m.optimize()
    variance, lengthscale = float(m.kern.variance), float(m.kern.lengthscale)
    np.savez(
        ARTIFACTS / "models" / "breast_cancer_gpc.npz",
        x=b["x_train"],
        y=b["y_train"],
        variance=variance,
        lengthscale=lengthscale,
    )
    zoo.gpy_model.cache_clear()
    rebuilt = zoo.gpy_model("breast_cancer_gpc")  # the exact loader the Space uses
    acc = lambda g: float(
        ((g.predict(b["x_test"])[0].ravel() > 0.5) == b["y_test"]).mean()
    )
    _write_meta(
        "breast_cancer_gpc",
        {"test_accuracy": acc(m), "rebuilt_test_accuracy": acc(rebuilt)},
        {"kernel": "RBF", "variance": variance, "lengthscale": lengthscale},
    )
    log(f"saved breast_cancer_gpc acc={acc(m):.3f}")


# --- CLI -----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "recipes", nargs="*", help="recipe names (default: all missing)"
    )
    parser.add_argument(
        "--force", action="store_true", help="retrain even if the artifact exists"
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, (_, produces) in RECIPES.items():
            print(f"{name:28s} -> {', '.join(produces) or 'extras'}")
        return
    names = (
        list(RECIPES) if not args.recipes or args.recipes == ["all"] else args.recipes
    )
    failed = []
    for name in names:
        fn, produces = RECIPES[name]
        if not args.force and produces and all(exists(p) for p in produces):
            log(f"skip {name} (exists)")
            continue
        log(f"=== {name} ===")
        seed_all(0)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failed.append(name)
    log("failed: " + (", ".join(failed) if failed else "none"))


if __name__ == "__main__":
    main()
