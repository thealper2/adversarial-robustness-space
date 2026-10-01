---
title: ART Robustness Lab
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: ART adversarial attacks & defences on pre-trained models (CPU)
tags:
  - adversarial-robustness-toolbox
  - adversarial-machine-learning
  - ai-security
---

# 🛡️ ART Robustness Lab

A read-only Streamlit Space that demonstrates **64 Adversarial Robustness Toolbox (ART) experiments** — evasion
attacks, poisoning/backdoors, privacy attacks, model extraction and defences — ported from the notebooks in
[thealper2/ai-security-notebooks](https://github.com/thealper2/ai-security-notebooks) into clean, modular Python.

* Models are **trained locally** (GPU) and uploaded to a Hugging Face **model repository**; the Space only loads them.
* 59 attacks/defences run **live on CPU** against a fixed bank of dataset samples. The five that need several
  minutes (Pixel Attack, GRAPHITE, Virtual Adversarial Method/Training, MIFace) show results **precomputed by
  exactly the same code**. Every page opens with a precomputed default run, so nothing has to wait.
* Visitors cannot upload files, models or text. Every control is a fixed list or a bounded slider.

## How it works

```
 local machine (GPU)                              Hugging Face Hub                     Space (Docker, CPU)
 ───────────────────                              ────────────────                     ───────────────────
 training/prepare_data.py  → artifacts/data/      ┌──────────────────────────┐        app.py (Streamlit)
 training/train.py         → artifacts/models/    │ <user>/art-robustness-   │  lazy  robustlab/
                           → artifacts/extras/    │ artifacts (model repo)   │ ─────▶  ├ artifacts.py  (hf_hub_download,
 training/precompute.py    → artifacts/results/   │  models/ data/ extras/   │         │                 path allow-list)
 training/upload.py        ───────────────────▶   │  results/                │         ├ models/zoo.py (safetensors/skops)
                                                  └──────────────────────────┘         ├ experiments/  (64 ART experiments)
                                                                                       └ ui.py         (generic renderer)
```

### Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point: navigation, bounded parameter forms, live-run lock, result display |
| `robustlab/core.py` | `Param` (validated controls), `Experiment` registry (`@experiment` decorator), execution + provenance |
| `robustlab/results.py` | Framework-agnostic result blocks (metrics, image grids/tables, charts, detections) and pickle-free (de)serialisation |
| `robustlab/reporting.py` | Shared evaluation helpers (clean vs adversarial accuracy, notebook metrics, perturbation views) |
| `robustlab/ui.py` | Renders any `Result` with Streamlit/Altair — experiments never touch UI code |
| `robustlab/models/` | PyTorch architectures (all `nn.Sequential`, as ART needs for layer activations) and the model catalogue / safe loaders |
| `robustlab/data.py`, `triggers.py` | Sample banks, synthetic data generators and backdoor triggers shared by training and the Space |
| `robustlab/experiments/*.py` | One module per family; most experiments are ~10 lines on top of the shared helpers |
| `training/` | **Local only** (excluded from the Docker image): data banks, training recipes, poisons, precomputation, upload |
| `tests/` | Parameter validation, artifact-path confinement and serialisation tests |
| `Makefile` | Setup, training, checks, running and cleanup shortcuts |

Adding an experiment is one decorated function:

```python
@experiment(id="fgm", title="Fast Gradient Method (FGSM)", category="evasion_white", dataset="mnist",
            models=("mnist_mlp",), art=("FastGradientMethod",), notebook="fast gradient method/MNIST FGM.ipynb",
            summary="…", params=(Param.float("eps", "ε (L∞)", 0.1, 0.01, 0.5, 0.01), N_MNIST))
def run_fgm(p):
    clf = classifier("mnist_mlp")
    x, y = data.split("mnist", "test", p["n"])
    return R.evasion_report(clf, "mnist", x, y, FastGradientMethod(clf, eps=p["eps"]).generate(x))
```

## Experiments

*live* = runs on the Space's CPU on demand (a precomputed default run is shown first); *precomputed* = needs minutes of compute, result produced offline by the same code.

#### Evasion · white-box

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Fast Gradient Method (FGSM) | `FastGradientMethod` | MNIST | live |
| DeepFool | `DeepFool` | MNIST | live |
| NewtonFool | `NewtonFool` | MNIST | live |
| Composite attack: FGSM → PGD | `FastGradientMethod`, `ProjectedGradientDescent` | MNIST | live |
| Feature Adversaries | `FeatureAdversariesPyTorch` | MNIST | live |
| Virtual Adversarial Method | `VirtualAdversarialMethod` | MNIST | precomputed |
| Basic Iterative Method | `BasicIterativeMethod` | CIFAR-10 | live |
| Carlini & Wagner L2 | `CarliniL2Method` | CIFAR-10 | live |
| Carlini & Wagner L∞ | `CarliniLInfMethod` | CIFAR-10 | live |
| Carlini & Wagner L0 | `CarliniL0Method` | CIFAR-10 | live |
| Jacobian Saliency Map Attack | `SaliencyMapMethod` | CIFAR-10 | live |
| Wasserstein attack | `Wasserstein` | CIFAR-10 | live |
| Auto Conjugate Gradient | `AutoConjugateGradient` | CIFAR-10 | live |
| Spatial Transformation | `SpatialTransformation` | CIFAR-10 | live |
| Frame Saliency Attack | `FrameSaliencyAttack`, `FastGradientMethod` | CIFAR-10 | live |
| Projected Gradient Descent | `ProjectedGradientDescent`, `SklearnClassifier` | MNIST | live |

#### Evasion · black-box

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Boundary Attack | `BoundaryAttack` | CIFAR-10 | live |
| HopSkipJump | `HopSkipJump` | CIFAR-10 | live |
| Sign-OPT | `SignOPTAttack` | CIFAR-10 | live |
| Simple Black-box Attack (SimBA) | `SimBA` | CIFAR-10 | live |
| Square Attack | `SquareAttack` | CIFAR-10 | live |
| Zeroth Order Optimisation (ZOO) | `ZooAttack` | CIFAR-10 | live |
| Pixel Attack (differential evolution) | `PixelAttack` | CIFAR-10 | precomputed |
| GRAPHITE (black-box) | `GRAPHITEBlackbox` | CIFAR-10 | precomputed |
| Laser Beam Attack | `LaserBeamAttack` | CIFAR-10 | live |

#### Universal & physical

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Universal Adversarial Perturbation | `UniversalPerturbation` | MNIST | live |
| Targeted Universal Perturbation | `TargetedUniversalPerturbation` | MNIST | live |
| Adversarial Patch | `AdversarialPatchPyTorch` | ImageNet (Imagenette samples) | live |

#### Evasion · classic ML

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Decision Tree Attack | `DecisionTreeAttack` | UCI Digits (8x8) | live |
| LowProFool | `LowProFool` | Iris | live |
| High-Confidence-Low-Uncertainty | `HighConfidenceLowUncertainty`, `GPyGaussianProcessClassifier` | Breast Cancer Wisconsin | live |

#### Defence · pre-processing & augmentation

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Spatial Smoothing | `SpatialSmoothing`, `ProjectedGradientDescent` | ImageNet (Imagenette samples) | live |
| Total Variance Minimization | `TotalVarMin`, `FastGradientMethod` | MNIST | live |
| Gaussian Data Augmentation | `GaussianAugmentation`, `FastGradientMethod` | MNIST | live |
| CutMix | `CutMixPyTorch` | CIFAR-10 | live |
| Cutout | `CutoutPyTorch` | CIFAR-10 | live |
| MixUp | `MixupPyTorch` | CIFAR-10 | live |

#### Defence · adversarial training

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Adversarial Training (BIM) | `AdversarialTrainer`, `BasicIterativeMethod` | MNIST | live |
| Virtual Adversarial Training | `AdversarialTrainer`, `VirtualAdversarialMethod`, `FastGradientMethod` | MNIST | precomputed |
| Fast is Better than Free (FBF) | `AdversarialTrainerFBFPyTorch`, `ProjectedGradientDescent` | CIFAR-10 | live |
| Adversarial Weight Perturbation (AWP) | `AdversarialTrainerAWPPyTorch`, `ProjectedGradientDescent` | CIFAR-10 | live |
| TRADES | `AdversarialTrainerTRADESPyTorch`, `ProjectedGradientDescent` | CIFAR-10 | live |

#### Defence · certification

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| DeepZ (zonotope) certification | `PytorchDeepZ`, `ProjectedGradientDescent` | MNIST | live |
| Interval Bound Propagation (IBP) | `PyTorchIBPClassifier`, `ProjectedGradientDescent` | MNIST | live |

#### Poisoning & backdoors

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| BadDet · Global Misclassification (GMA) | `BadDetGlobalMisclassificationAttack`, `PoisoningAttackBackdoor` | Synthetic shapes (detection) | live |
| BadDet · Object Disappearance (ODA) | `BadDetObjectDisappearanceAttack`, `PoisoningAttackBackdoor` | Synthetic shapes (detection) | live |
| BadDet · Object Generation (OGA) | `BadDetObjectGenerationAttack`, `PoisoningAttackBackdoor` | Synthetic shapes (detection) | live |
| BadDet · Regional Misclassification (RMA) | `BadDetRegionalMisclassificationAttack`, `PoisoningAttackBackdoor` | Synthetic shapes (detection) | live |
| Clean-Label Backdoor | `PoisoningAttackCleanLabelBackdoor`, `PoisoningAttackBackdoor`, `AdversarialTrainerMadryPGD` | CIFAR-10 | live |
| Feature Collision (Poison Frogs) | `FeatureCollisionAttack` | CIFAR-10 | live |
| Bullseye Polytope | `BullseyePolytopeAttackPyTorch` | MNIST | live |
| Sleeper Agent | `SleeperAgentAttack` | CIFAR-10 | live |
| Witches' Brew (gradient matching) | `GradientMatchingAttack` | CIFAR-10 | live |
| Lambda-layer backdoor | `PyTorchClassifier` | CIFAR-10 | live |
| Neural payload injection | `PyTorchClassifier` | CIFAR-10 | live |

#### Defence · poison detection

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Activation Clustering | `ActivationDefence` | CIFAR-10 | live |
| Spectral Signature | `SpectralSignatureDefense` | CIFAR-10 | live |
| STRIP (STRong Intentional Perturbation) | `STRIP`, `PoisoningAttackBackdoor` | MNIST | live |

#### Privacy · inference attacks

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| Attribute Inference (black-box) | `AttributeInferenceBlackBox` | Synthetic tabular | live |
| Membership Inference (black-box) | `MembershipInferenceBlackBox`, `AttributeInferenceMembership` | Synthetic tabular | live |
| Database Reconstruction | `DatabaseReconstruction`, `ScikitlearnLogisticRegression` | Iris | live |
| Model Inversion (MIFace) | `MIFace` | Fashion-MNIST | precomputed |

#### Model extraction

| Experiment | ART components | Data | Space mode |
|---|---|---|---|
| CopycatCNN | `CopycatCNN` | MNIST | live |
| Knockoff Nets | `KnockoffNets` | MNIST | live |

Not ported: *Faster Genetic Alzantot* and *Improved Genetic Algorithm* — they use TextAttack with a BERT model
rather than ART, and would add several GB of dependencies. The LLM/tooling notebooks (garak, promptfoo,
llm-guard, NeMo Guardrails, modelscan, fickling, pickle attacks, code-shield, …) are out of scope by design.

### Differences from the notebooks

All TensorFlow/Keras models were rewritten in PyTorch so that the image ships a single framework. Architectures and
hyper-parameters follow the notebooks. Where a notebook (or ART itself) had a bug, or a setting that made the demo
meaningless, the Space uses a working setting and says so in the experiment's *Details* panel:

| Experiment | Issue found | What the Space does |
|---|---|---|
| CopycatCNN / Knockoff Nets | victim and thief were passed the wrong way round (stolen models stayed at ~10%) | correct roles: Copycat 93.6%; Knockoff queries only Fashion-MNIST (transfer set) and still reaches ~50% |
| Bullseye Polytope | ART 1.20.1 never uses `self.target` (poisons were optimised towards themselves) | small subclass implementing the paper's objective |
| AWP / TRADES | runs collapsed to 10% / gained no robustness | TRADES-style objective, 20 epochs: ≈42% PGD-10 accuracy (standard model: 13%) |
| LowProFool | ART's `y` is the *target*; the true labels were passed | each sample targets its second most likely class |
| Targeted UAP | unrelated labels and an untargeted inner FGSM (perturbation never grew) | targeted inner FGSM, 99% targeted success |
| HCLU | scored `argmax` of the adversarial *features* | GP predictions, with ART's reversed class order |
| Clean-label backdoor | test set indexed with an integer mask | proper non-target mask; Madry proxy with ε = 8/255 |
| Wasserstein | default regularization overflows | settings from ART's Wasserstein example |
| Feature Collision | learning rate tuned for 0–255 pixels | learning rate 5 for [0, 1] images |
| Neural payload / Lambda backdoor | detector untrained; Keras `Lambda` code in the model file | trained trigger detector; backdoors as explicit, audited wrapper modules |
| TVM | TVM-trained model scores *above* its clean accuracy on FGSM data | explained as label leaking; test-time denoising shown too |
| Database Reconstruction | ART's finite-difference search stays at its initial guess | shown transparently next to the target and the initial guess |

Other notes: MNIST dense models are trained on the full training set (several notebooks used 100 samples);
ART's FBF trainer needs a non-`Sequential` model and its certification converters need a `Flatten`-free layout
(both handled in `robustlab/models/architectures.py`); diffprivlib 0.6.6 is incompatible with scikit-learn 1.8, so its
DP model is trained locally and exported as plain coefficients.

## Run locally

A `Makefile` wraps the common tasks (`make` lists them all):

```bash
make venv && source .venv/bin/activate
make install-train      # runtime + training extras (install a CUDA torch build first for GPU training)

make data               # sample banks (downloads MNIST/CIFAR/Fashion-MNIST/Imagenette once)
make train              # all missing models, poisons and patches (GPU recommended)
make precompute         # results for every experiment (defaults)

make check              # pyflakes + unit tests
make smoke              # every live experiment on CPU with the Space's settings
make run                # http://localhost:7860, same settings as the Space
make dev                # auto-reloading app on port 8501
make docker-build && make docker-run
make clean              # caches, logs and temp files (never touches artifacts/)
```

Selective runs: `make train RECIPES="cifar_cnn cifar_simple_awp" FORCE=1`, `make precompute IDS="fgm strip" FORCE=1`,
`make smoke IDS=fgm`. `PYTHON=`, `PORT=` and `IMAGE=` can be overridden.

## Deploy

1. Upload the artifacts (a write token is needed once: `hf auth login`):
   ```bash
   make upload REPO=<user>/art-robustness-artifacts
   ```
   The command prints the commit hash of the upload.
2. Create a **Docker** Space and push this repository (without `artifacts/`, which `.gitignore` excludes).
3. In the Space settings add the variables
   * `ARTIFACTS_REPO` = `<user>/art-robustness-artifacts`
   * `ARTIFACTS_REVISION` = the printed commit hash (recommended: pins a reviewed artifact version)
   * `HF_TOKEN` as a **secret** only if the artifact repository is private.

The free CPU Space (2 vCPU / 16 GB) is sufficient. Artifacts are downloaded lazily on first use (~0.6 GB in total,
most of it ResNet-50 and the four detectors) and cached for the lifetime of the container.

## Security design

* **No visitor input.** No uploaders or text fields exist; every widget maps to a `Param` whose value is re-validated
  server-side (`Param.coerce`: whitelisted choices, clamped and step-snapped numbers).
* **No code execution from artifacts.** PyTorch weights are *safetensors*, scikit-learn models are *skops* files loaded
  with an explicit trusted-type allow-list, the GP is rebuilt from numeric arrays (GPy's own loader `eval`s class names),
  data and results are `.npz` loaded with `allow_pickle=False` plus JSON. `torch.load`/pickle is never used.
* **Confined file access.** Artifact names are validated against fixed top-level folders and a strict character set.
* **No external targets.** The only network access is `hf_hub_download` from the configured repository. Nothing is
  attacked except the Space's own models; torchvision models are built with `weights=None`.
* **Bounded resources.** One live run at a time (global lock), capped sample counts, results cached per parameter set.
* **Hardened Streamlit config.** XSRF protection on, uploads capped, error details hidden, usage stats off,
  non-root container user.
* Several models in the artifact repository are **intentionally backdoored** — they are for demonstration only.
