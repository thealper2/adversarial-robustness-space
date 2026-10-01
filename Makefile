# ART Robustness Lab — common tasks. Run `make` (or `make help`) to list them.
# Uses the active virtualenv if there is one, otherwise ./.venv (create it with `make venv`).

VENV     ?= .venv
PYTHON   ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3))
PORT     ?= 7860
IMAGE    ?= art-robustness-lab
REPO     ?=
RECIPES  ?=
IDS      ?=
FORCE    ?=

# Settings the Space uses: CPU only, two threads.
SPACE_ENV = CUDA_VISIBLE_DEVICES= ROBUSTLAB_DEVICE=cpu ROBUSTLAB_THREADS=2 OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore

.DEFAULT_GOAL := help
.PHONY: help venv install install-train data train precompute smoke test lint check run dev \
        docker-build docker-run upload clean distclean

help: ## List the available targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Variables: PYTHON=$(PYTHON) PORT=$(PORT) IMAGE=$(IMAGE)"
	@echo "  Examples:  make train RECIPES=\"cifar_cnn cifar_simple_awp\" FORCE=1 · make precompute IDS=fgm · make upload REPO=user/art-robustness-artifacts"

## --- setup ---------------------------------------------------------------------------------------

venv: ## Create ./.venv (then: source .venv/bin/activate)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	@echo "Created $(VENV). Activate it with: source $(VENV)/bin/activate"

install: ## Install the Space runtime dependencies (CPU torch)
	$(PYTHON) -m pip install -r requirements.txt

install-train: ## Install runtime + local training extras (install a CUDA torch first for GPU training)
	$(PYTHON) -m pip install -r requirements-train.txt pytest pyflakes

## --- offline pipeline (local machine) ------------------------------------------------------------

data: ## Build the sample banks in artifacts/data (downloads datasets once)
	$(PYTHON) -m training.prepare_data

train: ## Train missing models (RECIPES="a b" to select, FORCE=1 to retrain)
	PYTHONWARNINGS=ignore $(PYTHON) -m training.train $(if $(FORCE),--force) $(RECIPES)

precompute: ## Precompute results for missing experiments (IDS="a b", FORCE=1 to redo)
	PYTHONWARNINGS=ignore ROBUSTLAB_THREADS=2 $(PYTHON) -m training.precompute $(if $(FORCE),--force) $(IDS)

upload: ## Upload artifacts to the Hugging Face Hub (REPO=user/name, needs `hf auth login`)
	@test -n "$(REPO)" || (echo "Usage: make upload REPO=<user>/art-robustness-artifacts" && exit 1)
	$(PYTHON) -m training.upload --repo $(REPO)

## --- checks --------------------------------------------------------------------------------------

test: ## Run the unit tests
	$(PYTHON) -m pytest -q tests

lint: ## Static checks (pyflakes); side-effect imports that register experiments are ignored
	@out=$$($(PYTHON) -m pyflakes app.py robustlab training tests | grep -vE "experiments/__init__.py|'robustlab\.experiments' imported"); 	if [ -n "$$out" ]; then echo "$$out"; exit 1; else echo "lint ok"; fi

smoke: ## Run every live experiment with its defaults on CPU, like the Space (IDS="a b" to select)
	$(SPACE_ENV) $(PYTHON) -m training.smoke $(IDS)

check: lint test ## Lint + unit tests

## --- run -----------------------------------------------------------------------------------------

run: ## Start the app like the Space does (CPU, http://localhost:7860, PORT=... to change)
	$(SPACE_ENV) $(PYTHON) -m streamlit run app.py --server.port=$(PORT) --server.address=0.0.0.0

dev: ## Start the app with auto-reload on file changes (port 8501)
	$(SPACE_ENV) $(PYTHON) -m streamlit run app.py --server.port=8501 --server.fileWatcherType=auto --server.runOnSave=true

docker-build: ## Build the Space image
	docker build -t $(IMAGE) .

docker-run: ## Run the image with local artifacts mounted read-only (http://localhost:7860)
	docker run --rm -it -p $(PORT):7860 -v "$(CURDIR)/artifacts:/home/user/app/artifacts:ro" $(IMAGE)

## --- cleanup -------------------------------------------------------------------------------------

clean: ## Remove caches, logs and temporary files (keeps artifacts/)
	find . -path ./$(VENV) -prune -o -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -path ./$(VENV) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' \) -exec rm -f {} +
	rm -rf logs build dist *.egg-info

distclean: clean ## Also remove ./.venv (artifacts/ is never deleted by make)
	rm -rf $(VENV)
