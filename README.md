# Concept Transformer assessment

The walkthrough currently covers Step 2, environment setup, and Step 3,
manual data download and preparation, plus Step 4's small debug training run.
We will add the remaining assessment steps as we work through them together.

Use uv with the project-local `.venv`, running **Python 3.8.20 under Rosetta
(x86_64)**. This architecture was chosen because the original SciPy 1.7.1 and
pandas 1.3.3 pins lack macOS ARM64 wheels. The original PyTorch 1.10.0 stack
uses **CPU** on this Mac; it predates Apple's MPS backend.

To recreate the environment on an Apple Silicon Mac with Rosetta installed,
run from this directory:

```bash
uv python install cpython-3.8.20-macos-x86_64-none
uv venv --python cpython-3.8.20-macos-x86_64-none .venv
uv sync --locked --python .venv/bin/python
uv run --locked python -m ipykernel install --sys-prefix \
  --name concept_transformer --display-name "Concept Transformer (Python 3.8 Rosetta)"
uv run --locked jupyter lab concept_transformer_walkthrough.ipynb
```

Select **Concept Transformer (Python 3.8 Rosetta)** and run the notebook from
the top. It checks the interpreter, package versions, repository imports,
PyTorch devices, and a small forward/backward computation. All notebook
project paths are relative to the configurable `PROJECT_DIR`.

Model package versions match `concept_transformer/requirements.txt`.
**NumPy differs between upstream documents:** the GitHub README lists 1.20.3,
while requirements.txt pins 1.22.0. We use 1.22.0, following the README's
instruction to install from requirements.txt. This is documented in the
notebook's compatibility log.

Upstream leaves epyc unpinned; we select 1.7.2. Additional compatibility pins
and notebook tools are specified in `pyproject.toml`. `uv.lock` records the
complete dependency resolution. Keep that file and use `uv sync --locked`
for reproducible installs. If changing dependencies, use `uv sync` to update
the lock, then restart the notebook kernel.

The earlier Python 3.12 environment is retained at `.venv-py312-backup`.
Installation logs are in `setup_logs/`. Environment verification does not
establish that a full CUB training run or the paper's metrics have been
reproduced; those are later assessment steps.

Verified: pip dependency consistency, repository imports, `cvit_cub.py --help`,
original Lightning Trainer construction, and PyTorch CPU forward/backward
computation. Notebook cells are executed with their outputs saved.

The manually downloaded CUB data is under `CUB-200-2011/`. Set `DATA_DIR` to
`CUB-200-2011/CUB_200_2011` (the outer directory), as the loader appends
`CUB_200_2011/` itself. All 11,788 images and required source annotations
are present. The user downloaded the data manually from the official Caltech
dataset page linked in the assessment:
https://www.vision.caltech.edu/datasets/cub_200_2011/.
The notebook copies the outer `attributes.txt` into the inner dataset
directory and validates this placement with the original loader. All
11,788 bounding-box crops have now been generated with the unchanged
repository code and `download=False`: 46 seconds for metadata loading and
generation, 49 seconds including verification, and 132.90 MiB of output.
All crops passed Pillow verification; 200 representative crops fully
decoded, and a transformed loader sample passed its shape checks. The
notebook quotes the original preparation code and reuses a complete crop
cache on subsequent runs. Audit and log are saved in
`setup_logs/cub-cropping.json` and `setup_logs/cub-cropping.log`.
Pretrained ViT weights are now cached under `.cache/torch/`. A small CPU
debug run completed training, validation, and testing with batch size 2
and no data-loader workers in approximately 50 seconds. Regular training has
started with the repository-default batch size 32 and 50 epochs, after
an initial regular batch-size-2 run was interrupted at the user's request.
This repository records a work-in-progress snapshot, not completed training
results. Segmentation masks are present but unused by this loader.


To repeat the debug run:

```bash
uv run --locked python scripts/run_debug.py
```

The runner saves the exact CLI command and raw output under
`training_runs/debug/`. Batch size 1 exposes an upstream spatial-loss
mask-shape bug, so we use batch size 2. One debug-only line in
`concept_transformer/cvit_cub.py` explicitly passes the in-memory model to
`trainer.test`, because Lightning disables checkpoint saving in debug
mode. Normal runs retain the original evaluation call. The change is
recorded in `training_runs/debug/code.diff`; both failed attempts are
also preserved there in separate subdirectories. The notebook documents
these issues and displays the successful run's one-batch metrics. These
metrics are a pipeline check, not an estimate of dataset performance.

The notebook reuses the recorded successful debug run by default. Set
`RERUN_DEBUG = True` in Section 4 to run it again. If no run record exists,
that cell executes the debug runner automatically.


This repository vendors the IBM source at the revision in `upstream.json`.
The upstream Apache-2.0 license is retained in `concept_transformer/LICENSE`,
and our debug-only change is recorded in
`patches/debug-test-no-checkpoint.patch`. The upstream Git metadata and
MNIST model checkpoints are not included. The provenance manifest lets
the notebook and runners identify the upstream revision after cloning.

Dataset files, virtual environments, downloaded weights, and training
checkpoints are excluded from Git. To reproduce the data setup:

```bash
uv run --locked python scripts/download_cub.py
uv run --locked python scripts/download_cub.py --check-only
```

The script downloads the official Caltech archive, checks its published
MD5, extracts it into the loader-compatible directory structure, copies
`attributes.txt` into place, and verifies image paths and required
annotations. If the data is already present, it reuses it. The notebook
then generates bounding-box crops. Optional segmentation masks are not
needed by this model. The script's local verification mode has been run;
we have not repeated the approximately 1.2 GB download.

To run regular training with the repository batch size (32):

```bash
uv run --locked python scripts/run_training.py \
  --run-dir training_runs/my-run --batch-size 32 --max-epochs 50
```

The runner adds only a timing callback and retains the repository's
training configuration. It records per-batch timing and loss in
`batch_timings.jsonl`; TensorBoard logs and best-model checkpoints are
created by the original training code. The committed regular-run logs
and metadata are snapshots captured while training is in progress.
`training_runs/active_run.json` contains the originating machine's run
path and PID, not a process that starts when the repository is cloned.
