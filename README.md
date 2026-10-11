# Concept Transformer assessment

The executed `concept_transformer_walkthrough.ipynb` covers environment setup, official CUB download/preparation, and a one-batch CPU debug training/evaluation run. Full GPU training and reproduction of the paper metrics remain pending.

The project uses **Linux x86_64 Python 3.8.20** in a project-local uv `.venv`. The installed **PyTorch 2.0.1+cu118 / torchvision 0.15.2+cu118** wheels bundle CUDA 11.8 and also run on CPU-only nodes. Lightning 1.9.5, Bolts 0.7.0, and TorchMetrics 0.11.4 form the selected training runtime. Scientific/data/model dependencies retain their selected original pins. All differences from upstream requirements are listed in the notebook.

Recreate and launch from this directory:

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export UV_CACHE_DIR="$PWD/.uv-cache"
uv python install 3.8.20
uv venv --python 3.8.20 .venv
uv sync --locked --python .venv/bin/python
.venv/bin/python -m ipykernel install --sys-prefix \
  --name concept_transformer --display-name "Concept Transformer (HPG Python 3.8)"
.venv/bin/jupyter lab concept_transformer_walkthrough.ipynb
```

Select **Concept Transformer (HPG Python 3.8)** and run from the top. This kernelspec is installed inside the venv; launch Jupyter with that venv, or configure OnDemand to use it. Package consistency, repository imports, CLI help, and tensor forward/backward checks precede data and training checks. `uv.lock` fixes the complete Linux dependency resolution. Restart the kernel after any dependency changes.

The runtime uses Lightning's supported accelerator/device API and native mixed precision on CUDA. Model-summary/progress callbacks replace removed trainer flags, and accuracy uses the explicit multiclass task. The architecture, losses, AdamW optimizer, augmentation, and default model hyperparameters retain the repository implementation. Source changes are recorded in `patches/hpg-runtime.patch` and `patches/debug-test-no-checkpoint.patch`. This runtime differs from the original dependency stack and has only been exercised on CPU here.

NumPy uses **1.22.0** from upstream `requirements.txt`; upstream README lists 1.20.3. The complete selected pins are in `pyproject.toml`. The IBM revision is recorded in `upstream.json`; the Apache-2.0 license is retained in `concept_transformer/LICENSE`.

Download and verify the official CUB archive:

```bash
.venv/bin/python scripts/download_cub.py
.venv/bin/python scripts/download_cub.py --check-only
.venv/bin/python scripts/verify_cub.py  # full archive/image/annotation audit
```

The archive is cached at `.cache/downloads/CUB_200_2011.tgz` and verified against the CaltechDATA published MD5 `97eceeb196236b17998738112f37df78`. The loader root is `CUB-200-2011/CUB_200_2011`, with the actual dataset inside its nested `CUB_200_2011` directory. The downloader copies the outer `attributes.txt` into that inner directory. Optional segmentation masks are unused and omitted.

The fresh download audit is in `setup_logs/cub-download.json` and `.log`: every original image is fully decoded and all annotation IDs/pairs are verified. Source annotation irregularities are recorded without rewriting the official files. The notebook generates all 11,788 bounding-box crops with the repository code, fully decodes each crop, and checks a transformed sample. Current crop audit: `setup_logs/cub-cropping.json` and `.log`. Pretrained ViT-Large weights are cached in `.cache/torch/`.

Repeat the CPU debug check within an allocated CPU session:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/run_debug.py
```

Use thread counts within your Slurm allocation. Debug mode runs one training batch, one validation batch, and one test batch. Batch size 2 avoids the upstream spatial-loss mask bug at batch size 1. Testing explicitly passes the data module; debug mode uses the in-memory model and regular mode selects the best checkpoint. Logs, metrics, current-platform metadata, and runtime/lock hashes are saved under `training_runs/debug/`. The notebook only reuses a record matching its setup. These metrics check pipeline execution and do not estimate dataset performance. The original validation subset overlaps the official test split; this behavior is documented in the notebook.

For regular training, use the prepared **L4 Slurm script** `scripts/train_l4.sbatch`. It requests account/QOS `yonghui.wu`, partition `hpg-turin`, `gpu:l4:1`, four CPUs, 32 GiB memory, and 24 hours. It verifies CUDA mixed precision before starting batch-size-32, 50-epoch training with two data-loader workers. `--require-cuda` prevents an accidental CPU fallback. The runner records Slurm metadata and per-batch timing/loss; normal checkpoint and TensorBoard outputs remain enabled.

HPG requires a **Blue working directory** for computation. Prepare a separate working copy and recreate its venv at its own path:

```bash
SOURCE=/orange/yonghui.wu/zhangm3/2026/concept_transformer_explain
WORK=/blue/yonghui.wu/zhangm3/concept_transformer_explain
mkdir -p "$WORK"
rsync -a --exclude=.venv --exclude=.uv-cache --exclude=.uv-python \
  --exclude=.git --exclude=.cache/downloads --exclude=training_runs \
  "$SOURCE/" "$WORK/"
cd "$WORK"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export UV_CACHE_DIR="$PWD/.uv-cache"
uv python install 3.8.20
uv venv --python 3.8.20 .venv
uv sync --locked --python .venv/bin/python
mkdir -p training_runs
sbatch --test-only scripts/train_l4.sbatch
sbatch scripts/train_l4.sbatch
```

Run the preparation commands in an allocated CPU session with `uv` available. This copies prepared data and cached weights; it excludes the old venv because environments contain absolute paths. Create `training_runs/` before submitting so Slurm can open the output/error files. The batch script preserves Slurm's `CUDA_VISIBLE_DEVICES` and uses the venv directly without an inherited Conda environment or an Apex/CUDA toolkit module.

Monitor with `squeue -u zhangm3`; logs are `training_runs/slurm-<job-id>.out` and `.err`, with training artifacts under `training_runs/l4-<job-id>/`. GPU training job **45387524** is the current submission on `hpg-turin`; use `squeue -j 45387524` for live status. Earlier attempts 45377754 and 45387301 failed in timing-hook and metadata compatibility code before training, and their logs are retained. Both errors are fixed; the actual runner was tested end-to-end with a small CPU model. Hyperparameters remain unchanged. The submission record is `setup_logs/slurm-submission.json`; measured timing estimates, when available, are in `setup_logs/training-eta.json`. Older regular-run records and `training_runs/active_run.json` are historical snapshots; their process IDs do not identify active jobs here. The 24-hour request is an initial walltime allocation, not a measured completion estimate.

Dataset files, virtual environments, downloaded weights, and checkpoints are excluded from Git. Environment audit: `setup_logs/environment.json`; install log: `setup_logs/uv-sync-hpg.log`.

References: [HPG GPU requests](https://docs.rc.ufl.edu/scheduler/gpu_access/) and [HPG computation](https://docs.rc.ufl.edu/quickstart/computation/).

The current notebook's **Section 5.4** shows three correct and three incorrect predictions from the completed disjoint run **45478721**. Each example includes spatial concept heatmaps, global concept attention, and signed contributions to the predicted-versus-comparison class margin. Open `explanation_artifacts/disjoint-45478721/gallery.html` for the standalone gallery; PNG/PDF exports and sample provenance are in the same directory. Displaying cached examples does not load the model. To recompute them from the saved Blue checkpoint and split, run:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/visualize_test_explanations.py
```

Use `--render-only` to regenerate figures from cached tensors. These are CPU float32 predictions from the saved epoch-28 checkpoint, with an exact additive logit reconstruction check; the six examples are qualitative illustrations rather than a full-test explanation-quality metric.
