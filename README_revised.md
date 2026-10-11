# Revised assessment notebook

Open `concept_transformer_walkthrough_revised.ipynb` from the project directory using the project `.venv` kernel. The original notebook is preserved.

Recreate the tested Linux environment:

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export UV_CACHE_DIR="$PWD/.uv-cache"
uv python install 3.8.20
uv venv --python 3.8.20 .venv
uv sync --locked --python .venv/bin/python
.venv/bin/python -m ipykernel install --sys-prefix --name concept_transformer
.venv/bin/jupyter lab concept_transformer_walkthrough_revised.ipynb
```

Configure `PROJECT_DIR`, `DATA_DIR`, and `ARTIFACT_DIR` in the first code cell. Obtain CUB using `scripts/download_cub.py`; the notebook can invoke this when data is absent, prepare crops, and verify them. Network access is needed for the initial dataset and pretrained model downloads.

`revision_artifacts/` contains the recorded successful CPU debug run and a fixed snapshot of GPU job 45387524 captured at 2026-10-10T02:09:21 UTC. GPU CSVs contain actual TensorBoard scalars and timing-callback batch losses. The run was training at capture time; the snapshot includes 17 complete training epochs and part of the next epoch. Final GPU test metrics are unavailable. The data-sample plot shows annotations, not learned attention.

Set `RERUN_DEBUG=True` to rerun the full model for one training, validation, and test batch. This updates `training_runs/debug/`; the original notebook and frozen revision artifacts remain untouched. Full GPU training uses `scripts/train_l4.sbatch` from a prepared Blue project directory; notebook execution does not submit a job.

Share the notebook, this note, `revision_artifacts/`, `concept_transformer/`, `scripts/`, `patches/`, `upstream.json`, `pyproject.toml`, `uv.lock`, and the small `setup_logs/cub-download.json` source-audit record. Dataset files and model checkpoints are not part of the deliverable. The runtime is adapted from the upstream dependency stack, and its validation/test splits overlap as documented in the notebook.
