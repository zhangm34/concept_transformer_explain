#!/bin/bash
#SBATCH --job-name=concept-cub-l4
#SBATCH --account=yonghui.wu
#SBATCH --qos=yonghui.wu
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=training_runs/slurm-%j.out
#SBATCH --error=training_runs/slurm-%j.err
#SBATCH --export=NONE

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the prepared project on Blue}"
case "$PWD" in
    /blue/*) ;;
    *) echo "Submit from the project working copy on /blue." >&2; exit 1 ;;
esac
unset PYTHONHOME PYTHONPATH
export PATH=/opt/slurm/bin:/usr/bin:/bin
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
PYTHON="$PWD/.venv/bin/python"
test -x "$PYTHON"
# The wheel bundles CUDA 11.8; no CUDA toolkit or Apex module is required.
# Keep CUDA_VISIBLE_DEVICES as assigned by Slurm.
srun --ntasks=1 "$PYTHON" - <<'GPU_CHECK'
import torch
assert torch.cuda.is_available(), "No CUDA device visible in the allocated job"
assert torch.cuda.device_count() == 1, "Expected one Slurm-allocated GPU"
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.float16):
    loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print("CUDA mixed precision forward/backward passed")
GPU_CHECK
srun --ntasks=1 "$PYTHON" scripts/run_training.py \
    --run-dir "training_runs/l4-${SLURM_JOB_ID}" \
    --batch-size 32 --max-epochs 50 --num-workers 2 --require-cuda
