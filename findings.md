# Findings

- Existing environment pins Python 3.8.20, PyTorch 1.10.0, Lightning 1.4.8; README and notebook assume macOS Rosetta.
- Dataset and venv are absent from this checkout.
- User requested a data-download subagent; data_refresh owns download and source verification.

- Current allocation: CPU-only hpg-default, 4 CPUs, 24 GiB. Limit OMP/MKL/OpenBLAS threads to allocation.
- Live Slurm: hpg-turin has gpu:l4:3 per node, 14-day walltime limit. Account/QOS yonghui.wu.
- Use Python 3.8.20 with torch 2.0.1+cu118, torchvision 0.15.2+cu118, Lightning 1.9.5, Bolts 0.7.0, TorchMetrics 0.11.4. Preserve scientific/data/model pins and document all runtime changes.
- GPU trainer needs native AMP and Lightning 1.9 arguments; accuracy requires task=multiclass.
- HPG documentation requires training work directory under /blue; confirmed /blue/yonghui.wu/zhangm3 exists. Submission example will stage a working copy there.

- Fresh full source audit passed: 11,788 decoded images, 3,677,856 attribute pairs, 176,820 part pairs; 606 trailing-field rows and 8 extending bounding boxes preserved.
- Current crop generation/verification passed with 11,788 fully decoded crops and transformed sample shapes (3,224,224), (13,), (196,95).
- Lightning 1.9 testing requires explicit datamodule; final evaluation supplies model and datamodule, None checkpoint in debug and best checkpoint in normal training.

- Hyperparameter clarification: submitted L4 job follows cvit_cub.py defaults (batch32, epochs50, lr5e-5, weight_decay1e-3, warmup10, explanation1.0, sparsity0.5, seed42). Only loader workers change 8→2; training runtime uses native AMP and updated package versions. Local 16-page paper does not state the complete optimizer/scheduler/batch configuration.

- Real L4 training throughput: 0.651 s/batch including inter-batch loading/logging, 0.533 s compute, 187 train batches/epoch, 32 validation batches/epoch. Early remaining ETA includes 10–40% checkpoint/variability allowance.

- Failure provenance: job 45377754 callback signature and job 45387301 runner-hash path came from our local additions, not original IBM code. Upstream issues remain batch-size-1 spatial-loss squeeze and original debug evaluation checkpoint assumption; explicit datamodule requirement results from our Lightning upgrade.

## Revision requirements
Assessment requires six sections, executed outputs, configurable paths, recorded commands/metrics/visuals, and 3–5 proposed extensions. Exclude locally introduced failures from revision. Original Blue notebook SHA256: 299f03352e2fa8d9a3819419510a8b4978e3911bf4a2342814e63bf0cd908b25.

Revision evidence: upstream ctc.py byte match verified; synthetic tensor reproduction raises batch-1 IndexError and returns 0.0 loss for batch 2. Captured GPU job 45387524 at 2026-10-10T02:09:21 UTC: 3,331 train batches, 17 complete train epochs; no final test metrics. Frozen artifacts under revision_artifacts/. Removed local-runner failure history and unsupported debug-checkpoint bug attribution.

## Completed-run tuning review (2026-10-10)

- The main `concept_transformer_walkthrough.ipynb` now has 39 cells with executed final results and embedded `completed_l4_training` metadata. This is the target notebook. The revised notebook is currently a four-cell outline; older revision artifacts are incomplete snapshots.
- Main notebook records job 45387524 complete (50 epochs, 9,350 train batches), 01:50:04 elapsed, best validation-CE checkpoint at epoch 30, and test accuracy 0.8733172416687012. Validation uses 1,000 official test images and overlaps final testing; tuning needs a separate validation split.
- Existing runner only accepts run directory, epochs, batch size, loader workers, and require-cuda. Model hyperparameter flags exist in `concept_transformer/cvit_cub.py`; any suggested command must distinguish this entrypoint from the wrapper.
- Verified source run `run.json` reports complete. Epoch 30→50: train CE 0.657577→0.430191, validation CE 0.785258→0.815805; train explanation 0.458310→0.389278, validation explanation 0.525119→0.517565. Mild late generalization divergence; more epochs is not the first recommendation.
- Validation accuracy is 86.9% at epoch 30, peaks at 87.9% at epochs 41/42/45, and finishes at 87.6%. CE-based early stopping can miss later accuracy improvements; no saved test measurement for the peak-accuracy checkpoints was established.
- `concepts_sparsity_cost` is mean negative p-log-p entropy, not an L1 norm despite `*_l1_loss` metric names. At epoch 50 weighted train terms are CE 0.4302, explanation 0.3893, entropy 0.0780. Loss magnitudes alone do not establish gradient conflict or optimal weights.
- CUB head attention/projection dropout are hardcoded to 0.1. Optimizer is one AdamW parameter group over all trainable parameters. Warmup is epoch-stepped with default starting LR zero; measured first-epoch mean LR is near zero. Early-stopping callback support exists but no CUB CLI flag exposes patience. Changing max_epochs also changes cosine schedule.
- Primary references checked: AdamW paper https://arxiv.org/abs/1711.05101 and Bolts 0.7 scheduler documentation https://lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html . Proposed numeric ranges are experimental choices inferred from this run, not established improvements from these references.
