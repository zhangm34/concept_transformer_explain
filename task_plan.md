# Task plan

Goal: rebuild the HPG environment, download and verify CUB, replace obsolete notebook setup, and provide a verified L4 Slurm submission example without submitting training.

1. Review notebook, dependencies, and cluster configuration — complete
2. Download and verify data with requested subagent — complete
3. Recreate Linux venv and verify CPU pipeline — complete
4. Replace notebook sections and setup documentation — complete
5. Validate L4 batch script and report submission procedure — complete

## Errors
- Sandbox process cannot start: bwrap cannot create temporary directory under /local/scratch. Commands require escalation.

- Bolts 0.6.0 conflicts with Lightning 1.9.5 lightning-utilities constraints; use Bolts 0.7.0.
- Batch-script generation failed due to nested heredoc delimiter collision; use distinct delimiters. No training started.

- First notebook run passed loader/crops/train/validation; Lightning 1.9 test requires explicit datamodule. Fixed test call with model, datamodule, and None/best checkpoint selection.
- CPU trainer check compared string precision to integer; corrected validation to int(trainer.precision).

## Final verification
- All 10 code cells executed and saved without errors.
- CPU debug training, validation, and testing passed in 63.1 seconds.
- Source archive/images/annotations and all generated crops verified.
- Normal CPU trainer construction and require-cuda refusal passed.
- Slurm L4 test-only validation accepted the request; no training job queued/submitted.
- GPU execution is intentionally unverified until an L4 allocation is used.

- Lock check initially selected the loaded Conda module Python 3.14; explicitly select .venv/bin/python, as in documented setup commands.

## Authorized submission
6. Prepare Blue working copy and recreate its venv — complete
7. Submit the validated single-L4 training job and verify scheduler state — complete

- A staging-progress du scan raced with an rsync temporary filename; this does not affect the copy. Use final data checks after rsync exits.

- Current test-only reported resources temporarily unavailable. Actual sbatch automatically retried controller acceptance and succeeded; no duplicate submission.
- Blue env/data/crops/weights/source consistency checks passed. Job 45377754 submitted; queue snapshot: 45377754|PENDING|(Priority).

- Paper inspection: pdftotext unavailable; extracted local PDF with installed pypdf instead.

## Status check and training recovery
8. Fix and exercise timing callback through train/validation/test hooks — complete
9. Retry unchanged L4 configuration, check status, and estimate time from measured throughput — complete

- Job 45377754 failed before its first training batch: Lightning 1.9 train hooks omit dataloader_idx. Add compatible default to timing callback hook signatures. GPU mixed precision check and sanity validation passed.

- Actual callback fit/validation/test passed in both environments. The Orange check finished just before the attempted stop; no process was killed. Retried on Blue to record the exact staged runner hash.

- Replacement job 45387301 failed in newly added runner hash metadata because Python 3.8 __file__ was relative after os.chdir. Use absolute PROJECT_DIR path and exercise actual runner initialization and fit/test end-to-end.

- Job 45387524 RUNNING on L4; 147 training batches complete with finite losses. Median wall time 0.651 s/batch. Early ETA saved to setup_logs/training-eta.json.
- Reading an actively written UTF-8 progress log raced with a partial multibyte glyph; use errors=replace for live log reads.

## Notebook revision
9. Review assessment and source notebook — in_progress
10. Verify source claims and recorded results — pending
11. Create and execute separate concise revision — pending

Revision review: source read; focused read retried after incorrect cvit.py filename; PDF read with system Python because project env lacks pypdf. Upstream loss retrieved successfully.

Revision status: review and evidence checks complete; separate notebook saved. Execution stopped following user clarification; final GPU results deferred until ongoing training completes.

## Completed-training hyperparameter review (2026-10-10)

User request: inspect the successful training and annotate the notebook with hyperparameters to improve performance, reasons, and tradeoffs.

12. Inspect final run artifacts, learning curves, and implemented hyperparameters — complete
13. Prioritize evidence-based experiments and verify technical references — complete
14. Add recommendations to the notebook and validate notebook integrity — complete

Current constraint: sandbox startup still fails with a temporary-directory permission error; use approved escalated commands for necessary filesystem inspection.

- One version-specific Lightning documentation URL could not be fetched (cache miss); callback behavior is available in the installed source. Bolts scheduler documentation and the AdamW paper are accessible.
- Final diff check initially expected one removed closing-brace line, but difflib reported zero removed lines (a pure insertion). Adjusted the verification to accept either valid alignment; notebook schema and semantic-preservation checks had already passed.

Hyperparameter review complete: Section 4.7 added to the main notebook with evidence, candidate values, reasons, tradeoffs, and a controlled evaluation plan. Notebook schema, numerical evidence, Markdown tables, and local links passed validation. All original 39 cells, code outputs, and embedded metadata preserved; no training executed.
