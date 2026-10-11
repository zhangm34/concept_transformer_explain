# Task plan

## Paper and model explanation (2026-10-11)

21. Read the paper and verify global/spatial attention semantics — complete
22. Inspect the first correct notebook example and its signed contributions — complete
23. Explain interpretability, concept discovery, and zero-shot extensions in plain language — complete

Errors in this explanation review: the sandbox startup permission issue persists, so necessary shell reads use escalation. OpenReview currently presents a browser challenge; use the complete local paper. The example metadata key is `examples`, not `samples`; correct the focused reader.

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

## Test explanation visualizations (2026-10-10)

User request: visualize the paper's explainability with several correct and incorrect test predictions.

15. Inspect current notebook, paper, completed checkpoints, and test split — complete
16. Extract concept attention and choose traceable correct/incorrect examples — complete
17. Generate figures and add an executed explanation section to the notebook — complete
18. Verify sample identities, explanation semantics, figures, and notebook preservation — complete

Current evidence: the notebook now records a second completed disjoint-split run (job 45478721), newer than the previous planning notes. Prefer its selected checkpoint and saved test IDs for these examples. Sandbox startup still fails; filesystem commands require escalation.

Completed 2026-10-11: Section 4.10 contains eight new cells, four executed code cells, and seven inline figures. Six detailed PNG/PDF pairs, an overview PNG/PDF, a local gallery, cached arrays, provenance, and verification.json are saved in explanation_artifacts/disjoint-45478721/. All 57 existing cells and all prior metadata are preserved. Official sample IDs/labels/test membership, probabilities, attention normalization, complete additive logit reconstruction, full PNG decoding, PDF signatures, notebook schema, and whitespace checks passed.

## Regroup results to match the assessment (2026-10-11)

User request: regroup the existing results into Section 5, following the assessment outline.

19. Determine section boundaries and update the notebook organization — complete
20. Update cross-references and verify preserved code, outputs, and dependencies — complete

Scope: reorganize and renumber existing results/interpretation material, keep executed figures and metrics, and place existing future-experiment discussion after the results. No rerunning training or inference is needed for this editorial change.

- Stopped an overly broad supplemental AGENTS.md search; use only explicit ancestor paths and scoped project directories for any further instruction checks.

Regrouping complete: Section 5 now contains 5.1 first-run results, 5.2 disjoint-baseline results, 5.3 comparison, and 5.4 concept explanations. Existing future-experiment Markdown moved into Section 6 Discussion (6.1–6.4). All 65 original cell IDs, 28 code cells in their original execution order, numeric outputs, and 16 inline figures are preserved; only one printed section-label caption changed. Updated notebook/README references. Notebook schema, preservation assertions, headings, absence of stale result references, and git diff --check passed; no code execution was needed.
