# Progress

## Paper explanation review (2026-10-11)

- Completed inspection of Section 5.4, first-example PNG, exact cached attention/contributions, Equation 2, Proposition 1, and Section 4.3's extension claim. Prepared the explanation using the reproduced example without changing notebook contents or running training/inference.

- Read existing reproduction notes, extracted the local 16-page paper to a temporary text file, and inspected the model and figure implementation. Preparing a worked explanation of the first correct prediction and checking the paper's extension claim.

- Reviewed root documentation and download/training scripts.
- Delegated dataset download and verification.

- Official data extracted by subagent; complete source audit in progress.
- Linux runtime lock resolved with Bolts 0.7.0; wheel installation underway.
- Updated notebook sections in place and prepared a Blue-based L4 job example.

- All source images and annotations verified by requested subagent.
- Generated and fully decoded all 11,788 crops; transformed loader sample passed.
- Slurm test-only accepted L4 request (no submission).
- Corrected explicit evaluation datamodule requirement discovered by CPU debug run.

- Completed notebook execution: all 10 code cells pass; CPU debug run completed in 63.1 s with finite metrics.
- Verified notebook source no longer contains obsolete platform instructions, matching environment and source/lock hashes, full dataset/crop audits, and Slurm dry-run evidence.
- Final script compilation/syntax and relevant-source whitespace checks passed. Raw console logs and patch copies retain original formatting.
- No GPU training submission performed. Blue staging/recreation procedure is documented and ready for user review.

- User authorized actual submission. Preparing Blue working copy and its own environment.

- Blue working copy and separately recreated Python 3.8.20 venv ready. Package consistency and all staged image/crop paths passed.
- Submitted L4 job 45377754; queue snapshot: 45377754|PENDING|(Priority).
- Saved submission record in Orange and Blue setup_logs/slurm-submission.json; updated notebook training status and README.

- Status query found job 45377754 FAILED after 37 seconds. Cause: required dataloader_idx in timing callback train hook. No training batch completed. GPU startup/validation succeeded.

- Tested corrected timing callback through 2 train batches, sanity validation, 2 validation batches and 2 test batches. Added runner source hash to run records; hyperparameters unchanged.

- Job 45387301 exposed a relative source-hash path after chdir; corrected to the absolute project path.
- Verified the actual runner startup, metadata, CPU train/validation/test callbacks end-to-end with a small model.
- Resubmitted unchanged configuration as job 45387524. Waiting for throughput/status evidence.

- Recovery successful: 45387524 running on allocated L4, 147 batches complete, median wall time 0.651 s/batch; actual losses finite. Early remaining estimate 1.92–2.44 h saved with timestamp and assumptions.

- User requested upstream attribution where applicable. Notebook now explicitly attributes both Slurm failures to local runner additions, and separately identifies upstream batch-size-1 spatial-loss and debug-checkpoint issues. Both notebook copies updated.

Revision review: source read; focused read retried after incorrect cvit.py filename; PDF read with system Python because project env lacks pypdf. Upstream loss retrieved successfully.

Created separate 18-cell revised notebook and README_revised.md; copied fixed CPU/GPU evidence under revision_artifacts/. Running all cells using project Python kernel.

User clarified training is ongoing and results will be completed later. Stopped only our notebook execution process (PID 824665); cleanup ended with an assertion after interruption and no executed outputs were saved. Original Blue notebook hash unchanged. Revision clearly marked as unexecuted; no training submitted or restarted.

- 2026-10-10: User reports successful completion and requests notebook recommendations for hyperparameter improvements, reasons, and tradeoffs. Restored prior project notes; beginning final-run inspection. Sandbox startup remains unavailable; read commands require escalation.
- Reviewed final notebook evidence and original Blue run status, derived late-training diagnostics, and checked the optimizer/loss/scheduler/data implementation. Chose the main completed-results notebook for a new Section 4.7 with ranked tuning hypotheses and a controlled experiment plan.
- Added four Markdown cells in Section 4.7: observed diagnostics; six ranked settings with candidate values, reasons, and tradeoffs; secondary dropout/rate-group/batch options; and a fixed-split, staged, multi-seed evaluation plan. Explained entropy-versus-L1 terminology, checkpoint metric choice, scheduler interactions, and missing CLI controls. Existing 39 cells and embedded results are preserved. Notebook schema and evidence validation is running; no training launched.
- Completed validation: valid nbformat 4.5, accurate quoted epoch metrics and compute estimate, valid local links, consistent Markdown tables, unique cell IDs, and exact preservation of all original code cells, outputs, metadata, and their formatting. Final edit is an insertion of four Markdown cells; recommendations are explicitly untested hypotheses. No new training or source-code changes.

- 2026-10-10: Started requested correct/incorrect prediction explanation visualizations. Restored planning notes and inspected current notebook; it includes a newer completed disjoint rerun (45478721). Inspecting the actual concept attention, spatial concept mapping, and checkpoint/split provenance before plotting.
- Added a reproducible CPU extraction/plotting script. It selects three correct and three incorrect cases with distinct true classes from evenly spaced test positions, records original image IDs, and verifies exact additive reconstruction of all class logits before plotting. Outputs include concept-specific heatmaps, global attention, signed class-margin contributions, PNG/PDF figures, cached arrays, and a local HTML gallery.
- CPU extraction completed successfully; six real test cases selected after 16 forward-evaluated candidates. Reviewed the overview and correct/error detailed plots, calculated supporting/opposing concepts, and adjusted chart spacing before notebook integration. No training or GPU submission occurred.
- 2026-10-11: Added and executed Section 4.10 (eight cells / four code cells / seven embedded figures), with evidence-based interpretations of all six cases. Preserved all 57 original cells, outputs, and metadata. Added gallery/reproduction instructions to README. Final independent validation passed for official image IDs and classes, disjoint test membership, attention normalization, probabilities and signed margins, all-class logit reconstruction, selected maps, seven decoded PNGs, seven PDF exports, notebook schema, and git diff --check. Verification saved in explanation_artifacts/disjoint-45478721/verification.json.
- 2026-10-11: User approved regrouping the existing results into the assessment's Section 5. Restored current notes and inspected notebook boundaries. Results currently occupy 4.6, 4.8, 4.9, and 4.10; 4.7 contains future-experiment discussion. Preserve all executed code and its dependency order while reorganizing Markdown and updating section references.
- Regrouped results as 5.1–5.4 under Results & Visualization; moved the four existing tuning Markdown cells to 6.1–6.4 under Discussion. Updated all section references and README's example location. Preserved all existing cell IDs, computational code/order, metrics, metadata, and 16 figure outputs; one source/stdout caption received a section-number-only update. Notebook schema and exact preservation checks passed, all saved outputs remain error-free, and git diff --check passed. No training, inference, or notebook computation rerun.
