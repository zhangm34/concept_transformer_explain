# Findings

## Paper explanation review (2026-10-11)

- First correct example: Black footed Albatross probability 0.92726195; runner-up Sooty Albatross 0.01857961; margin 3.9101715. Global attention brown 0.31616 (positive target), buff 0.19320, medium size 0.16399, black 0.12147, duck-like 0.10012 (remaining four not positive targets).
- Signed margin contributions: brown +2.20624, buff +1.21606, buff legs +1.07448, medium size −0.73278, black primary color −0.30498, black upperparts +0.30086, duck-like shape −0.16571, buff wings +0.14779; other concepts +0.16074; bias +0.00747. Eye map is near the eye, throat map overlaps its positive target, buff-leg map is diffuse on body/background with no positive target patches.
- Paper Section 4.3/Figure 6 reports known vocabulary concepts (wheel, door, shiny, snout) inferred where annotations omit them. Its concept-discovery/zero-shot sentence is a proposed extension. Zero-shot would require mapping known concepts to unseen class descriptions; the reproduced fixed 200-class head does not implement that.
- Mathematical faithfulness concerns how concept relevance participates in the classifier. It does not guarantee correct concept labels/localization or identify the pixel-level causal basis. The final explanation should distinguish these properties and identify the sample's positive contributions with questionable semantics.

- The relevant notebook section is 5.4; its first saved correct example is official image ID 3.
- The CUB CVIT uses the ViT class token for global concepts and 196 patch embeddings for spatial concepts. Cross-attention normalizes across concepts, not across image locations. Spatial scores therefore need not sum to one across a displayed map.
- Figure heatmaps use a fixed 0–1 scale, nearest-neighbor 14×14 patches, and cyan circles for positive annotation target patches. Global bars average attention across heads; asterisks mark positive CUB targets. Signed bars include every head's learned value/projection weights and reconstruct predicted-minus-comparison logits, with other concepts and biases included.
- The saved model has no unsupervised concepts; concept discovery and zero-shot learning should be distinguished from this supervised 200-class reproduction.

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

## Test explanation visualizations

- Current notebook has 57 cells and records a newer completed disjoint run 45478721: 5,525 train / 922 validation / 5,341 test images, selected epoch 28, test accuracy 0.8629469871520996. Its checkpoint is under the Blue working copy; Orange source files do not yet include the disjoint datamodule changes.
- Model forward returns class log probabilities, unsupervised concept attention, 13 global concept weights, and 196 × 95 spatial concept weights. Attribute labels must follow the dataset's actual concept-index ordering.
- Upstream plotting helper hardcodes an unrelated dataset root. Upstream predict_step computes idx as batch_idx × current batch length, which gives incorrect identities for the final partial batch; use explicit dataset indices in new extraction.
- Read the local paper and checked its primary online copy (https://openreview.net/pdf?id=kAa9eDS0RdO): Eq. 2 gives additive concept-based logits; Figs. 4–5 show correct examples and diagnosis of a misclassified bird. Implemented signed contributions from each head's concept values and classifier projection, rather than multiplying head-averaged attention by head-averaged coefficients.
- Blue split manifest hash is 8e7894dc5f436196eb7edfd4d9e3bfd48020ab60fdf1404a3d0e4c510d5011aa. Checkpoint is 3,721,749,527 bytes. Current allocation is CPU-only. New extraction uses saved test IDs, deterministic evaluation transforms, and CPU float32; no training is needed.
- Extraction succeeded after scanning 16 of 64 predetermined candidate positions. Correct IDs: 3 (Black footed Albatross, 92.7%), 190 (Groove billed Ani, 86.4%), 622 (Yellow headed Blackbird, 94.2%). Wrong IDs: 434 (Brewer Blackbird → Fish Crow, 38.2%), 1593 (American Crow → Common Raven, 30.0%), 2904 (Horned Grebe → Eared Grebe, 95.9%). Exact all-class logit reconstruction maximum error: 9.5367431640625e-06; actual attention heads: 2 per stream.
- Visual inspection: correct predictions can still have attention on background or incorrect concept assignments. Horned Grebe's largest wrong-class support is grey upperparts (+4.447 logit margin) and very small size (+1.941); its strongest global concepts are not marked in its CUB targets. American Crow/Common Raven is a near tie (30.026% vs 29.663%). Avoid interpreting these six examples as an aggregate explanation-quality measurement.
- Initial detailed plots let contribution labels overlap the global chart; adjusted panel widths/spacing and wrapped long class-comparison titles. OpenReview direct PDF fetch met a browser challenge; the complete local PDF was already available and read successfully.
- Final deliverables: main notebook Section 4.10 with seven embedded figures and four executed cells; standalone gallery, six detailed PNG/PDF pairs, overview PNG/PDF, selected_examples.npz, examples.json, and verification.json. Checkpoint SHA256: 0a6c72638680cdc01b3d9a4470b24e5bb80632511c8bb8bb39f64e22032727fc. All final numerical, source identity, export, schema, and preservation checks passed.

## Assessment section regrouping

- Results can be regrouped without changing code execution order: existing 4.6 → 5.1, 4.8 → 5.2, 4.9 → 5.3, 4.10 → 5.4. Move the four Markdown cells in 4.7 after all results into Section 6 Discussion. One debug code cell's printed section reference and its saved caption need a label-only update from 4.6 to 5.1.
- An additional instruction-file search accidentally included broad parent-directory trees; stopped only that exact search process. Earlier project/ancestor instruction checks had already found no applicable AGENTS.md. No project files were changed by the search.
- Current notebook organization: 5.1 first repository-protocol run; 5.2 disjoint baseline; 5.3 comparison; 5.4 concept examples and interpretations. Existing proposals now reside in Discussion 6.1–6.4. The notebook contains 66 cells (one new Section 5 introduction), with all 28 code cells in their original order and all 16 figures intact. New assessment_section_regrouping metadata records the mapping and preservation verification.
