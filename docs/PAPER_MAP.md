# Current manuscript table and figure map

This map follows the manuscript and supplement source labels inspected on 2026-11-11: five main tables, five main figures, eighteen supplementary tables and four supplementary figures. Manuscript sources and PDFs are not public inputs and were not edited. Numbers refer to those current items; stable labels are included for identification.

Commands, run from the repository root with fresh output directories:

- **P**: `python scripts/reproduce.py --suite paper --output replay-output`
- **S**: `python scripts/reproduce.py --suite supporting --output supporting-output`
- **H**: `python scripts/audit_public_repository.py` (integrity/schema check, no recalculation or figure rendering)

Input abbreviations: **A** = `reproduction/aggregate-data/`; **B** = `reproduction/supporting/`; **R** = `results/raw/`. Output names below are relative to `replay-output/` for P and `supporting-output/` for S. H reports to standard output and leaves supplied files unchanged. A named CSV is a full-precision source, not a promise of identical manuscript typography or rounding.

## Main manuscript

| Item / stable label | Input files | Command | Output / coverage limitation |
| --- | --- | --- | --- |
| Table 1 — `tab:decoder-contrast` | A `focused-all-decoder-metrics.csv`, `followup-decoder-contrast.csv` | P | `decoder-contrast.csv`: ten-run means, SDs and paired AFD contrasts; joint architecture/objective comparison. |
| Table 2 — `tab:focused-primary` | A `focused-all-primary-individual.csv`, `focused-all-decoder-metrics.csv`, `focused-cohort-descriptive.csv` | P | `focused-cohort-descriptive.csv`, original-five rows, including DISCO. No model fitting. |
| Table 3 — `tab:focused-seed-extension-main` | A `focused-all-primary-individual.csv`, `focused-cohort-descriptive.csv`, `focused-matched-paired-descriptive.csv` | P | The two descriptive CSVs, ten/nine-run higher-resolution cohorts. Coupled victim/attacker variation and frozen selection. |
| Table 4 — `tab:semantic` | R `semantic_seed_results.csv`; B `expected/supplementary-statistics.csv` | S | `supplementary-statistics.csv` checks paired semantic differences/intervals. Absolute means are supplied in R and are not separately emitted by this command. Probes are not rerun. |
| Table 5 — `tab:focused-jpeg-selected` | A `focused-jpeg-full-curves.csv`, `focused-jpeg-selected-points.csv` | P | `jpeg-selection-replay.csv`; validation-only selection among six qualities, one victim seed. No codec execution. |
| Figure 1 — `fig:threat` | `config/study_contract.json`, `config/implementation_protocol_v2.json`, `config/focused-recipe.json`, `src/models.py`, `src/focused/components.py` | H | Protocol/source integrity only. Current diagram artwork is not regenerated. |
| Figure 2 — `fig:primary` | A `focused-all-primary-individual.csv`, `focused-all-decoder-metrics.csv`, `focused-cohort-descriptive.csv` | P | Group statistics replayed in `focused-cohort-descriptive.csv`; current five-condition plot is not rendered. S's `four-condition-accuracy-vs-ssim.pdf` is a different retained plot without DISCO. |
| Figure 3 — `fig:examples` | Separately acquired official images, seed-7 victim and residual/perceptual decoder checkpoints | Unavailable from public inputs | Requires original inference context; no raw/reconstructed images or checkpoints are distributed. |
| Figure 4 — `fig:budget` | B `inputs/benchmark_budget_conditions.csv`, `expected/budget_method_summaries.csv`, `expected/figure_signatures.json` | S | `attacker-budget-sensitivity.pdf`, budget CSVs and signature verification. Five coupled runs; no independent attacker-seed effect. |
| Figure 5 — `fig:focused-jpeg` | A `focused-jpeg-full-curves.csv`, `focused-jpeg-selected-points.csv` | P | Grid and selected points verified in `jpeg-selection-replay.csv`; current plot not rendered, no model-seed interval. |

## Supplementary tables

| Item / stable label | Input files | Command | Output / coverage limitation |
| --- | --- | --- | --- |
| S1 — `tab:protocols` | `config/study_contract.json`, `config/implementation_protocol_v2.json`, `config/benchmark_extension_v1.json`, `config/focused-recipe.json` | H | Fixed protocol integrity; no experiment execution or table rendering. |
| S2 — `tab:implementation-details` | `src/models.py`, `src/dataset.py`, `src/metrics.py`, `src/focused/components.py`, protocol configurations | H | Public source inventory plus protocol hashes; implementation inspection, not fresh numerical evidence. |
| S3 — `tab:attacker-restarts` | B `inputs/attacker_restart_rows.csv`, `expected/attacker_restart_summaries.csv` | S | `attacker_restart_summaries.csv`; median/minimum/maximum of three attacker seeds conditional on victim seed 42. |
| S4 — `tab:benchmark-budget` | B `inputs/benchmark_budget_conditions.csv`, `expected/benchmark_budget_paired_effects.csv` | S | `benchmark_budget_paired_effects.csv`; five coupled pipeline seeds and conditional intervals. |
| S5 — `tab:compute-costs` | B `inputs/compute_condition_metrics.csv`, `expected/compute_group_summary.csv`, `expected/compute_paired_effects.csv`, `expected/compute_interface_effects.csv` | S | Three compute summary CSVs; component timings only. Manuscript unit conversions/formatting are not separately rendered. |
| S6 — `tab:seeds-celeba-early` | R `primary_seed_results.csv` | S | Original input rows checked while replaying `supplementary-statistics.csv`; individual four-condition values remain in the source CSV. |
| S7 — `tab:seeds-celeba-current` | R `primary_seed_results.csv` | S | Same retained individual-run source, CelebA lower-resolution subset. |
| S8 — `tab:seeds-cifar10-early` | R `primary_seed_results.csv` | S | Same retained individual-run source, CIFAR-10 higher-resolution subset. |
| S9 — `tab:seeds-cifar10-current` | R `primary_seed_results.csv` | S | Same retained individual-run source, CIFAR-10 lower-resolution subset. |
| S10 — `tab:all-exclude42` | A `focused-all-primary-individual.csv`, `focused-matched-paired-descriptive.csv` | P | `focused-matched-paired-descriptive.csv`, original-four-excluding42 rows; no reselection. |
| S11 — `tab:historical-metrics-deconv-mse` | A `historical-secondary-individual.csv`, `historical-secondary-group.csv` | H | Supplied five-run secondary summaries hash-verified; not recalculated by P or S. |
| S12 — `tab:historical-metrics-residual-lpips` | A `historical-secondary-individual.csv`, `historical-secondary-group.csv` | H | Supplied residual/perceptual subset hash-verified; not recalculated. |
| S13 — `tab:historical-secondary-paired` | A `historical-secondary-paired-individual.csv`, `historical-secondary-paired.csv`, `historical-secondary-directions.csv` | H | Supplied paired summaries and directions hash-verified; not recalculated. |
| S14 — `tab:semantic-contrasts` | R `semantic_seed_results.csv`; B `inputs/laplace_draw_metrics.csv`, `inputs/laplace_run_metrics.csv` | S | `supplementary-statistics.csv` semantic rows and `laplace-single-release-statistics.csv`. Retained tests remain in outputs; current table uses descriptive quantities. Draws/attributes are not independent runs. |
| S15 — `tab:focused-disco-selection` | A `followup-disco-candidates.csv` | P | `disco-selection-replay.csv` verifies four frozen decisions using all 12 supplied candidates. Candidate rows remain in the input CSV. No refitting or retuning. |
| S16 — `tab:focused-seed-extension` | A `focused-all-primary-individual.csv`, `focused-matched-paired-descriptive.csv` | P | `focused-matched-paired-descriptive.csv`, original/added/combined and exclusion cohorts. |
| S17 — `tab:original-primary-afd` | A `focused-all-primary-individual.csv`, `focused-all-decoder-metrics.csv`, `focused-cohort-descriptive.csv`, `focused-matched-paired-descriptive.csv`, `followup-decoder-contrast.csv` | P | Original-five AFD summaries and ten-run decoder contrasts checked. The post hoc within-run change between the two attacker contrasts, including its SD and nine-run sensitivity rows, is not separately recalculated by this command; underlying decoder rows are supplied. |
| S18 — `tab:disco-provenance` | `reproduction/upstream/disco.py`, `disco.json`, `LICENSE`; `reproduction/provenance.json`; `config/focused-recipe.json`; `src/focused/components.py` | H | Pinned source/recipe hashes and attribution; no claim to reproduce all original DISCO choices or benchmark scores. |

## Supplementary figures

| Item / stable label | Input files | Command | Output / coverage limitation |
| --- | --- | --- | --- |
| S1 — `fig:compute-costs` | B `inputs/compute_condition_metrics.csv`, `expected/figure_signatures.json` | S | `component-compute-tradeoffs.pdf`; amortized per-item component costs and retained signature, not end-to-end latency. |
| S2 — `fig:payload` | Six-channel 16×16 / 8×8 feature shapes and 3×32×32 image shape from `config/implementation_protocol_v2.json` / `src/models.py`; manuscript precision/bandwidth settings | H | Source integrity only; analytical `8S/B` plot not generated by P/S. No network measurement. |
| S3 — `fig:focused-seed-extension` | A `focused-matched-paired-individual.csv`, `focused-all-primary-individual.csv`, `focused-matched-paired-descriptive.csv` | P | Group paired statistics recalculated; supplied individual-difference file hash-verified. Plot not rendered. |
| S4 — `fig:disco-validation` | A `followup-training-history.csv` | P | `training-history-summary.csv` validates all 360 epochs and 36 roles/histories; 20 evaluation histories underlie the figure. Plot not rendered; loss curves unavailable. |

Eight supporting CSVs and three figure signatures are the complete supporting acceptance set. The preserved known-offset diagnostic is hash-checked only. The 8,227 paper comparisons do not imply complete reproduction of every manuscript calculation or visual. All intervals and comparisons retain their original conditional interpretation.
