# v1.3.0 aggregate reproduction

This directory supplies the aggregate data and reproduction code for the manuscript. The journal supplementary attachment is one PDF containing Tables S1–S3. Repository files provide the full-precision tables and diagnostic aggregates.

From the repository root, create a fresh Python 3.11 environment and run:

```bash
python3.11 -m venv /tmp/eng-reproduction-venv
/tmp/eng-reproduction-venv/bin/python -m pip install -r release/v1.3.0/requirements.txt
/tmp/eng-reproduction-venv/bin/python release/v1.3.0/reproduce.py --output-dir /tmp/eng-reproduction --check
(cd release/v1.3.0 && sha256sum --check SHA256SUMS)
```

After dependency installation, reproduction works offline on CPU without datasets, checkpoints, private workspaces or a GPU. Outputs must be outside this directory so its exact checksum inventory remains intact. Generated PDFs are not stored in this text-only repository.

## File index

| Files | Contents |
| --- | --- |
| `reproduce.py` | Reproduction entry point and numeric/figure comparisons |
| `statistics_core.py`, `compute_core.py`, `primary_figure.py`, `budget_figure.py` | Statistical and plotting implementations |
| `requirements.txt` | Exact direct dependency versions |
| `inputs/*.csv` | Seven aggregate input tables, 556 rows in total |
| `expected/supplementary-statistics.csv` | Full-precision Tables S1/S2, 24 primary and 12 semantic contrasts |
| `expected/laplace-single-release-statistics.csv` | Full-precision Table S3, eight separately adjusted post hoc contrasts |
| Other `expected/*.csv` | Budget, fixed-victim retraining and compute reference summaries |
| `expected/figure_sources.json`, `expected/figure_signatures.json` | Reference data and plot signatures for Figures 2, 3 and 5 |
| `diagnostics/known-offset-diagnostic.csv`, `.json` | Existing CPU diagnostic aggregates: 40 reconstructor cases, 20 learned offsets; no per-example records |
| `DATA_DICTIONARY.md`, `analysis_contract.json`, `provenance.json` | Field meanings, analysis scope and original input commitments |
| `analysis/*.json` | Two preserved component-compute analyses |
| `evidence/benchmark_budget_attacker_rows.csv` | 240 previously disclosed attacker rows; separate from the seven reproduction inputs |
| `schemas/*.json` | Seven preserved historical schema files |
| `SHA256SUMS` | 39 checksum entries covering the other files in this 40-file directory |

Coverage: the original 24 primary and 12 semantic contrasts; eight separate post hoc Laplace contrasts; primary direction counts and utility violations; budget group and paired summaries; fixed-victim attacker-retraining summaries; compute group, paired and interface summaries; Figures 2, 3 and 5. All numerical comparisons use absolute tolerance 1e-12 with exact key membership. Plot verification compares all coordinates, uncertainty segments, scales, ticks, lines, marker styles and series with retained signatures from the current manuscript figures. Rendering metadata may differ. Figures 1 and 4, the known-offset diagnostic, training and prediction generation are outside this aggregate reproduction command.

The expected directory holds retained full-precision reference outputs. statistics_core.py is an unchanged copy of the existing statistical implementation. compute_core.py, primary_figure.py and budget_figure.py retain the original numeric/plotting functions with private input loading removed. All original scientific evidence and original 24/12 Holm families are preserved. The new eight contrasts have a separate Holm adjustment.

Inputs contain 80 primary runs, 40 semantic runs, 120 budget conditions, 96 attacker-retraining rows, 160 accepted compute conditions, 50 draw aggregates, and ten Laplace run aggregates (556 input rows; some repeat previously public data). See DATA_DICTIONARY.md, provenance.json and analysis_contract.json. These are aggregate tables; no probabilities, targets, images or example identifiers are supplied.

Eight analysis CSVs, three PDFs, figure-verification.json and reproduction-report.json are generated. Expected primary counts are 18 negative AFD-minus-standard SSIM differences among 20 pairs, and three AFD test utility violations. The inferential unit is the model run, never the draw, image or attribute. The fixed-victim retraining rows use one victim seed and do not support population inference.

The diagnostic files are preserved results from the existing CPU implementation check. The reproduction command verifies their file checksums but does not rerun that diagnostic. They cover the first 32 official test examples per dataset, without releasing sample-level values. The original reproduction archive remains unchanged in the authors' local records; scripts, inputs, expected outputs and dependency pins here retain its exact bytes.

This is reproduction from aggregate metrics. It does not regenerate predictions, authenticate experiment execution, retrain models, or repeat timing measurements. CIFAR-10 and CelebA must be obtained from their original providers under their respective terms. Datasets, images, per-example records, private execution records and model checkpoints are not redistributed.

Code and authored documentation use the root MIT license. Contributor-owned numeric results use the exact CC BY 4.0 scope in the root `LICENSE-RESULTS`. No third-party permission or waiver is claimed; original dataset terms remain applicable. This version is distributed as repository files; no GitHub Release, tag or DOI is created.

## Schema-location mapping

Historical schema identifiers remain byte-identical. Validators resolve each historical `results/` identifier to the physical path below, relative to this directory; schema `file` fields are identifiers, not current filesystem paths.

| Historical identifier | Current physical path | Schema |
| --- | --- | --- |
| `results/attacker_restart_rows.csv` | `inputs/attacker_restart_rows.csv` | `schemas/attacker_restart_rows.schema.json` |
| `results/attacker_restart_summaries.csv` | `expected/attacker_restart_summaries.csv` | `schemas/attacker_restart_summaries.schema.json` |
| `results/benchmark_budget_attacker_rows.csv` | `evidence/benchmark_budget_attacker_rows.csv` | `schemas/benchmark_budget_attacker_rows.schema.json` |
| `results/benchmark_budget_conditions.csv` | `inputs/benchmark_budget_conditions.csv` | `schemas/benchmark_budget_conditions.schema.json` |
| `results/benchmark_budget_paired_effects.csv` | `expected/benchmark_budget_paired_effects.csv` | `schemas/benchmark_budget_paired_effects.schema.json` |
| `results/compute_group_summary.csv` | `expected/compute_group_summary.csv` | `schemas/eng_compute_v1.schema.json (package boundary)` |
| `results/compute_interface_effects.csv` | `expected/compute_interface_effects.csv` | `schemas/eng_compute_v1.schema.json (package boundary)` |
| `results/compute_paired_effects.csv` | `expected/compute_paired_effects.csv` | `schemas/eng_compute_v1.schema.json (package boundary)` |

`schemas/benchmark_v1_tables.schema.json` retains the five benchmark table names and their original definitions. The compute schema defines the measurement boundary; exact compute CSV columns and counts are checked by the public validators. The two analyses retain their paths under `analysis/`.

The additional 240-row evidence table is preserved for inspection and schema verification; it is not an eighth reproduction input. The seven input tables still contain 556 rows. The canonical disclosure remains 13 tables and 2,308 rows, excluding repeated reproduction copies and derived outputs.
