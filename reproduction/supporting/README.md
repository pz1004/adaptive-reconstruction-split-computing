# Retained supporting analyses

These analyses cover the four-condition comparison (standard, AFD, Laplace and deterministic offset), auxiliary-budget sensitivity, fixed-victim attacker restarts, component costs and post hoc Laplace endpoints. They retain the original data and statistical families. They do not reconstruct the current DISCO-inclusive primary figure.

From the repository root:

```bash
python -m pip install -r reproduction/requirements-supporting.txt
python scripts/reproduce.py --suite supporting --output ../supporting-output
```

The destination must be new. All eight CSV comparisons and three figure-signature checks are mandatory, with absolute numerical tolerance `1e-12`. Outputs appear only after verification. The old release command is retired.

The external path keeps an extracted source archive clean for the exact public-inventory audit. Generated outputs inside an archive without `.git` would cause that audit to fail; see the [reproduction guide](../README.md).

| Output | Meaning |
| --- | --- |
| `supplementary-statistics.csv` | 24 primary and 12 semantic contrasts, with separate original Holm families. |
| `laplace-single-release-statistics.csv` | Eight post hoc contrasts with their own Holm adjustment. |
| `budget_method_summaries.csv`, `benchmark_budget_paired_effects.csv` | Five coupled runs per condition; model and attacker seeds vary together. |
| `attacker_restart_summaries.csv` | Three attacker seeds conditional on victim seed 42. |
| `compute_group_summary.csv`, `compute_paired_effects.csv`, `compute_interface_effects.csv` | Retained component-cost summaries, excluding transfer/network/queueing and other unmeasured costs. |
| `four-condition-accuracy-vs-ssim.pdf` | Retained four-condition plot; does not draw the current five-condition comparison. |
| `attacker-budget-sensitivity.pdf` | Budget plot corresponding to current main Figure 4. |
| `component-compute-tradeoffs.pdf` | Component-cost plot corresponding to Supplementary Figure S1. |
| `figure-verification.json`, `reproduction-report.json` | Signatures, CSV checks and aggregate-only scope. |

The signature keys `2`, `3`, `5` inside the byte-preserved reference JSON identify the original figures. Output filenames and the [current item map](../../docs/PAPER_MAP.md) provide their current meaning. Signature checks cover coordinates, uncertainty, scales, series and styles; PDF byte identity is not claimed.

There are seven logical inputs with 556 rows. Five files (436 rows) live in `inputs/`; `primary_seed_results.csv` (80) and `semantic_seed_results.csv` (40) reuse `results/raw/` in the repository root. The implementation fixes these canonical paths and checks their bytes against independent scientific hashes before parsing those same bytes. [provenance.json](provenance.json) must name exactly these inputs and agree with the fixed paths and digests; editing provenance or checksums cannot redirect calculations. The separate 240-row `evidence/benchmark_budget_attacker_rows.csv` is retained evidence, not an eighth input. Distinct tables are not collapsed just because observations overlap.

`expected/` contains unchanged reference data. `analysis/`, `schemas/`, `diagnostics/`, [DATA_DICTIONARY.md](DATA_DICTIONARY.md) and `analysis_contract.json` retain the original scientific definitions. Historical schema `file` identifiers remain unchanged; public validators map `results/attacker_restart_rows.csv` and `results/benchmark_budget_conditions.csv` to `inputs/`, the attacker-budget evidence identifier to `evidence/`, and summary identifiers to `expected/`.

The known-offset diagnostic preserves 40 existing CPU cases over 32 examples each; this command verifies its hashes but does not rerun it. Aggregate replay does not regenerate predictions, authenticate experiment execution, retrain models or remeasure costs. See the root license and third-party notices.
