# Aggregate data guide

All tables contain model-, decoder-, candidate- or epoch-level aggregates. They contain no per-image predictions or identifiers. Filenames are stable identifiers used by the replay and provenance records.

| File | Rows |
| --- | ---: |
| `focused-all-decoder-metrics.csv` | 240 |
| `focused-all-primary-individual.csv` | 120 |
| `focused-cohort-descriptive.csv` | 520 |
| `focused-jpeg-full-curves.csv` | 48 |
| `focused-jpeg-selected-points.csv` | 4 |
| `focused-matched-paired-descriptive.csv` | 620 |
| `focused-matched-paired-individual.csv` | 900 |
| `followup-decoder-contrast.csv` | 4 |
| `followup-disco-candidates.csv` | 12 |
| `followup-test-utility.csv` | 90 |
| `followup-training-history.csv` | 360 |
| `historical-secondary-directions.csv` | 96 |
| `historical-secondary-group.csv` | 128 |
| `historical-secondary-individual.csv` | 160 |
| `historical-secondary-paired-individual.csv` | 120 |
| `historical-secondary-paired.csv` | 96 |

## Interpretation

- `focused-all-primary-individual.csv` has 120 model cells; `focused-all-decoder-metrics.csv` has 240 decoder cells. An individual row represents a model/decoder run, not a person or an image.
- `focused-cohort-descriptive.csv` and `focused-matched-paired-descriptive.csv` contain 520 and 620 summaries. Cohorts overlap; their row counts are not numbers of independent experiments. Paired differences are method minus standard.
- `followup-decoder-contrast.csv` contains the four ten-run higher-resolution contrasts by dataset and decoder. `followup-test-utility.csv` retains all 90 nonstandard model comparisons, including failures.
- `followup-disco-candidates.csv` contains every candidate. Validation accuracies are percentages, and qualification allows at most one percentage point loss. Selection uses seed 424242. Frozen choices and validation-infeasible cases are retained.
- `followup-training-history.csv` contains 360 epoch records with explicit model roles, selected-epoch flags and receipt hashes. Candidate, evaluation and standard-reference models have different roles; training losses were not recorded.
- `focused-jpeg-full-curves.csv` contains six qualities for each dataset/interface and validation/test partition. Selection uses validation data. An empty selected quality denotes no qualifying tested point.
- `historical-secondary-*` tables retain the five-run scope of the corresponding secondary analyses. They are supplied for inspection and are not recomputed by the paper replay.

The seed blocks are `[7, 42, 123, 2024, 2025]` and `[1001, 1002, 1003, 1004, 1005]`. The ten-run cohort combines them; the nine-run sensitivity cohort excludes seed 42. The five/four-run sensitivity analyses remain separate. Attribute rows, stochastic draws and timing blocks are not additional model runs.

See [the replay guide](../README.md) for executable checks and output definitions. CSV values and scientific source are preserved from the recorded analysis; documentation does not change their numerical interpretation.
