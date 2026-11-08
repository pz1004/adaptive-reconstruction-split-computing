# Paper reproduction

Aggregate data and experimental source for **Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker Configuration**, by Sooyoung Jang, Sungpil Woo and Ahyun Lee. The manuscript is under review.

## Run the aggregate replay

From the repository root:

```bash
python reproduction/replay_aggregates.py \
  --data reproduction/aggregate-data \
  --output replay-output
```

Python 3.10+ is sufficient. The script uses only the standard library and included CSVs. It runs offline without datasets, checkpoints, GPU libraries or access to private workspaces. The output directory must not exist.

## Outputs and checks

| Output | Coverage |
| --- | --- |
| `focused-cohort-descriptive.csv` | 520 cohort summaries. |
| `focused-matched-paired-descriptive.csv` | 620 paired summaries. |
| `decoder-contrast.csv` | Four ten-run contrasts between the evaluated decoder configurations. |
| `test-utility.csv` | All 90 nonstandard model comparisons, including utility failures. |
| `disco-selection-replay.csv` | Four operating-point decisions from all 12 candidates. |
| `training-history-summary.csv` | 36 model histories, each with ten recorded epochs. |
| `jpeg-selection-replay.csv` | Four decisions from all 48 validation/test grid records. |
| `verification.json` | Pass status, cohort sizes, numerical tolerance, comparison counts and failure counts. |

Saved numerical summaries are checked at absolute tolerance `1e-12`. A consistent run performs 8,227 comparisons and reports `PASS_CPU_AGGREGATE_REPLAY`. Missing required files or inconsistent inputs produce a nonzero exit before output creation. An existing output directory is never overwritten.

The five-run seed block, additional five-run block, ten/nine-run higher-resolution cohorts and five/four-run sensitivity cohorts remain separate. Attacker and victim seeds are coupled in the additional block. Candidate models, evaluation models and standard selection references have distinct roles. Training losses were not recorded.

This command replays aggregate calculations. It does not rerun semantic probes, sign-flip tests, hardware measurements or image inference. The supporting secondary tables are provided for inspection; their analysis context is described in the [repository reproducibility guide](../docs/REPRODUCIBILITY.md).

## Files

- [`aggregate-data/`](aggregate-data/README.md): model-, decoder-, candidate- and epoch-level tables with a data guide.
- [`scientific/`](scientific/): exact implementation and base-module snapshots, plus sanitized scientific recipes.
- [`environment/`](environment/): recorded versions and explicit gaps in the execution record.
- [`upstream/`](upstream/): pinned DISCO author source, configuration and license.
- [`provenance.json`](provenance.json): scientific input hashes, source identifiers, upstream attribution and documentation transformations.
- [`SHA256SUMS`](SHA256SUMS): checksums for every other file in this directory tree.

Scientific orchestration scripts retain their recorded path, receipt and checkpoint requirements. They are supplied for inspection; a public clone cannot execute the full raw-image study without reconstructing that context. The portable command above needs none of those private inputs.

## Verify and cite

```bash
(cd reproduction && sha256sum --check SHA256SUMS)
```

Use the root [CITATION.cff](../CITATION.cff) for the paper and software. Authored code and documentation use [MIT](LICENSE), contributor-owned numeric tables use the exact [CC BY 4.0 scope](LICENSE-RESULTS), and the pinned DISCO source retains [Apache-2.0](upstream/LICENSE). See the [third-party notice](THIRD_PARTY_DATA.md) before reusing data or upstream code.
