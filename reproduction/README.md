# Paper reproduction

Aggregate data for **Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker Configuration**, by Sooyoung Jang, Sungpil Woo and Ahyun Lee; manuscript under review.

From the repository root:

```bash
python scripts/reproduce.py --suite paper --output ../paper-replay-output
python -m pip install -r reproduction/requirements-supporting.txt
python scripts/reproduce.py --suite supporting --output ../supporting-output
python scripts/reproduce.py --suite all --output ../all-output
```

`paper` is the default and uses only Python 3.10+ standard-library modules. Supporting analyses use the existing NumPy, SciPy and Matplotlib pins on Python 3.11. Both suites work offline after dependencies are installed. Output directories must be new, inside or outside the repository, and must not overlap source, input or local research trees. Results are published only after every requested check succeeds. For `all`, output contains `paper/`, `supporting/` and `verification.json` with both outcomes.

The external output paths above keep a source archive without `.git` ready for `python scripts/audit_public_repository.py` after replay. That audit requires exactly the public files and rejects generated outputs inside an extracted archive. For a Git checkout, the default audit includes cached Git paths even if absent from the worktree. Run `python scripts/audit_public_repository.py --require-staged` after staging the intended changes and before committing or publishing. `--worktree-only` is a local migration check and does not certify the index.

| Paper output | Coverage |
| --- | --- |
| `focused-cohort-descriptive.csv` | 520 cohort summaries. |
| `focused-matched-paired-descriptive.csv` | 620 paired summaries. |
| `decoder-contrast.csv` | Four ten-run contrasts by joint decoder configuration. |
| `test-utility.csv` | All 90 nonstandard comparisons, including utility failures. |
| `disco-selection-replay.csv` | Four frozen decisions from all 12 candidates. |
| `training-history-summary.csv` | 36 ten-epoch histories: 20 evaluation models, 12 candidates and four standard references. |
| `jpeg-selection-replay.csv` | Four decisions from all 48 validation/test quality-grid records. |
| `verification.json` | `PASS_CPU_AGGREGATE_REPLAY`, 8,227 comparisons, tolerance `1e-12`, cohort and failure counts. |

The original five-run block, added five-run block, combined ten/nine-run cohorts and original five/four-run cohorts remain distinct. Their 1,140 descriptive rows overlap and do not constitute independent observations. Training losses were not recorded.

The compatibility command remains available in a complete public checkout:

```bash
python reproduction/replay_aggregates.py --data reproduction/aggregate-data --output ../compatibility-output
```

Its `--data` directory must contain the published aggregate bytes. Copying this wrapper alone is insufficient; its canonical implementation is in `src/reproduction/`.

[Supporting analyses](supporting/README.md) retain eight numerical comparisons and three figure signatures. [PAPER_MAP.md](../docs/PAPER_MAP.md) distinguishes recalculated values, supplied tables, and unavailable renderings for every current manuscript item. Secondary tables supplied for inspection are hash-verified but not recalculated by the paper suite.

[provenance.json](provenance.json) retains upstream source commitments; [migration-map.json](migration-map.json) records original hashes and canonical destinations. Source implementations are under `../src/`, recipes under `../config/`, recorded environments under `environment/`, and DISCO author source/license under `upstream/`. The duplicate scientific snapshot is preserved at the original commit rather than maintained here.

```bash
(cd reproduction && sha256sum --check SHA256SUMS)
```

This manifest covers every other public file in this reproduction tree, including the supporting manifest. Independent fixed scientific hashes are checked separately. Root [LICENSE-RESULTS](../LICENSE-RESULTS) gives the complete numeric license scope, including supporting files; the [local scope](LICENSE-RESULTS) lists the paper aggregate tables. Authored code/documentation use MIT and pinned DISCO source retains [Apache-2.0](upstream/LICENSE). See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
