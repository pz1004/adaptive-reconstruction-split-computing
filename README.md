# Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker Configuration

Official code and aggregate results for the manuscript by Sooyoung Jang, Sungpil Woo and Ahyun Lee, under review. This repository replays published aggregate calculations and provides the scientific implementation. It does not reproduce the raw-image experiments from a public checkout.

## Reproduce the paper calculations

From the repository root, with Python 3.10 or later:

```bash
python scripts/reproduce.py --output ../paper-replay-output
```

The default `paper` suite uses only the Python standard library. It runs offline and verifies **8,227 numerical comparisons**, **1,140 descriptive rows**, the fixed run cohorts, all 12 DISCO candidates, 360 epoch records, and the six-quality JPEG grid at absolute tolerance `1e-12`.

For the retained supporting analyses, use Python 3.11 and the existing dependency pins:

```bash
python -m pip install -r reproduction/requirements-supporting.txt
python scripts/reproduce.py --suite supporting --output ../supporting-output
python scripts/reproduce.py --suite all --output ../all-output
```

Supporting replay verifies eight CSVs and three retained figure signatures. Its four-condition primary plot contains standard, AFD, Laplace and the offset control. The current five-condition comparison also includes DISCO; the retained plot does not redraw that figure. The [table and figure map](docs/PAPER_MAP.md) identifies coverage for every current manuscript item.

Every output directory must be new. Paths such as `replay-output`, `outputs/run-1`, or an external directory are supported. Source, input and local research trees are protected. The command computes and verifies in a temporary sibling directory, then publishes the complete result with an atomic no-replace rename. `all` creates separate `paper/` and `supporting/` directories and a report containing both outcomes; either failure leaves the requested destination absent. Atomic publication is implemented for Linux, macOS and Windows; other platforms fail without publication. Only Linux has been verified here.

The existing command remains a thin compatibility wrapper:

```bash
python reproduction/replay_aggregates.py --data reproduction/aggregate-data --output ../compatibility-output
```

The old `release/v1.3.0/reproduce.py --output-dir ... --check` command is retired; use `scripts/reproduce.py --suite supporting --output ...`. Verification is always enabled.

## Source and data

| Location | Contents |
| --- | --- |
| `src/` | Canonical base modules, including `analysis.py`, `benchmark_v1.py` and `eng_compute.py`. |
| `src/focused/` | Reusable DISCO/JPEG components and encoder/decoder fitting functions. |
| `src/reproduction/` | Paper replay, supporting analysis/plotting, independent scientific hashes and output publication. |
| `scripts/` | Public commands, audits and existing base-study tools. |
| `config/` | Unchanged contracts and captured scientific recipes. |
| `reproduction/aggregate-data/` | Current paper aggregates and distinct secondary tables. |
| `reproduction/supporting/` | Supporting inputs, expected outputs, schemas and diagnostic aggregates. |
| `results/raw/` | Canonical primary, semantic and budget CSVs; identical copies were consolidated here. |
| `reproduction/upstream/` | Pinned DISCO source, configuration and Apache-2.0 license together. |

See the [reproduction guide](reproduction/README.md), [environment and execution limits](docs/REPRODUCIBILITY.md), and [migration/provenance record](docs/MIGRATION.md). Original source bytes remain available at commit `fd587a219fcba1e270c136514b501a2d28f9cffe`. Private-context orchestration retains its original execution gates at that reference; moving reusable functions does not grant admission or provide an end-to-end runner.

## Verify the public repository

For an extracted source archive without `.git`, keep the source copy clean and write outputs outside it. From the extracted root, this sequence replays the paper and then audits the unchanged public inventory:

```bash
python scripts/reproduce.py --output ../paper-replay-output
python scripts/audit_public_repository.py
```

Use a new destination each time; skip the replay line if already run above. An archive audit includes generated files inside the source tree and rejects them as extra public files. A Git checkout uses tracked and unignored paths; generated outputs remain excluded by `.gitignore`.

```bash
(cd reproduction && sha256sum --check SHA256SUMS)
python scripts/audit_public_repository.py
```

The default audit includes every cached Git path, including staged files deleted only from the worktree. Before committing or publishing, stage the intended public changes and run `python scripts/audit_public_repository.py --require-staged`; this also verifies exact index membership and that staged bytes match the worktree. During an unstaged migration, `--worktree-only` checks local files and explicitly reports that it has not verified the index. It is insufficient for a commit or publication check.

The independent scientific hashes prevent a changed checksum manifest from authorizing altered results. For the complete public gate, install the unchanged scientific versions in a separate Python 3.11 environment:

```bash
python -m pip install -r requirements.txt -r reproduction/requirements-supporting.txt
./run_public_checks.sh
```

This gate downloads and hash-checks the required AlexNet weight into a temporary cache, then runs export checks, the public inventory audit, both replay suites and tests. In Git it requires the staged audit; outside Git it requires a clean public source copy. Paper replay itself requires neither weights nor network access. The base CLI can be inspected with `python scripts/run_study.py --help`; full execution requires its declared data, checkpoints and protocol context. Never load an untrusted `.pt` file. The two reviewed checkpoint compatibility loaders retain their existing trusted-artifact boundary.

## Interpretation and reuse

The results describe the evaluated attacker configurations and selected operating points. Attacker architecture and objective vary together; added-run victim and attacker seeds are coupled. Conditional intervals omit selection uncertainty. DISCO convergence is not established, and JPEG infeasibility refers to the six tested qualities. Aggregate replay does not authenticate raw predictions, retrain models, repeat timing measurements, or establish practical privacy or deployment savings. Datasets, checkpoints, manuscript files, private records and generated outputs are outside the public inventory.

Use [CITATION.cff](CITATION.cff) for citation. Authored code and documentation use [MIT](LICENSE); contributor-owned numeric results use the exact [CC BY 4.0 scope](LICENSE-RESULTS). The licensed results are distributed as repository files. This repository does not claim third-party permission; dataset terms remain applicable. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
