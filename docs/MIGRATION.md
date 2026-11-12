# Source migration and original reference

The original public source is commit [`fd587a219fcba1e270c136514b501a2d28f9cffe`](https://github.com/pz1004/adaptive-reconstruction-split-computing/tree/fd587a219fcba1e270c136514b501a2d28f9cffe). Its Git history was preserved locally in a verified bundle before editing. A verified local workspace archive and starting hashes protect the manuscript and unrelated research files; those private artifacts are excluded from the public inventory.

[migration-map.json](../reproduction/migration-map.json) records every relocated, consolidated or archived public file, its original SHA-256, exact source link, destination and action. The original source archive commitment `88f4516febd32092c443d08068c97861b01802eff69399109ea15e016aa88865` remains in the provenance metadata; it describes the original archive, not the new layout.

| Original location | Canonical location / disposition |
| --- | --- |
| Twenty `reproduction/scientific/src/` duplicates | Existing `src/` modules, byte-identical. |
| `scientific/src/benchmark_v1.py`, `eng_compute.py` | `src/benchmark_v1.py`, `src/eng_compute.py`; ignored local copies were byte-verified against published snapshots before adoption. |
| `release/v1.3.0/statistics_core.py` | Existing `src/analysis.py`, byte-identical; supporting imports use it. |
| `scientific/implementation/components.py`, `training.py` | `src/focused/`; components unchanged, training changes only its package import. |
| Release replay and plotting modules | `src/reproduction/`; calculations retained, input lookup/output publication adapted. |
| Original `reproduction/replay_aggregates.py` | Calculation body in `src/reproduction/paper.py`; thin compatibility wrapper retained. |
| Release inputs, expected data, schemas, diagnostics, analyses and evidence | `reproduction/supporting/`; primary and semantic CSV duplicates reuse `results/raw/`. |
| Scientific configurations | `config/`; duplicate contracts reused byte-for-byte. |
| Duplicate protocol environment | `results/protocol/environment.json`. |
| Duplicate dependency lists | Root pins reused through requirement/constraint includes; no upgrades. |
| Pinned upstream DISCO source | Remains with configuration, attribution and Apache-2.0 license in `reproduction/upstream/`. |

Distinct tables with overlapping observations are preserved. Original `results/` schema identifiers and retained figure signature keys remain unchanged. Supporting output PDFs now have descriptive names; current-item correspondence is in [PAPER_MAP.md](PAPER_MAP.md).

The supporting compute module imports `student_t_interval` and `ComputeContractError` from the byte-preserved `src/eng_compute.py`. `scripts/plot_results.py` imports `_figure_bytes` from `src/reproduction/primary_figure.py`; its original private-context gates remain in the command. Both shared functions retain their original algorithms.

## Archived execution context

These originals are preserved at the reference commit and intentionally require their original context:

- [admission.py](https://github.com/pz1004/adaptive-reconstruction-split-computing/blob/fd587a219fcba1e270c136514b501a2d28f9cffe/reproduction/scientific/implementation/admission.py): private admission evidence and checks.
- [supervise.py](https://github.com/pz1004/adaptive-reconstruction-split-computing/blob/fd587a219fcba1e270c136514b501a2d28f9cffe/reproduction/scientific/implementation/supervise.py): resource supervision and accounting.
- [run_matrix.py](https://github.com/pz1004/adaptive-reconstruction-split-computing/blob/fd587a219fcba1e270c136514b501a2d28f9cffe/reproduction/scientific/implementation/run_matrix.py): admitted matrix execution, protocol/seal/jobs requirements.
- [science.py](https://github.com/pz1004/adaptive-reconstruction-split-computing/blob/fd587a219fcba1e270c136514b501a2d28f9cffe/reproduction/scientific/implementation/science.py): scientific job definitions and historical checkpoint context.
- [analyze_focused_results.py](https://github.com/pz1004/adaptive-reconstruction-split-computing/blob/fd587a219fcba1e270c136514b501a2d28f9cffe/reproduction/scientific/implementation/analyze_focused_results.py): receipt-bound analysis and publication outputs.

For a repository with the reference object available, `git show fd587a219fcba1e270c136514b501a2d28f9cffe:PATH` recovers exact bytes. A shallow clone may require obtaining that commit first. Recovering source does not reconstruct admission, ledger, checkpoint or receipt context and does not authorize bypassing execution gates.

## Integrity boundary

The duplicate trees were removed only after every original tracked file had a byte-verified destination or was verified recoverable at the reference commit. `src/reproduction/integrity.py` retains independent starting hashes for scientific data, configurations, environments and third-party source. Public policy manifests reflect current destinations; changing a manifest alone cannot authorize changed results. No manuscript change, new experiment, history rewrite, remote push, tag or release publication is part of this migration.
