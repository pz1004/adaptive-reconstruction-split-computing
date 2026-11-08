# Reproducibility guide

The repository supports two kinds of work: replaying saved aggregate analyses and inspecting the experimental implementation. The first runs from the public files alone. Raw-image experiments additionally require the data, weights and execution context described below.

## Aggregate analyses

The [paper replay](../reproduction/README.md) runs with Python 3.10+ and no third-party packages. It checks decoder-specific and maximum-over-decoder summaries, paired contrasts, utility failures, validation-frozen DISCO choices, complete selection/evaluation histories and the six-quality JPEG grid. Its output directory must be new; missing or inconsistent required inputs fail before output creation.

The [supporting analysis package](../release/v1.3.0/README.md) uses Python 3.11 and its own dependency list. It covers reconstruction and semantic summaries, attacker-budget comparisons, component measurements and single-release Laplace diagnostics. Its figure numbers belong to that versioned package; use the table descriptions to identify the analysis rather than assuming manuscript numbering.

Neither replay performs image inference, fits a model, repeats hardware measurements, or authenticates unavailable raw evidence. Copies and derived summaries do not count as additional runs.

## Scientific implementation

| Location | Role |
| --- | --- |
| `src/`, `config/`, `scripts/` | Base experimental framework and protocol tools. |
| `reproduction/scientific/implementation/components.py` | DISCO adaptation, JPEG encoding and operating-point selection. |
| `reproduction/scientific/implementation/training.py` | Encoder and reconstruction-decoder fitting. |
| `reproduction/scientific/implementation/science.py` | Dataset interfaces and scientific job definitions. |
| `reproduction/scientific/implementation/analyze_focused_results.py` | Saved-result analysis for the paper comparisons. |
| `reproduction/scientific/config/` | Captured scientific recipes and protocol specifications. |
| `reproduction/scientific/src/` | Exact base-module snapshot used with the captured implementation. |
| `reproduction/upstream/` | Pinned DISCO author source, configuration and Apache-2.0 license. |

The snapshots preserve source bytes and identifiers recorded in [provenance.json](../reproduction/provenance.json). Their orchestration entry points require the recorded directory structure, admission records and checkpoints. A public clone does not contain those private artifacts, so running the orchestration scripts directly is not a supported end-to-end reproduction path. The self-contained command is the aggregate replay.

The base CLI can be inspected without starting an experiment:

```bash
python scripts/run_study.py --help
```

This entry point uses the base framework without changing its scientific contract or command semantics.

## Datasets and weights

- Obtain CIFAR-10 through its official distribution or `torchvision`; the declared selection protocol uses a class-stratified 10% validation split with seed 314159.
- Obtain aligned CelebA images and annotations under the official terms. The data interface expects the annotation and partition files beneath `<data-root>/celeba/`.
- Obtain the required LPIPS/AlexNet weights separately. Checkpoint-dependent comparisons also require the applicable victim and decoder checkpoints, which are not distributed.
- Synthetic samples are test fixtures enabled by explicit flags. They are not substitutes for study observations.

See [THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md) for the data and licensing boundaries.

## Environments

The base framework and full public tests use Python 3.11. Install their dependencies in an isolated environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r release/v1.3.0/requirements.txt
./run_public_checks.sh
```

Select PyTorch wheels for the target CPU/CUDA platform. The requirements are version pins, not a cross-platform hash lock. Record substitutions explicitly.

Captured environment records are under [`reproduction/environment/`](../reproduction/environment/). The paper jobs recorded Torch `2.9.0+cu128`, scikit-image `0.25.0` and LPIPS `0.1.4`, but did not record complete Python, TorchVision, NumPy, Pillow or JPEG-library versions per job. The other environment snapshots do not establish those missing versions. Encoded JPEG streams were not retained, so exact codec-byte replay is not established.

## Integrity and provenance

`reproduction/SHA256SUMS` covers the paper reproduction payload. `release/v1.3.0/SHA256SUMS` covers the versioned supporting package. The independent repository audit also fixes expected scientific-file hashes and the exact public inventory; updating a checksum list alone cannot authorize changed scientific data. Dataset files, checkpoints, private research records and submission correspondence are excluded from Git.
