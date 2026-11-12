# Reproducibility guide

Use `python scripts/reproduce.py --suite paper|supporting|all --output PATH` from a complete public checkout. Choose one suite name, not the literal pipe-separated expression. `paper` is the default; `PATH` must not exist. [The reproduction guide](../reproduction/README.md) lists exact runnable commands and outputs; [PAPER_MAP.md](PAPER_MAP.md) lists each current manuscript item.

## Implementation and original context

`src/` is the canonical implementation. `src/analysis.py` replaces three identical statistical copies. `src/focused/components.py` and `training.py` contain the original DISCO/JPEG components and fitting functions; only the fitting module's package import changed. `src/benchmark_v1.py` and `src/eng_compute.py` retain the published snapshot bytes. `src/reproduction/` owns aggregate replay and supporting plots. The 20 pre-existing base modules and all protocol values are unchanged.

Original admission, supervision, matrix orchestration, scientific job definitions and receipt analysis remain recoverable at the [preserved commit and source links](MIGRATION.md). They require the original admission context, protocol, implementation seal, jobs, ledger, receipts and checkpoints. Those gates remain applicable. No public adapter for full raw-image execution was added.

The base CLI can be inspected with `python scripts/run_study.py --help`. It retains the base framework's contract and is not a replacement for the focused-study orchestration.

Supporting replay binds all seven logical inputs to canonical repository paths and independent scientific digests. Provenance must agree with both; calculations parse the verified bytes without reopening metadata-directed paths. The compute interval and exception type are shared with `src/eng_compute.py`; the existing plotting command imports its figure serializer from `src/reproduction/primary_figure.py` and retains its execution gates.

## Data, weights and scientific limits

CIFAR-10 and CelebA must be acquired separately under their original terms. The base selection protocol uses a class-stratified 10% CIFAR-10 validation split with seed 314159. CelebA interfaces require annotation and partition files beneath `<data-root>/celeba/`. LPIPS/AlexNet weights and relevant victim/decoder checkpoints are also required for raw-image work; checkpoints are not distributed. Explicit synthetic test fixtures are not study observations.

Recorded environments are under `reproduction/environment/`; the duplicate protocol environment was consolidated into `results/protocol/environment.json`. Focused jobs recorded Torch `2.9.0+cu128`, scikit-image `0.25.0` and LPIPS `0.1.4`. Complete Python, TorchVision, NumPy, Pillow and JPEG-library versions were not recorded per job. Other snapshots cannot fill those gaps. Encoded JPEG streams were not retained, so codec-byte replay is not established.

Paper replay checks aggregates without third-party packages. Supporting replay recomputes retained statistics and plot signatures from aggregate metrics. Neither authenticates missing raw evidence, fits models, performs image inference or repeats hardware measurements. Retained four-condition analyses do not acquire DISCO coverage through relocation. Conditional intervals, coupled victim/attacker seeds and joint architecture/objective changes remain limitations.

## Environments and checks

Supporting installation reuses the exact NumPy `1.26.4`, SciPy `1.13.1` and Matplotlib `3.10.0` constraints in the root dependency lists. `reproduction/requirements-supporting.txt` installs only those direct dependencies. Optional scientific extras reuse the same root files; the pre-existing `openpyxl>=3.1` constraint is unchanged. This migration upgrades no dependency.

The full public gate uses Python 3.11:

```bash
python -m pip install -r requirements.txt -r reproduction/requirements-supporting.txt
./run_public_checks.sh
```

The gate verifies the pretrained weight hash in a temporary cache, export checks, exact public inventory, paper replay with site packages disabled, supporting replay and tests. In a Git checkout it uses `--require-staged`, so stage the intended public changes before running the gate. The staged audit checks exact index membership and equality with worktree bytes. The default audit includes cached paths even if their worktree files are absent. `--worktree-only` supports checks during an unstaged migration and explicitly leaves index verification incomplete. Installation and weight download need network access; aggregate replay is offline afterward.

For a source archive without `.git`, keep the extracted copy free of generated outputs. With a new external destination, this sequence succeeds without changing its public inventory:

```bash
python scripts/reproduce.py --output ../paper-replay-output
python scripts/audit_public_repository.py
```

The archive audit and full gate reject extra files inside that copy, including replay outputs. Use external destinations for the supporting and combined suites as well.

`reproduction/SHA256SUMS` covers the reproduction tree; the nested supporting manifest covers that input/reference subtree. Fixed policy hashes and `src/reproduction/integrity.py` independently preserve original scientific bytes. Updating a manifest cannot authorize altered results. Dataset files, checkpoints, manuscripts, private records and generated outputs stay outside the public allowlist.

A pre-existing weight can be reused offline by setting `PUBLIC_ALEXNET_WEIGHTS` to its path when running the gate. The gate checks the same fixed SHA-256 before copying it into its temporary cache. This option does not add weights to the public inventory.
