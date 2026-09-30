# Adaptive Reconstruction Split Computing

This is the official repository for the manuscript on reconstruction risk, task utility, and bounded component compute in split computing. It contains the implementation, frozen protocol records, public verification tools, and aggregate numeric evidence.

The public boundary is exactly 123 files. It discloses 13 aggregate CSV tables with 2,308 rows: 1,752 rows from the validation-frozen reconstruction, semantic, and auxiliary-budget study, plus 556 benchmark and component-compute rows.

## Study scope

The empirical study evaluates reconstruction exposure and task utility for four fixed methods across two split interfaces and two datasets. A separate prospectively specified extension reports bounded central- and graphics-processor component measurements. The disclosed evidence supports setting-specific comparisons only.

- Primary reconstruction and utility comparisons use five predeclared model seeds.
- CelebA semantic probes use model seed as the inferential unit; attribute rows and stochastic draws are not independent replicates.
- The auxiliary-budget grid is descriptive because its original base table uses one predeclared seed.
- The benchmark extension uses paired model-attacker run seed as its analysis unit. Attacker restarts measure optimizer sensitivity and are not independent scientific replicates.
- The compute extension uses model seed as the inferential unit. Technical timing blocks quantify measurement noise; they are not inferential units.
- Reported intervals are descriptive two-sided Student-t intervals. The evidence does not establish formal privacy, universal method superiority, architecture-only attacker effects, equivalence, causal hardware effects, deployed end-to-end latency, or hardware-general conclusions.

## Public files

- `src/`: split models, defenses, data interfaces, experiment logic, metrics, audit helpers, and protocol utilities.
- `scripts/`: public pipeline, analysis, export, and verification entry points.
- `config/`: the frozen study contract and the executable implementation supplement.
- `results/raw/`: five public-safe aggregate base tables. The directory name refers to the disclosure layer; these are not private per-example records.
- `results/protocol/`: the sanitized environment, frozen-selection commitment, and schema-version-3 disclosure manifest.
- `release/v1.2.0/`: eight supplemental aggregate tables, two compute-analysis JSON files, seven schemas, release notes, and checksums.
- `tests/`, `.github/workflows/public-checks.yml`, and `run_public_checks.sh`: the public regression and boundary checks.

`results/protocol/disclosure_manifest.json` fixes the exact 123-path allowlist, all 13 CSV schemas and row counts, 58 verified public outputs, the exact Creative Commons license scope, and the authorized v1.3.0 repository-commit state.

## Aggregate reproduction (v1.3.0)

[`release/v1.3.0`](release/v1.3.0) contains seven input tables, full-precision Tables S1–S3, expected budget/retraining/compute outputs, diagnostic aggregates, pinned dependencies and offline reproduction code for Figures 2, 3 and 5. Its 556 input rows overlap earlier tables and do not represent new experimental runs. The 2,308-row count above describes the original 13-table disclosure; it excludes these reproduction copies and derived tables.

```bash
python -m pip install -r release/v1.3.0/requirements.txt
python release/v1.3.0/reproduce.py --output-dir /tmp/eng-reproduction --check
(cd release/v1.3.0 && sha256sum --check SHA256SUMS)
```

Use Python 3.11. After dependency installation this command runs offline on CPU. It reproduces calculations from aggregate metrics; it does not regenerate predictions, authenticate experiment execution or repeat timing measurements. The journal supplement is one PDF containing Tables S1–S3. See the directory README for its exact coverage and file index. Historical `release/v1.2.0` bytes, including its original staging notes, are retained unchanged.

## Environment

The recorded execution used Python 3.11.5. Package pins are listed in `requirements.txt`; figure-only dependencies are in `requirements-figures.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional figure dependencies:
python -m pip install -r requirements-figures.txt
```

These pins are not a hash-locked cross-platform environment. PyTorch wheel selection depends on the target central processor, graphics processor, and CUDA platform. If a recorded wheel is unavailable, record any substitution and do not describe the new environment as byte-identical.

## Dataset acquisition

Datasets and labels are not redistributed.

- CIFAR-10: obtain the standard `torchvision` distribution and place it under the directory passed through `--data-dir`. The locked protocol uses a class-stratified 10% validation subset with split seed 314159.
- CelebA: accept the official terms, obtain the aligned images and annotations, and place `list_attr_celeba.txt` and preferably `list_eval_partition.txt` under `<data-root>/celeba/`.

Synthetic samples are available only through explicit test flags and are never a silent substitute for study data. See `THIRD_PARTY_DATA.md` for the third-party rights boundary.

## Public verification

From a checkout containing only the 123 allowed files:

```bash
./run_public_checks.sh
```

The runner verifies both public exporters in public-only mode, the fixed inventory, SHA-256 values, schemas, row counts, licensing, citation policy, file-size ceiling, common secret patterns, source compilation, and public tests. It establishes integrity of the disclosed bytes. It does not authenticate unpublished private inputs or reproduce the experiments.

Verify the 18 versioned-release checksum entries from the repository root with:

```bash
(cd release/v1.2.0 && sha256sum --check SHA256SUMS)
```

Individual public-only checks are:

```bash
python scripts/export_eng_compute_public.py --check --public-only
python scripts/export_public_results.py --check --public-only
python scripts/audit_public_repository.py
```

In the full private workspace, stronger regeneration checks rebuild the public tables and v1.2.0 package from canonical private evidence and require the staged release to match the retained private candidate byte-for-byte:

```bash
python scripts/export_eng_compute_public.py --check
python scripts/export_public_results.py --check
python scripts/audit_public_repository.py --require-staged
```

The private commands require canonical manifests, audit records, checkpoints, timing records, or other evidence that is intentionally absent from a public-only checkout.

## Running the disclosed implementation

The CLI defaults to `config/study_contract.json` and accepts explicit `--contract`, `--revision-root`, `--data-dir`, `--device`, and `--num-workers` arguments.

```bash
python scripts/run_revision_pipeline.py --help
python scripts/run_revision_pipeline.py init \
  --revision-root work/adaptive-reconstruction
```

Experiment execution requires separately acquired datasets, suitable compute, and locally generated private artifacts. The aggregate tables are sufficient to inspect the disclosed results but not to regenerate private raw-evidence audits; v1.3.0 reproduces Figures 2, 3 and 5 from aggregate metrics.

## Excluded material

The deny-by-default `.gitignore` excludes manuscript and submission sources, bibliography files, datasets and third-party labels, checkpoints and learned perturbations, raw technical-block timings, images and sample tensors, per-example metrics and identifiers, semantic prediction arrays, the 240 MB frozen-selection artifact, private audit trees, local analysis candidates, caches, and the archived former Git state.

Every public file must be UTF-8 text and smaller than 1 MiB. The repository does not use Git Large File Storage.

## Checkpoint security

Never load an untrusted `.pt` file with this code. The manuscript-bound compatibility loaders in `src/checkpointing.py` and `src/defenses.py` retain `torch.load(..., weights_only=False)` for internally generated, schema-checked artifacts. That mode can execute code embedded in a malicious file. No checkpoint or perturbation binary is included here.

## Licensing

- Authored scripts, source modules, schemas, tests, and documentation are licensed under the MIT License in `LICENSE`.
- The aggregate tables, diagnostics and numeric analysis files named in `LICENSE-RESULTS` are licensed under CC BY 4.0 only to the extent the contributors own the applicable rights.
- Third-party datasets, labels, images, model weights, logos, templates, and assets are not covered by either grant and are not redistributed.

The Creative Commons grant does not claim third-party permission, supersede dataset terms, or license an underlying dataset. The licensed results are distributed as repository files.

## Citation and release status

`CITATION.cff` records software version `1.3.0` and intentionally omits a repository URL, release date, and DOI. The disclosure state `authorized_for_repository_commit` records the authorization for a normal commit and push of the exact file inventory. This version is distributed through the existing repository. No GitHub Release, tag, DOI or journal submission is created by this update.
