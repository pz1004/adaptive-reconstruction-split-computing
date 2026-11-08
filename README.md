# Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker Configuration

Official code and aggregate results for the paper by **Sooyoung Jang, Sungpil Woo, and Ahyun Lee**. The manuscript is **under review**.

The study examines how measured image reconstruction exposure changes with attacker configuration, and how task utility relates to the bytes sent across a split-computing interface. Experiments use CIFAR-10 and CelebA, two encoder–interface configurations, two reconstruction decoders, an architecture-adapted DISCO comparator, and a fixed-victim JPEG comparison.

## Reproduce the paper's aggregate results

Run from the repository root with **Python 3.10 or later**. This command uses only the Python standard library and the included CSVs; it runs offline without images, model weights, or a GPU.

```bash
python reproduction/replay_aggregates.py \
  --data reproduction/aggregate-data \
  --output replay-output
```

Use an output directory that does not already exist. A successful run writes `replay-output/verification.json` with status `PASS_CPU_AGGREGATE_REPLAY`. It reconstructs **1,140 descriptive rows**, four decoder contrasts, 90 utility comparisons, DISCO selections and training-history summaries, and JPEG selections. Numerical checks use absolute tolerance `1e-12`.

See the [reproduction guide](reproduction/README.md) for outputs and scope, the [data guide](reproduction/aggregate-data/README.md) for table contents, and [reproducibility details](docs/REPRODUCIBILITY.md) for the experimental environment and source layout.

## Study scope

- At the higher-resolution interface, ten paired runs per dataset give an AFD-minus-standard SSIM contrast of approximately −0.14 for the deconvolutional/MSE decoder and −0.01 for the residual/perceptual decoder. Architecture and objective vary together.
- The two interface settings also differ in encoder configuration. Their comparison does not isolate a causal resolution effect.
- DISCO is adapted to six-channel features and evaluated with a ten-epoch recipe. Training adequacy is unestablished; infeasible validation settings and test utility failures remain in the tables.
- Validation-qualified JPEG points use fewer bytes than corresponding uncompressed float32 feature tensors under the fixed-victim protocol. JPEG changes server exposure and computation placement; compressed-feature codecs and deployed network performance were not evaluated.
- Five/four-run and ten/nine-run cohorts remain separate. Model runs, attacker restarts, attributes and timing blocks are not interchangeable independent replicates. Between-run spread describes the evaluated pipelines.

These results characterize achieved reconstruction and task utility under the stated protocols. They do not establish practical privacy protection, general defense superiority, or measured deployment savings.

## Repository layout

| Path | Contents |
| --- | --- |
| [`reproduction/`](reproduction/README.md) | Paper aggregate tables, portable replay, scientific source snapshots, recorded environments and provenance. |
| [`src/`](src/) | Base split-computing models, data interfaces, training, evaluation and protocol utilities. |
| [`config/`](config/) | Study contract and implementation protocol. |
| [`results/raw/`](results/raw/) | Model-, attribute- and decoder-level aggregate tables; these are not per-image records. |
| [`release/v1.3.0/`](release/v1.3.0/README.md) | Versioned supporting analyses for reconstruction, semantics, attacker budgets and component compute. |
| [`scripts/`](scripts/) | Base experiment, analysis and repository-verification entry points. |
| [`tests/`](tests/) | Reproduction, scientific-protocol and public-file-boundary checks. |

Scientific snapshots retain their recorded filenames and provenance identifiers. The source paths in provenance records identify the captured inputs; they are not dependencies of aggregate replay.

## Supporting analyses

The versioned analysis package reproduces calculations and figure data from aggregate measurements with Python 3.11:

```bash
python -m pip install -r release/v1.3.0/requirements.txt
python release/v1.3.0/reproduce.py --output-dir supporting-output --check
(cd release/v1.3.0 && sha256sum --check SHA256SUMS)
```

Its inputs and outputs overlap other tables in this repository; row counts must not be added as independent experimental runs. See its [coverage and file index](release/v1.3.0/README.md).

## Experimental source and data

Aggregate replay is self-contained. Training and raw-image evaluation require separately acquired datasets, pretrained weights, suitable compute and the execution context described in the [reproducibility guide](docs/REPRODUCIBILITY.md). The scientific orchestration snapshots retain their recorded layout and receipt requirements; they are not a turnkey raw-image training command.

Dataset images, annotations, model checkpoints, per-image predictions and editorial correspondence are not distributed. Dataset acquisition and reuse are governed by the [third-party data notice](THIRD_PARTY_DATA.md).

**Never load an untrusted `.pt` file** with the scientific code. Its compatibility loaders use `torch.load(..., weights_only=False)` for internally generated artifacts; that mode can execute code from a malicious checkpoint. This repository contains no checkpoint binaries.

## Verification

Verify the current reproduction files with:

```bash
(cd reproduction && sha256sum --check SHA256SUMS)
python scripts/audit_public_repository.py
```

With the pinned test dependencies installed, `./run_public_checks.sh` checks the exact public-file inventory, both aggregate-analysis packages and scientific-protocol tests. The complete test suite uses Python 3.11 and downloads a hash-verified AlexNet weight for metric tests; the aggregate replay above does not. GitHub Actions runs the same checks.

## Citation and licenses

[CITATION.cff](CITATION.cff) contains the paper title, author order and under-review status, together with the software citation. No publication DOI is assigned here.

Authored code and documentation use the [MIT License](LICENSE). Contributor-owned aggregate results use CC BY 4.0 for the exact files listed in [LICENSE-RESULTS](LICENSE-RESULTS). The licensed results are distributed as repository files. This grant does not claim third-party permission or override dataset terms. The pinned DISCO source retains its [Apache-2.0 license](reproduction/upstream/LICENSE) and [attribution](reproduction/THIRD_PARTY_DATA.md).
