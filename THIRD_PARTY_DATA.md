# Third-party data notice

This repository does not redistribute CIFAR-10 or CelebA images, annotation
files, the full CelebA attribute-name list, sample tensors, or per-example
records.

- CIFAR-10 must be obtained from its official distribution or through the
  dataset tooling used by `torchvision`.
- CelebA access and use remain subject to the official
  [CelebA terms and download process](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

`results/raw/semantic_attribute_results.csv` contains aggregate measurements
indexed by the ordered CelebA annotation columns. Those indices are linkable by
anyone who lawfully has the annotation schema; removing the names is a
non-redistribution measure, not anonymization. Index 31, the supervised task
target, is excluded from that table.

The MIT grant applies to authored code, schemas, tests, and documentation. The
CC BY 4.0 grant applies only to the aggregate tables, diagnostics and numeric analysis files named in `LICENSE-RESULTS`, and only to rights the
contributors can grant. Neither license grants rights to third-party data or
assets.

The official CelebA terms limit use to non-commercial research and address
commercial exploitation of derived data. This repository does not make a legal
determination that aggregate measurements are outside that clause. CC BY 4.0
permits commercial reuse only for rights the licensor can grant; it does not
override the CelebA terms.

The exact CC BY 4.0 scope is listed in LICENSE-RESULTS. Distribution as repository
files is not a representation that MMLAB, image owners, or another
rights holder authorized commercial use. No third-party permission or waiver is
claimed, no third-party terms are superseded, and recipients remain responsible
for rights outside the contributors' license grant.
