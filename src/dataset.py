"""Dataset interfaces for the adaptive reconstruction robustness study.

Real data are mandatory by default. Synthetic data exist only for explicit,
test-only smoke runs and are never a silent fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR_SPLIT_SEED = 314159
CIFAR_VAL_FRACTION = 0.10
CELEBA_PARTITION_SIZES = {"train": 162_770, "val": 19_867, "test": 19_962}
CELEBA_TOTAL = sum(CELEBA_PARTITION_SIZES.values())


class DatasetAvailabilityError(RuntimeError):
    """Raised when a required real dataset is unavailable or malformed."""


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    train_size: int
    val_size: int
    test_size: int
    partition_source: str
    synthetic: bool


def _normalizer(dataset_name: str) -> transforms.Normalize:
    if dataset_name == "cifar10":
        return transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )
    if dataset_name == "celeba":
        return transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    raise ValueError(f"Unsupported dataset normalizer: {dataset_name}")


def get_transforms(is_train: bool = True, normalize: bool = True) -> transforms.Compose:
    """Backward-compatible CIFAR-10 transform factory."""
    steps: list[object] = []
    if is_train:
        steps.extend((transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()))
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(_normalizer("cifar10"))
    return transforms.Compose(steps)


def _celeba_transform(is_train: bool) -> transforms.Compose:
    # Aligned CelebA images are 178 x 218. Crop the centered 178 x 178 face region
    # before resizing, as declared in the study contract.
    steps: list[object] = [transforms.CenterCrop(178)]
    if is_train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend((transforms.Resize((32, 32)), transforms.ToTensor()))
    return transforms.Compose(steps)


class SyntheticCIFAR10(Dataset):
    """Deterministic test fixture, available only through an explicit flag."""

    def __init__(self, train: bool = True, transform: object | None = None, size: int | None = None):
        self.train = train
        self.transform = transform
        self.size = size if size is not None else (1_000 if train else 200)
        self.targets = [index % 10 for index in range(self.size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, int]:
        generator = torch.Generator().manual_seed(index + (0 if self.train else 1_000_000))
        label = self.targets[index]
        image = torch.rand((3, 32, 32), generator=generator) * 0.20
        row = 4 + (label // 5) * 16
        col = 3 + (label % 5) * 6
        image[:, row : row + 8, col : col + 5] += 0.75
        image = image.clamp(0.0, 1.0)
        pil_image = transforms.ToPILImage()(image)
        return (self.transform(pil_image) if self.transform else pil_image), label


class NormalizedImageDataset(Dataset):
    """Return normalized input, raw reconstruction target, and task label."""

    def __init__(self, base_dataset: Dataset, dataset_name: str):
        self.base_dataset = base_dataset
        self.normalize = _normalizer(dataset_name)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_raw, label = self.base_dataset[index]
        if not isinstance(image_raw, torch.Tensor):
            image_raw = transforms.ToTensor()(image_raw)
        return self.normalize(image_raw), image_raw, int(label)


class SubsetNormalizedDataset(Dataset):
    """Compatibility wrapper for a fixed CIFAR-10 index subset."""

    def __init__(
        self,
        data_dir: str,
        train_indices_or_val_indices: Sequence[int],
        train: bool = True,
        is_train_subset: bool = True,
        download: bool = False,
        test_only_synthetic: bool = False,
    ) -> None:
        transform = get_transforms(is_train=is_train_subset, normalize=False)
        self.base_dataset = _load_cifar_base(
            data_dir,
            train=train,
            transform=transform,
            download=download,
            test_only_synthetic=test_only_synthetic,
        )
        self.indices = list(map(int, train_indices_or_val_indices))
        self.normalize = _normalizer("cifar10")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_raw, label = self.base_dataset[self.indices[index]]
        if not isinstance(image_raw, torch.Tensor):
            image_raw = transforms.ToTensor()(image_raw)
        return self.normalize(image_raw), image_raw, int(label)


def _load_cifar_base(
    data_dir: str,
    *,
    train: bool,
    transform: object,
    download: bool,
    test_only_synthetic: bool,
) -> Dataset:
    if test_only_synthetic:
        return SyntheticCIFAR10(train=train, transform=transform)
    try:
        return datasets.CIFAR10(root=data_dir, train=train, download=download, transform=transform)
    except Exception as error:
        raise DatasetAvailabilityError(
            "Real CIFAR-10 is required. Place the extracted dataset under the data root, "
            "enable an explicit download, or use test_only_synthetic=True only in tests."
        ) from error


def fixed_stratified_cifar_indices(
    targets: Sequence[int],
    val_fraction: float = CIFAR_VAL_FRACTION,
    split_seed: int = CIFAR_SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Create fixed per-class membership independent of the model seed."""
    targets_array = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_id in sorted(np.unique(targets_array).tolist()):
        class_indices = np.flatnonzero(targets_array == class_id)
        class_indices = rng.permutation(class_indices)
        val_count = int(round(len(class_indices) * val_fraction))
        val_indices.extend(map(int, class_indices[:val_count]))
        train_indices.extend(map(int, class_indices[val_count:]))
    return sorted(train_indices), sorted(val_indices)


def _limit_indices(indices: Sequence[int], max_samples: int | None) -> list[int]:
    result = list(map(int, indices))
    return result if max_samples is None else result[:max_samples]


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    val_split: float = CIFAR_VAL_FRACTION,
    seed: int = 42,
    *,
    split_seed: int = CIFAR_SPLIT_SEED,
    download: bool = False,
    test_only_synthetic: bool = False,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 with fixed class-stratified train/validation membership."""
    if val_split != CIFAR_VAL_FRACTION:
        raise ValueError(f"The locked protocol requires val_split={CIFAR_VAL_FRACTION}")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    membership_base = _load_cifar_base(
        data_dir,
        train=True,
        transform=get_transforms(is_train=False, normalize=False),
        download=download,
        test_only_synthetic=test_only_synthetic,
    )
    targets = getattr(membership_base, "targets", None)
    if targets is None:
        raise DatasetAvailabilityError("CIFAR-10 target labels are unavailable for stratification.")
    train_indices, val_indices = fixed_stratified_cifar_indices(targets, val_split, split_seed)
    train_indices = _limit_indices(train_indices, max_samples)
    val_indices = _limit_indices(val_indices, max_samples)

    train_dataset = SubsetNormalizedDataset(
        data_dir,
        train_indices,
        is_train_subset=True,
        download=download,
        test_only_synthetic=test_only_synthetic,
    )
    val_dataset = SubsetNormalizedDataset(
        data_dir,
        val_indices,
        is_train_subset=False,
        download=download,
        test_only_synthetic=test_only_synthetic,
    )
    test_base = _load_cifar_base(
        data_dir,
        train=False,
        transform=get_transforms(is_train=False, normalize=False),
        download=download,
        test_only_synthetic=test_only_synthetic,
    )
    test_dataset: Dataset = NormalizedImageDataset(test_base, "cifar10")
    if max_samples is not None:
        test_dataset = Subset(test_dataset, range(min(max_samples, len(test_dataset))))

    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


class CelebADataset(Dataset):
    """CelebA using official partition membership and all 40 attributes."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: object | None = None,
        include_attributes: bool = False,
    ) -> None:
        if split not in CELEBA_PARTITION_SIZES:
            raise ValueError(f"Unknown CelebA split: {split}")
        root = Path(data_dir) / "celeba"
        self.image_dir = root / "images"
        self.attr_file = root / "list_attr_celeba.txt"
        self.partition_file = root / "list_eval_partition.txt"
        if not self.image_dir.is_dir() or not self.attr_file.is_file():
            raise DatasetAvailabilityError(
                "Real CelebA images and list_attr_celeba.txt are required under <data_dir>/celeba/."
            )
        self.transform = transform or _celeba_transform(is_train=(split == "train"))
        self.include_attributes = include_attributes

        with self.attr_file.open("r", encoding="utf-8") as handle:
            declared_count = int(handle.readline().strip())
            self.attr_names = handle.readline().strip().split()
            rows = [line.split() for line in handle if line.strip()]
        if declared_count != CELEBA_TOTAL or len(rows) != CELEBA_TOTAL or len(self.attr_names) != 40:
            raise DatasetAvailabilityError(
                f"CelebA metadata must contain {CELEBA_TOTAL} images and 40 attributes; "
                f"found declared={declared_count}, rows={len(rows)}, attrs={len(self.attr_names)}."
            )
        self.smiling_index = self.attr_names.index("Smiling")

        if self.partition_file.is_file():
            partition_map = {
                name: int(partition)
                for name, partition in (
                    line.split() for line in self.partition_file.read_text(encoding="utf-8").splitlines() if line.strip()
                )
            }
            partition_id = {"train": 0, "val": 1, "test": 2}[split]
            selected = [row for row in rows if partition_map.get(row[0]) == partition_id]
            self.partition_source = "list_eval_partition.txt"
        else:
            if any(row[0] != f"{index:06d}.jpg" for index, row in enumerate(rows, start=1)):
                raise DatasetAvailabilityError(
                    "CelebA canonical-boundary fallback requires list_attr_celeba.txt in exact 000001.jpg--202599.jpg order."
                )
            boundaries = {
                "train": (0, CELEBA_PARTITION_SIZES["train"]),
                "val": (CELEBA_PARTITION_SIZES["train"], CELEBA_PARTITION_SIZES["train"] + CELEBA_PARTITION_SIZES["val"]),
                "test": (CELEBA_PARTITION_SIZES["train"] + CELEBA_PARTITION_SIZES["val"], CELEBA_TOTAL),
            }
            start, end = boundaries[split]
            selected = rows[start:end]
            self.partition_source = "canonical_boundaries"
        if len(selected) != CELEBA_PARTITION_SIZES[split]:
            raise DatasetAvailabilityError(
                f"CelebA {split} partition has {len(selected)} rows; expected {CELEBA_PARTITION_SIZES[split]}."
            )
        self.records = selected

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = self.image_dir / row[0]
        if not image_path.is_file():
            raise DatasetAvailabilityError(f"Missing CelebA image: {image_path}")
        image_raw = self.transform(Image.open(image_path).convert("RGB"))
        attributes = torch.tensor([1 if value == "1" else 0 for value in row[1:]], dtype=torch.float32)
        label = int(attributes[self.smiling_index].item())
        normalized = _normalizer("celeba")(image_raw)
        if self.include_attributes:
            return normalized, image_raw, label, attributes
        return normalized, image_raw, label


def get_celeba_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
    *,
    num_workers: int = 4,
    include_attributes: bool = False,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets_by_split: dict[str, Dataset] = {
        split: CelebADataset(
            data_dir,
            split=split,
            transform=_celeba_transform(is_train=(split == "train")),
            include_attributes=include_attributes,
        )
        for split in ("train", "val", "test")
    }
    if max_samples is not None:
        datasets_by_split = {
            split: Subset(dataset, range(min(max_samples, len(dataset))))
            for split, dataset in datasets_by_split.items()
        }
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(datasets_by_split["train"], shuffle=True, generator=generator, **common),
        DataLoader(datasets_by_split["val"], shuffle=False, **common),
        DataLoader(datasets_by_split["test"], shuffle=False, **common),
    )


def get_dataset_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    seed: int,
    *,
    num_workers: int = 0,
    test_only_synthetic: bool = False,
    include_attributes: bool = False,
    max_samples: int | None = None,
) -> tuple[tuple[DataLoader, DataLoader, DataLoader], int, DatasetMetadata]:
    if dataset_name == "cifar10":
        loaders = get_cifar10_loaders(
            data_dir,
            batch_size,
            num_workers,
            seed=seed,
            test_only_synthetic=test_only_synthetic,
            max_samples=max_samples,
        )
        metadata = DatasetMetadata(
            "cifar10",
            len(loaders[0].dataset),
            len(loaders[1].dataset),
            len(loaders[2].dataset),
            f"fixed_stratified_seed_{CIFAR_SPLIT_SEED}",
            test_only_synthetic,
        )
        return loaders, 10, metadata
    if dataset_name == "celeba":
        if test_only_synthetic:
            raise ValueError("Synthetic CelebA is not implemented; use synthetic CIFAR-10 for test-only runs.")
        loaders = get_celeba_loaders(
            data_dir,
            batch_size,
            seed,
            num_workers=num_workers,
            include_attributes=include_attributes,
            max_samples=max_samples,
        )
        base = loaders[0].dataset.dataset if isinstance(loaders[0].dataset, Subset) else loaders[0].dataset
        metadata = DatasetMetadata(
            "celeba",
            len(loaders[0].dataset),
            len(loaders[1].dataset),
            len(loaders[2].dataset),
            getattr(base, "partition_source", "unknown"),
            False,
        )
        return loaders, 2, metadata
    raise ValueError(f"Unsupported locked-study dataset: {dataset_name}")


def get_budget_pool_and_test_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    seed: int,
    *,
    num_workers: int = 0,
    test_only_synthetic: bool = False,
    max_samples: int | None = None,
) -> tuple[tuple[DataLoader, DataLoader, DataLoader], int, DatasetMetadata]:
    """Return fit/evaluation views of the training pool plus the test loader.

    Corrected auxiliary-budget early stopping splits its validation pairs from the
    capped training pool. The ordinary validation partition is deliberately absent
    from this interface so budget execution cannot consume it accidentally. The two
    pool views have identical membership: only the fitting view uses training-time
    augmentation, while the internal-validation view uses deterministic evaluation
    preprocessing.
    """
    if dataset_name == "cifar10":
        membership_base = _load_cifar_base(
            data_dir,
            train=True,
            transform=get_transforms(is_train=False, normalize=False),
            download=False,
            test_only_synthetic=test_only_synthetic,
        )
        targets = getattr(membership_base, "targets", None)
        if targets is None:
            raise DatasetAvailabilityError("CIFAR-10 target labels are unavailable for stratification.")
        train_indices, _ordinary_validation_indices = fixed_stratified_cifar_indices(targets)
        train_indices = _limit_indices(train_indices, max_samples)
        fitting_pool = SubsetNormalizedDataset(
            data_dir,
            train_indices,
            is_train_subset=True,
            test_only_synthetic=test_only_synthetic,
        )
        validation_pool = SubsetNormalizedDataset(
            data_dir,
            train_indices,
            is_train_subset=False,
            test_only_synthetic=test_only_synthetic,
        )
        test_base = _load_cifar_base(
            data_dir,
            train=False,
            transform=get_transforms(is_train=False, normalize=False),
            download=False,
            test_only_synthetic=test_only_synthetic,
        )
        test_dataset: Dataset = NormalizedImageDataset(test_base, "cifar10")
        if max_samples is not None:
            test_dataset = Subset(test_dataset, range(min(max_samples, len(test_dataset))))
        common = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        loaders = (
            DataLoader(
                fitting_pool,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                **common,
            ),
            DataLoader(validation_pool, shuffle=False, **common),
            DataLoader(test_dataset, shuffle=False, **common),
        )
        metadata = DatasetMetadata(
            "cifar10",
            len(fitting_pool),
            5_000 if not test_only_synthetic else len(_ordinary_validation_indices),
            len(test_dataset),
            f"fixed_stratified_seed_{CIFAR_SPLIT_SEED}",
            test_only_synthetic,
        )
        return loaders, 10, metadata
    if dataset_name == "celeba":
        if test_only_synthetic:
            raise ValueError("Synthetic CelebA is not implemented.")
        fitting_pool: Dataset = CelebADataset(data_dir, "train", _celeba_transform(True))
        validation_pool: Dataset = CelebADataset(data_dir, "train", _celeba_transform(False))
        test_dataset = CelebADataset(data_dir, "test", _celeba_transform(False))
        partition_source = fitting_pool.partition_source
        if max_samples is not None:
            fitting_pool = Subset(fitting_pool, range(min(max_samples, len(fitting_pool))))
            validation_pool = Subset(
                validation_pool,
                range(min(max_samples, len(validation_pool))),
            )
            test_dataset = Subset(test_dataset, range(min(max_samples, len(test_dataset))))
        common = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        loaders = (
            DataLoader(
                fitting_pool,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                **common,
            ),
            DataLoader(validation_pool, shuffle=False, **common),
            DataLoader(test_dataset, shuffle=False, **common),
        )
        metadata = DatasetMetadata(
            "celeba",
            len(fitting_pool),
            CELEBA_PARTITION_SIZES["val"],
            len(test_dataset),
            partition_source,
            False,
        )
        return loaders, 2, metadata
    raise ValueError(f"Unsupported locked-study dataset: {dataset_name}")


def get_selection_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    seed: int,
    *,
    num_workers: int = 0,
    test_only_synthetic: bool = False,
    max_samples: int | None = None,
) -> tuple[tuple[DataLoader, DataLoader], int, DatasetMetadata]:
    """Return train/validation loaders without constructing or reading a test dataset."""
    if dataset_name == "cifar10":
        membership_base = _load_cifar_base(
            data_dir,
            train=True,
            transform=get_transforms(is_train=False, normalize=False),
            download=False,
            test_only_synthetic=test_only_synthetic,
        )
        targets = getattr(membership_base, "targets", None)
        if targets is None:
            raise DatasetAvailabilityError("CIFAR-10 target labels are unavailable for stratification.")
        train_indices, val_indices = fixed_stratified_cifar_indices(targets)
        train_indices = _limit_indices(train_indices, max_samples)
        val_indices = _limit_indices(val_indices, max_samples)
        train_dataset = SubsetNormalizedDataset(
            data_dir,
            train_indices,
            is_train_subset=True,
            test_only_synthetic=test_only_synthetic,
        )
        val_dataset = SubsetNormalizedDataset(
            data_dir,
            val_indices,
            is_train_subset=False,
            test_only_synthetic=test_only_synthetic,
        )
        common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
        generator = torch.Generator().manual_seed(seed)
        loaders = (
            DataLoader(train_dataset, shuffle=True, generator=generator, **common),
            DataLoader(val_dataset, shuffle=False, **common),
        )
        metadata = DatasetMetadata(
            "cifar10",
            len(train_dataset),
            len(val_dataset),
            0,
            f"fixed_stratified_seed_{CIFAR_SPLIT_SEED}",
            test_only_synthetic,
        )
        return loaders, 10, metadata
    if dataset_name == "celeba":
        if test_only_synthetic:
            raise ValueError("Synthetic CelebA is not implemented.")
        train_dataset: Dataset = CelebADataset(data_dir, "train", _celeba_transform(True))
        val_dataset: Dataset = CelebADataset(data_dir, "val", _celeba_transform(False))
        partition_source = train_dataset.partition_source
        if max_samples is not None:
            train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(max_samples, len(val_dataset))))
        common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
        generator = torch.Generator().manual_seed(seed)
        loaders = (
            DataLoader(train_dataset, shuffle=True, generator=generator, **common),
            DataLoader(val_dataset, shuffle=False, **common),
        )
        metadata = DatasetMetadata(
            "celeba", len(train_dataset), len(val_dataset), 0, partition_source, False
        )
        return loaders, 2, metadata
    raise ValueError(f"Unsupported locked-study dataset: {dataset_name}")


# Legacy SVHN access is intentionally removed from the locked revision protocol.
def get_svhn_loaders(*args, **kwargs):
    raise ValueError("SVHN is outside the locked revision study; use CIFAR-10 or CelebA.")
