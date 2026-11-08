"""Per-image reconstruction metrics and strict record validation."""

from __future__ import annotations

import math
import hashlib
import importlib.util
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from skimage.metrics import structural_similarity


METRIC_FIELDS = ("mse", "psnr", "ssim", "lpips")
LPIPS_BACKBONE = "alex"
LPIPS_PACKAGE_WEIGHT_RELATIVE_PATH = Path("weights/v0.1/alex.pth")
LPIPS_PACKAGE_WEIGHT_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
LPIPS_BACKBONE_WEIGHT_FILENAME = "alexnet-owt-7be5be79.pth"
LPIPS_BACKBONE_WEIGHT_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lpips_backbone_weight_path() -> Path:
    """Return the AlexNet checkpoint path controlled by PyTorch's hub cache."""
    return Path(torch.hub.get_dir()) / "checkpoints" / LPIPS_BACKBONE_WEIGHT_FILENAME


def verify_lpips_weight_files() -> dict[str, dict[str, str]]:
    """Verify both pretrained files used by lpips.LPIPS(net='alex')."""
    specification = importlib.util.find_spec("lpips")
    if specification is None or specification.origin is None:
        raise RuntimeError("The pinned lpips package is not importable")
    package_weight = Path(specification.origin).resolve().parent / LPIPS_PACKAGE_WEIGHT_RELATIVE_PATH
    backbone_weight = lpips_backbone_weight_path()
    expected = {
        "lpips_package_weight": (package_weight, LPIPS_PACKAGE_WEIGHT_SHA256),
        "torchvision_alexnet_weight": (backbone_weight, LPIPS_BACKBONE_WEIGHT_SHA256),
    }
    result: dict[str, dict[str, str]] = {}
    for role, (path, digest) in expected.items():
        if not path.is_file():
            raise RuntimeError(f"Missing pretrained weight for {role}: {path}")
        observed = _sha256_file(path)
        if observed != digest:
            raise RuntimeError(f"Pretrained-weight hash mismatch for {role}: {observed} != {digest}")
        result[role] = {"path": str(path), "sha256": observed}
    return result


class LPIPSMetric:
    def __init__(self, device: torch.device | str):
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("The residual_lpips attacker requires the 'lpips' package.") from error
        self.model = lpips.LPIPS(net=LPIPS_BACKBONE).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def differentiable(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        return self.model(target * 2.0 - 1.0, reconstruction * 2.0 - 1.0).mean()

    @torch.no_grad()
    def per_image(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        values = self.model(target * 2.0 - 1.0, reconstruction * 2.0 - 1.0)
        return values.flatten(start_dim=1).mean(dim=1)


def per_image_mse(target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    return (target - reconstruction).square().flatten(start_dim=1).mean(dim=1)


def per_image_psnr(target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    mse = per_image_mse(target, reconstruction).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def per_image_ssim(target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    reconstruction_np = reconstruction.detach().cpu().numpy().transpose(0, 2, 3, 1)
    values = [
        structural_similarity(
            np.clip(original, 0.0, 1.0),
            np.clip(reconstructed, 0.0, 1.0),
            data_range=1.0,
            channel_axis=2,
        )
        for original, reconstructed in zip(target_np, reconstruction_np)
    ]
    return torch.tensor(values, dtype=torch.float64)


def reconstruction_metric_batch(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    lpips_metric: LPIPSMetric,
) -> dict[str, list[float]]:
    values = {
        "mse": per_image_mse(target, reconstruction).detach().cpu(),
        "psnr": per_image_psnr(target, reconstruction).detach().cpu(),
        "ssim": per_image_ssim(target, reconstruction),
        "lpips": lpips_metric.per_image(target, reconstruction).detach().cpu(),
    }
    return {name: [float(value) for value in tensor.tolist()] for name, tensor in values.items()}


def validate_per_image_metrics(record: dict[str, Any], expected_count: int | None = None) -> None:
    missing = set(METRIC_FIELDS) - set(record)
    if missing:
        raise ValueError(f"Partial metric record; missing {sorted(missing)}")
    lengths = {len(record[field]) for field in METRIC_FIELDS}
    if len(lengths) != 1:
        raise ValueError(f"Metric arrays have inconsistent lengths: {lengths}")
    count = next(iter(lengths))
    if expected_count is not None and count != expected_count:
        raise ValueError(f"Metric record has {count} images; expected {expected_count}")
    for field in METRIC_FIELDS:
        if not all(math.isfinite(float(value)) for value in record[field]):
            raise ValueError(f"Metric {field} contains NaN or infinity")


def summarize_per_image_metrics(record: dict[str, list[float]]) -> dict[str, float]:
    validate_per_image_metrics(record)
    return {
        f"{field}_{statistic}": float(function(np.asarray(record[field], dtype=float)))
        for field in METRIC_FIELDS
        for statistic, function in (("mean", np.mean), ("sample_sd", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0))
    }


def metric_versions() -> dict[str, str]:
    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "missing"

    return {
        "mse": f"torch-{torch.__version__}-per-image-mean",
        "psnr": "10log10(1/per-image-mse)",
        "ssim": f"scikit-image-{version('scikit-image')}",
        "lpips": f"lpips-{version('lpips')}-alex",
    }
