"""Feature-defense transforms with explicit stochasticity and norm bounds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def sample_laplace_like(
    features: torch.Tensor,
    scale: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if scale < 0:
        raise ValueError("Laplace scale must be non-negative")
    if scale == 0:
        return torch.zeros_like(features)
    uniform = torch.rand(
        features.shape,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    ).clamp_(1e-7, 1.0 - 1e-7) - 0.5
    return -scale * torch.sign(uniform) * torch.log1p(-2.0 * uniform.abs())


class UniversalPerturbation(nn.Module):
    def __init__(self, feature_shape: tuple[int, int, int], max_l2: float, initial_scale: float = 0.01):
        super().__init__()
        if max_l2 <= 0:
            raise ValueError("max_l2 must be positive")
        self.max_l2 = float(max_l2)
        self.noise = nn.Parameter(torch.randn((1, *feature_shape)) * initial_scale)
        self.project_()

    @torch.no_grad()
    def project_(self) -> None:
        norm = self.noise.norm(p=2)
        if norm > self.max_l2:
            self.noise.mul_(self.max_l2 / norm)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.noise

    def metadata(self) -> dict[str, float]:
        return {"max_l2": self.max_l2, "realized_l2": float(self.noise.detach().norm(p=2).item())}


@dataclass
class FeatureDefense:
    method: str
    laplace_scale: float = 0.0
    learned_noise: torch.Tensor | None = None
    learned_max_l2: float | None = None

    @property
    def stochastic(self) -> bool:
        return self.method == "laplace" and self.laplace_scale > 0

    def apply(
        self,
        features: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.method in {"standard", "afd"}:
            return features
        if self.method == "laplace":
            return features + sample_laplace_like(features, self.laplace_scale, generator=generator)
        if self.method == "learned":
            if self.learned_noise is None:
                raise ValueError("learned defense requires a perturbation tensor")
            noise = self.learned_noise.to(device=features.device, dtype=features.dtype)
            if tuple(noise.shape[1:]) != tuple(features.shape[1:]):
                raise ValueError(f"Learned perturbation shape {tuple(noise.shape)} is incompatible with {tuple(features.shape)}")
            if self.learned_max_l2 is not None and float(noise.norm(p=2)) > self.learned_max_l2 + 1e-6:
                raise ValueError("Learned perturbation exceeds its declared L2 limit")
            return features + noise
        raise ValueError(f"Unknown defense method: {self.method}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"method": self.method, "stochastic": self.stochastic}
        if self.method == "laplace":
            result["laplace_scale"] = self.laplace_scale
        if self.method == "learned":
            result.update(
                {
                    "learned_max_l2": self.learned_max_l2,
                    "learned_realized_l2": float(self.learned_noise.norm(p=2)) if self.learned_noise is not None else None,
                }
            )
        return result


def save_universal_perturbation(
    path: str | Path,
    module: UniversalPerturbation,
    metadata: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "noise": module.noise.detach().cpu(),
            "max_l2": module.max_l2,
            "realized_l2": float(module.noise.detach().norm(p=2)),
            "metadata": dict(metadata),
        },
        target,
    )


def load_universal_perturbation(path: str | Path, expected_shape: tuple[int, int, int]) -> FeatureDefense:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Missing learned perturbation: {target}")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        noise = payload
        max_l2 = float(noise.norm(p=2))
    else:
        noise = payload["noise"]
        max_l2 = float(payload["max_l2"])
    if tuple(noise.shape) != (1, *expected_shape):
        raise ValueError(f"Perturbation shape {tuple(noise.shape)} does not match {(1, *expected_shape)}")
    if float(noise.norm(p=2)) > max_l2 + 1e-6:
        raise ValueError("Stored perturbation violates its L2 constraint")
    return FeatureDefense("learned", learned_noise=noise, learned_max_l2=max_l2)
