"""Versioned split models and reconstruction attackers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = 2
SPLIT_SPATIAL_SIZE = {"early": 16, "current": 8}
METHODS = {"standard", "afd", "laplace", "learned"}


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int = SCHEMA_VERSION
    split_point: str = "current"
    method: str = "standard"
    bottleneck_channels: int = 6
    alpha: float = 0.0
    lr: float = 1e-3
    batch_size: int = 128
    feature_shape: tuple[int, int, int] = (6, 8, 8)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version={self.schema_version}")
        if self.split_point not in SPLIT_SPATIAL_SIZE:
            raise ValueError(f"Unsupported split_point={self.split_point}")
        if self.method not in METHODS:
            raise ValueError(f"Unsupported method={self.method}")
        expected = (
            self.bottleneck_channels,
            SPLIT_SPATIAL_SIZE[self.split_point],
            SPLIT_SPATIAL_SIZE[self.split_point],
        )
        if tuple(self.feature_shape) != expected:
            raise ValueError(f"feature_shape={self.feature_shape} is incompatible with {expected}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "ModelConfig":
        data = dict(values or {})
        split_point = str(data.get("split_point", "current"))
        channels = int(data.get("bottleneck_channels", 6))
        alpha = float(data.get("alpha", 0.0))
        method = str(data.get("method", "afd" if alpha > 0 else "standard"))
        spatial = SPLIT_SPATIAL_SIZE[split_point]
        raw_shape = data.get("feature_shape", (channels, spatial, spatial))
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            split_point=split_point,
            method=method,
            bottleneck_channels=channels,
            alpha=alpha,
            lr=float(data.get("lr", 1e-3)),
            batch_size=int(data.get("batch_size", 128)),
            feature_shape=tuple(map(int, raw_shape)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_shape"] = list(self.feature_shape)
        return result


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(inputs, self.alpha)


class EdgeEncoder(nn.Module):
    """Encoder emitting either 6 x 16 x 16 or 6 x 8 x 8 features."""

    def __init__(self, bottleneck_channels: int = 6, split_point: str = "current"):
        super().__init__()
        if split_point == "current":
            # Preserve legacy parameter keys for strict current-checkpoint migration.
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, bottleneck_channels, 3, 1, 1),
                nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True),
            )
        elif split_point == "early":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, bottleneck_channels, 3, 1, 1),
                nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported split_point={split_point}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class CloudClassifier(nn.Module):
    def __init__(self, bottleneck_channels: int = 6, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(bottleneck_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class AdversarialDecoder(nn.Module):
    """Existing deconvolutional MSE attacker for either locked feature shape."""

    def __init__(self, bottleneck_channels: int = 6, feature_spatial_size: int = 8):
        super().__init__()
        if feature_spatial_size == 8:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(bottleneck_channels, 64, 4, 2, 1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
            )
        elif feature_spatial_size == 16:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(bottleneck_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported feature spatial size={feature_spatial_size}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.decoder(features)
        if output.shape[-2:] != (32, 32):
            raise RuntimeError(f"Decoder produced {tuple(output.shape)} instead of Bx3x32x32")
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.norm1(self.conv1(inputs)), inplace=True)
        return F.relu(inputs + self.norm2(self.conv2(residual)), inplace=True)


class ResidualUpsamplingDecoder(nn.Module):
    """Higher-capacity residual upsampling attacker."""

    def __init__(self, bottleneck_channels: int = 6, feature_spatial_size: int = 8):
        super().__init__()
        if feature_spatial_size not in (8, 16):
            raise ValueError(f"Unsupported feature spatial size={feature_spatial_size}")
        self.project = nn.Conv2d(bottleneck_channels, 128, 3, 1, 1)
        stages = 2 if feature_spatial_size == 8 else 1
        blocks: list[nn.Module] = [ResidualBlock(128), ResidualBlock(128)]
        channels = 128
        for _ in range(stages):
            next_channels = channels // 2
            blocks.extend(
                (
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels, next_channels, 3, 1, 1),
                    nn.GroupNorm(8, next_channels),
                    nn.ReLU(inplace=True),
                    ResidualBlock(next_channels),
                )
            )
            channels = next_channels
        self.body = nn.Sequential(*blocks)
        self.output = nn.Sequential(nn.Conv2d(channels, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.output(self.body(F.relu(self.project(features), inplace=True)))
        if output.shape[-3:] != (3, 32, 32):
            raise RuntimeError(f"Residual decoder produced {tuple(output.shape)}")
        return output


class AFDSplitModel(nn.Module):
    def __init__(
        self,
        bottleneck_channels: int = 6,
        num_classes: int = 10,
        alpha: float = 1.0,
        split_point: str = "current",
        method: str | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig.from_mapping(
            {
                "bottleneck_channels": bottleneck_channels,
                "split_point": split_point,
                "method": method or ("afd" if alpha > 0 else "standard"),
                "alpha": alpha,
            }
        )
        spatial = self.config.feature_shape[-1]
        self.edge_encoder = EdgeEncoder(self.config.bottleneck_channels, self.config.split_point)
        self.cloud_classifier = CloudClassifier(self.config.bottleneck_channels, num_classes)
        self.adversarial_decoder = AdversarialDecoder(self.config.bottleneck_channels, spatial)
        self.grl = GradientReversal(self.config.alpha)

    def forward(self, inputs: torch.Tensor, run_recon: bool = True):
        features = self.edge_encoder(inputs)
        predictions = self.cloud_classifier(features)
        if run_recon:
            reconstruction = self.adversarial_decoder(self.grl(features))
            return predictions, reconstruction, features
        return predictions, features


class EdgeOnlyModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(32, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_attacker(architecture: str, feature_shape: tuple[int, int, int]) -> nn.Module:
    channels, height, width = feature_shape
    if height != width:
        raise ValueError(f"Non-square feature shape is unsupported: {feature_shape}")
    if architecture == "deconv_mse":
        return AdversarialDecoder(channels, height)
    if architecture == "residual_lpips":
        return ResidualUpsamplingDecoder(channels, height)
    raise ValueError(f"Unknown attacker architecture: {architecture}")
