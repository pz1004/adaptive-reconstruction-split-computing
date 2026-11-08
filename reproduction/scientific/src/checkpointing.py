"""Checkpoint schema, validation, and legacy migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .models import AFDSplitModel, ModelConfig, SCHEMA_VERSION


class CheckpointCompatibilityError(RuntimeError):
    pass


def migrate_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an in-memory schema-v2 view without modifying the source file."""
    migrated = dict(payload)
    if "model_state_dict" not in migrated:
        raise CheckpointCompatibilityError("Checkpoint lacks model_state_dict")
    raw_config = dict(migrated.get("config", {}))
    if int(raw_config.get("schema_version", 1)) < SCHEMA_VERSION:
        alpha = float(raw_config.get("alpha", 0.0))
        raw_config.update(
            {
                "schema_version": SCHEMA_VERSION,
                "split_point": "current",
                "method": "afd" if alpha > 0 else "standard",
                "feature_shape": [int(raw_config.get("bottleneck_channels", 6)), 8, 8],
            }
        )
        migrated["migration"] = {
            "source_schema_version": 1,
            "target_schema_version": SCHEMA_VERSION,
            "assumed_split_point": "current",
            "source_file_unchanged": True,
        }
    migrated["config"] = ModelConfig.from_mapping(raw_config).to_dict()
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError(f"Checkpoint {checkpoint_path} is not a mapping")
    return migrate_checkpoint_payload(payload)


def load_split_model(
    path: str | Path,
    *,
    num_classes: int,
    device: torch.device | str,
    expected_feature_shape: tuple[int, int, int] | None = None,
) -> tuple[AFDSplitModel, dict[str, Any]]:
    payload = load_checkpoint(path, device)
    config = ModelConfig.from_mapping(payload["config"])
    if expected_feature_shape is not None and tuple(config.feature_shape) != tuple(expected_feature_shape):
        raise CheckpointCompatibilityError(
            f"Checkpoint feature shape {config.feature_shape} does not match {expected_feature_shape}"
        )
    model = AFDSplitModel(num_classes=num_classes, config=config).to(device)
    state_dict = {key.removeprefix("module."): value for key, value in payload["model_state_dict"].items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"Incompatible checkpoint tensors in {path}: {error}") from error
    return model, payload


def checkpoint_payload(
    model: AFDSplitModel,
    *,
    dataset: str,
    seed: int,
    selected_epoch: int,
    validation_accuracy: float,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "epoch": int(selected_epoch),
        "selected_epoch": int(selected_epoch),
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "feature_shape": list(model.config.feature_shape),
        "dataset": dataset,
        "seed": int(seed),
        "model_type": "proposed",
        "validation_accuracy": float(validation_accuracy),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    return payload
