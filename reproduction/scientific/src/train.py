"""Compatibility CLI for schema-v2 encoder training.

The revision pipeline is the authoritative study runner. This module preserves
the historical callable/CLI surface while enforcing versioned checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .experiment import set_seed, train_encoder
from .models import ModelConfig


def parse_model_config(config_value: str | dict[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "schema_version": 2,
        "split_point": "current",
        "method": "standard",
        "bottleneck_channels": 6,
        "alpha": 0.0,
        "lr": 1e-3,
        "batch_size": 128,
        "feature_shape": [6, 8, 8],
    }
    if not config_value:
        return defaults
    if isinstance(config_value, dict):
        supplied = config_value
    else:
        path = Path(config_value)
        if path.is_file():
            supplied = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                supplied = json.loads(config_value)
            except json.JSONDecodeError:
                supplied = {}
                for pair in config_value.split(","):
                    if "=" not in pair:
                        continue
                    key, raw = (part.strip() for part in pair.split("=", 1))
                    if key in {"bottleneck_channels", "batch_size", "schema_version"}:
                        supplied[key] = int(raw)
                    elif key in {"alpha", "lr"}:
                        supplied[key] = float(raw)
                    else:
                        supplied[key] = raw
    defaults.update(supplied)
    split_point = str(defaults.get("split_point", "current"))
    channels = int(defaults.get("bottleneck_channels", 6))
    spatial = 16 if split_point == "early" else 8
    defaults["feature_shape"] = [channels, spatial, spatial]
    if "method" not in supplied:
        defaults["method"] = "afd" if float(defaults.get("alpha", 0.0)) > 0 else "standard"
    return defaults


def train(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "model_type", "proposed") != "proposed":
        raise ValueError("Edge-only training is outside the locked robustness study")
    config = ModelConfig.from_mapping(parse_model_config(getattr(args, "model_config", "")))
    save_path = Path(args.save_path)
    revision_root = Path("revisions/2026-07-22_adaptive_reconstruction_study").resolve()
    if revision_root not in save_path.resolve().parents:
        raise ValueError(
            f"New revision checkpoints must be stored under {revision_root}; got {save_path.resolve()}"
        )
    log_path = Path(getattr(args, "log_path", "")) if getattr(args, "log_path", "") else save_path.with_suffix(".json")
    if revision_root not in log_path.resolve().parents:
        raise ValueError(f"New revision logs must be stored under {revision_root}; got {log_path.resolve()}")
    return train_encoder(
        dataset=args.dataset,
        data_dir=getattr(args, "data_dir", "data"),
        config=config,
        seed=args.seed,
        epochs=args.epochs,
        checkpoint_path=save_path,
        log_path=log_path,
        device_name=getattr(args, "device", "auto"),
        num_workers=getattr(args, "num_workers", 0),
        test_only_synthetic=getattr(args, "test_only_synthetic", False),
        max_samples=getattr(args, "max_samples", None),
        selection_stage=getattr(args, "selection_stage", False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a schema-v2 split encoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=("cifar10", "celeba"), default="cifar10")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_config", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--log_path", default="")
    parser.add_argument("--model_type", choices=("proposed",), default="proposed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-only-synthetic", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--selection-stage", action="store_true")
    return parser


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2))
