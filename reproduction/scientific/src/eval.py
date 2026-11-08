"""Strict checkpoint evaluation compatibility CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .checkpointing import load_split_model
from .dataset import get_dataset_loaders
from .defenses import FeatureDefense, load_universal_perturbation
from .experiment import TEST_DRAW_SEEDS, classifier_accuracy, evaluate_adaptive_attacker, resolve_device
from .train import parse_model_config


def calculate_ssim(target: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """Backward-compatible batch SSIM using the exact per-image implementation."""
    from .metrics import per_image_ssim

    return float(per_image_ssim(target, reconstruction).mean().item())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Evaluation refuses a missing checkpoint: {checkpoint_path}")
    device = resolve_device(getattr(args, "device", "auto"))
    (train_loader, val_loader, test_loader), num_classes, metadata = get_dataset_loaders(
        args.dataset,
        getattr(args, "data_dir", "data"),
        int(parse_model_config(getattr(args, "model_config", ""))["batch_size"]),
        getattr(args, "seed", 42),
        num_workers=getattr(args, "num_workers", 0),
        test_only_synthetic=getattr(args, "test_only_synthetic", False),
        max_samples=getattr(args, "max_samples", None),
    )
    model, payload = load_split_model(checkpoint_path, num_classes=num_classes, device=device)
    partition = getattr(args, "partition", "test")
    if partition == "train":
        loader = train_loader
    elif partition == "validation":
        loader = val_loader
    elif partition == "test":
        loader = test_loader
    else:
        raise ValueError(partition)
    if getattr(args, "learned_noise_path", ""):
        defense = load_universal_perturbation(args.learned_noise_path, model.config.feature_shape)
    elif float(getattr(args, "laplace_scale", 0.0)) > 0:
        defense = FeatureDefense("laplace", laplace_scale=float(args.laplace_scale))
    else:
        defense = FeatureDefense(model.config.method)
    utility = classifier_accuracy(
        model,
        loader,
        device,
        defense,
        TEST_DRAW_SEEDS if defense.stochastic else TEST_DRAW_SEEDS[:1],
    )

    # Co-trained decoder output is retained only as a diagnostic, not a primary defense comparison.
    diagnostic = evaluate_adaptive_attacker(
        encoder=model.edge_encoder,
        attacker=model.adversarial_decoder,
        loader=loader,
        defense=defense,
        device=device,
        stochastic_draw_seeds=TEST_DRAW_SEEDS,
    )
    result = {
        "dataset": args.dataset,
        "partition": partition,
        "dataset_metadata": metadata.__dict__,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_schema_version": payload["schema_version"],
        "config": model.config.to_dict(),
        "defense": defense.to_dict(),
        "utility": utility,
        "co_trained_decoder_diagnostic": diagnostic,
        "primary_comparison_eligible": False,
        "reason": "Primary comparisons require adaptive post-hoc attackers from src.pipeline.",
    }
    output_path = getattr(args, "output_path", "")
    if output_path:
        target = Path(output_path)
        revision_root = Path("revisions/2026-07-22_adaptive_reconstruction_study").resolve()
        if revision_root not in target.resolve().parents:
            raise ValueError(f"New revision evaluations must be stored under {revision_root}; got {target.resolve()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a schema-v2 split checkpoint")
    parser.add_argument("--dataset", choices=("cifar10", "celeba"), default="cifar10")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_config", default="")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--laplace-scale", type=float, default=0.0)
    parser.add_argument("--learned-noise-path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--test-only-synthetic", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
