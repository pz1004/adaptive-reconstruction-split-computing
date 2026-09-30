"""Train and evaluate one adaptive attacker against a frozen encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpointing import load_split_model
from .dataset import get_dataset_loaders
from .defenses import FeatureDefense, load_universal_perturbation
from .experiment import evaluate_adaptive_attacker, resolve_device, train_adaptive_attacker


REVISION_ROOT = Path("revisions/2026-07-22_adaptive_reconstruction_study").resolve()


def _require_revision_path(path: str | Path) -> Path:
    target = Path(path).resolve()
    if REVISION_ROOT not in target.parents:
        raise ValueError(f"New attacker artifacts must be stored under {REVISION_ROOT}; got {target}")
    return target


def main(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    (train_loader, val_loader, test_loader), num_classes, metadata = get_dataset_loaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.seed,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    model, payload = load_split_model(args.checkpoint_path, num_classes=num_classes, device=device)
    if args.learned_noise_path:
        defense = load_universal_perturbation(args.learned_noise_path, model.config.feature_shape)
    elif args.laplace_scale > 0:
        defense = FeatureDefense("laplace", laplace_scale=args.laplace_scale)
    else:
        defense = FeatureDefense(model.config.method)
    attacker_checkpoint = _require_revision_path(args.attacker_checkpoint)
    training_log = _require_revision_path(args.training_log)
    output_path = _require_revision_path(args.output_path)
    training = train_adaptive_attacker(
        encoder=model.edge_encoder,
        feature_shape=model.config.feature_shape,
        train_loader=train_loader,
        val_loader=val_loader,
        defense=defense,
        architecture=args.architecture,
        seed=args.seed,
        device=device,
        checkpoint_path=attacker_checkpoint,
        log_path=training_log,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    attacker_payload = torch.load(attacker_checkpoint, map_location=device, weights_only=True)
    from .models import build_attacker

    attacker = build_attacker(args.architecture, model.config.feature_shape).to(device)
    attacker.load_state_dict(attacker_payload["model_state_dict"])
    evaluation = evaluate_adaptive_attacker(
        encoder=model.edge_encoder,
        attacker=attacker,
        loader=test_loader,
        defense=defense,
        device=device,
    )
    result = {
        "dataset": args.dataset,
        "dataset_metadata": metadata.__dict__,
        "encoder_checkpoint": args.checkpoint_path,
        "encoder_selected_epoch": payload["selected_epoch"],
        "defense": defense.to_dict(),
        "attacker_training": training,
        "test_evaluation": evaluation,
        "primary_comparison_eligible": False,
        "reason": "Standalone compatibility evaluation; primary comparisons require frozen selection and a registered src.pipeline job.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "celeba"), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--architecture", choices=("deconv_mse", "residual_lpips"), required=True)
    parser.add_argument("--laplace-scale", type=float, default=0.0)
    parser.add_argument("--learned-noise-path", default="")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attacker-checkpoint", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(main(build_parser().parse_args()), indent=2))
