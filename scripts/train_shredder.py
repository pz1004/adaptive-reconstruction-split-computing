#!/usr/bin/env python3
"""Compatibility entry point for the norm-constrained learned perturbation.

The historical unconstrained "Shredder" script is retired. This entry point
uses the revision's global-L2 projection and versioned output policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpointing import load_split_model
from src.dataset import get_selection_loaders
from src.experiment import resolve_device, train_universal_perturbation


REVISION_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "celeba"), required=True)
    parser.add_argument("--checkpoint-path", "--checkpoint_path", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=Path, required=True)
    parser.add_argument("--max-l2", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    target = args.save_path.resolve()
    if not target.is_relative_to(REVISION_ROOT.resolve()):
        raise ValueError(f"New perturbations must be stored under {REVISION_ROOT}")
    if args.max_l2 <= 0:
        raise ValueError("--max-l2 must be positive")
    device = resolve_device(args.device)
    (train_loader, _), num_classes, metadata = get_selection_loaders(
        args.dataset,
        args.data_dir,
        batch_size=128,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model, checkpoint = load_split_model(args.checkpoint_path, num_classes=num_classes, device=device)
    record = train_universal_perturbation(
        model=model,
        train_loader=train_loader,
        device=device,
        max_l2=args.max_l2,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_path=target,
        metadata={
            "stage": "manual_versioned_learned_perturbation",
            "dataset": args.dataset,
            "dataset_metadata": metadata.__dict__,
            "seed": args.seed,
            "source_checkpoint": str(args.checkpoint_path),
            "source_schema_version": checkpoint["config"]["schema_version"],
        },
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
