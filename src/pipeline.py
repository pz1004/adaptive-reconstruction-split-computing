"""Manifest-driven, resumable execution for the revision study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpointing import load_split_model
from .dataset import get_budget_pool_and_test_loaders, get_dataset_loaders, get_selection_loaders
from .defenses import FeatureDefense, load_universal_perturbation
from .experiment import (
    ATTACKER_ARCHITECTURES,
    TEST_DRAW_SEEDS,
    classifier_accuracy,
    evaluate_adaptive_attacker,
    resolve_device,
    set_seed,
    train_adaptive_attacker,
    train_encoder,
    train_semantic_probe,
    train_universal_perturbation,
)
from .manifest import RunRegistry, deterministic_run_id
from .metrics import METRIC_FIELDS, metric_versions
from .models import AFDSplitModel, ModelConfig
from .selection_provenance import (
    FROZEN_RELATIVE_PATH,
    LOCK_RELATIVE_PATH,
    SelectionProvenanceError,
    load_sealed_selection,
    require_selection_provenance,
    seal_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
DEFAULT_REVISION_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
EXPECTED_BUDGET_RUN_ID_DIGEST = "982aa649ad291d08a426a647ede9d68a4f1741cfe1d9a1e004b9f79cd6105f27"
BUDGET_SUBSET_SEED = 2718
BUDGET_PROTOCOL_VERSION = "auxiliary_budget_total_cap_v2"
BUDGET_STAGE = "auxiliary_budget_v2"
BUDGET_SPLIT_POLICY = "seeded_permutation_prefix_every_tenth_validation_one_based_v2"
BUDGET_ARTIFACT_SCHEMA_VERSION = 2
BUDGET_REGISTRY_DIRECTORY = "budget_v2"


def load_contract(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    contract = json.loads(target.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "datasets",
        "split_points",
        "seeds",
        "methods",
        "attackers",
        "utility_loss_limit_percentage_points",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"Study contract is missing {sorted(missing)}")
    from .implementation_provenance import validate_implementation_protocol

    validate_implementation_protocol(contract, target)
    return contract


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_event(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "stage": "semantic_probe",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _budget_event(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "stage": "auxiliary_budget",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _json_write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(f"Refusing to overwrite existing budget artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def validate_primary_result(result: dict[str, Any]) -> None:
    required = {
        "dataset",
        "split_point",
        "seed",
        "method",
        "utility",
        "attackers",
        "strongest_attacker",
        "validation_selected_attacker",
        "worst_case_test_ssim",
        "metric_versions",
        "evaluation_partition",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"Partial primary result; missing {sorted(missing)}")
    attackers = result["attackers"]
    if set(attackers) != set(ATTACKER_ARCHITECTURES):
        raise ValueError(f"Partial primary result; attacker set is {sorted(attackers)}")
    for architecture in ATTACKER_ARCHITECTURES:
        summary = attackers[architecture].get("test_summary", {})
        missing_metrics = set(METRIC_FIELDS) - set(summary)
        if missing_metrics:
            raise ValueError(f"Partial primary result for {architecture}; missing {sorted(missing_metrics)}")
        if not all(np.isfinite(float(summary[field])) for field in METRIC_FIELDS):
            raise ValueError(f"Non-finite primary metric for {architecture}")
    strongest = max(attackers, key=lambda name: attackers[name]["test_summary"]["ssim"])
    if result["strongest_attacker"] != strongest:
        raise ValueError("Primary strongest_attacker does not match maximum attacker SSIM")
    if not np.isclose(result["worst_case_test_ssim"], attackers[strongest]["test_summary"]["ssim"]):
        raise ValueError("Primary worst_case_test_ssim does not match the strongest attacker")
    if result["validation_selected_attacker"] not in attackers:
        raise ValueError("Validation-selected attacker is not present in the primary attacker set")
    if result["evaluation_partition"] != "test_after_frozen_selection":
        raise ValueError("Primary result has invalid test-selection provenance")


def planned_primary_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for dataset in contract["datasets"]:
        for split_point in contract["split_points"]:
            for seed in contract["seeds"]:
                for method in contract["methods"]:
                    specs.append(
                        {
                            "stage": "primary_defense_evaluation",
                            "dataset": dataset,
                            "split_point": split_point,
                            "seed": seed,
                            "method": method,
                            "attacker_architectures": contract["attackers"]["architectures"],
                            "primary_metric": contract["primary_leakage_metric"],
                            "selection_source": "selection/frozen_operating_points.json",
                            "metrics_directory": f"metrics/{dataset}/{split_point}/seed_{seed}/{method}",
                            "checkpoint_directory": f"checkpoints/primary/{dataset}/{split_point}/seed_{seed}/{method}",
                        }
                    )
    return specs


def planned_semantic_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "semantic_probe",
            "dataset": "celeba",
            "split_point": split_point,
            "seed": seed,
            "method": method,
            "targets": "39_non_smiling_attributes",
            "checkpoint_path": f"checkpoints/semantic/celeba/{split_point}/seed_{seed}/{method}.pt",
        }
        for split_point in contract["split_points"]
        for seed in contract["seeds"]
        for method in contract["methods"]
    ]


def planned_budget_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": BUDGET_STAGE,
            "budget_protocol_version": BUDGET_PROTOCOL_VERSION,
            "budget_split_seed": BUDGET_SUBSET_SEED,
            "budget_split_policy": BUDGET_SPLIT_POLICY,
            "dataset": dataset,
            "split_point": split_point,
            "seed": int(contract["selection_seed"]),
            "method": method,
            "auxiliary_fraction": fraction,
            "attacker_architectures": list(ATTACKER_ARCHITECTURES),
            "output_directory": f"checkpoints/budget_v2/{dataset}/{split_point}/seed_{contract['selection_seed']}/{method}_budget_{float(fraction):g}",
        }
        for dataset in contract["datasets"]
        for split_point in contract["split_points"]
        for method in ("standard", "afd")
        for fraction in contract["auxiliary_data_budgets"]
    ]


def planned_selection_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [("standard", 0.0)] + [
        (method, float(value))
        for method, values in (
            ("afd", contract["afd_alphas"]),
            ("laplace", contract["laplace_scales"]),
            ("learned", contract["learned_l2_limits"]),
        )
        for value in values
    ]
    return [
        {
            "stage": "validation_selection_candidate",
            "dataset": dataset,
            "split_point": split_point,
            "seed": int(contract["selection_seed"]),
            "method": method,
            "candidate_value": value,
            "partition": "validation_only",
            "attacker_architectures": list(ATTACKER_ARCHITECTURES),
            "checkpoint_directory": f"checkpoints/selection/{dataset}/{split_point}/seed_{contract['selection_seed']}/{method}_{value:g}",
            "selection_record": f"selection/frozen_operating_points.json#settings/{dataset}/{split_point}",
        }
        for dataset in contract["datasets"]
        for split_point in contract["split_points"]
        for method, value in candidates
    ]


def planned_encoder_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "primary_encoder",
            "dataset": dataset,
            "split_point": split_point,
            "seed": seed,
            "method": method,
            "selection_source": "selection/frozen_operating_points.json" if method == "afd" else None,
        }
        for dataset in contract["datasets"]
        for split_point in contract["split_points"]
        for seed in contract["seeds"]
        for method in ("standard", "afd")
    ]


def initialize_study(contract_path: str | Path, revision_root: str | Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    root = Path(revision_root)
    root.mkdir(parents=True, exist_ok=True)
    primary_specs = planned_primary_specs(contract)
    encoder_specs = planned_encoder_specs(contract)
    semantic_specs = planned_semantic_specs(contract)
    budget_specs = planned_budget_specs(contract)
    selection_specs = planned_selection_specs(contract)
    study_manifest = {
        "schema_version": 2,
        "study_id": contract["study_id"],
        "contract_path": str(Path(contract_path)),
        "revision_root": str(root),
        "planned_counts": {
            "selection_encoder_checkpoints": 20,
            "selection_defense_candidates": 52,
            "selection_adaptive_attacker_trainings": 104,
            "primary_encoder_checkpoints": len(encoder_specs),
            "primary_defense_evaluations": len(primary_specs),
            "adaptive_attackers_per_defense_evaluation": len(contract["attackers"]["architectures"]),
            "semantic_probe_evaluations": 40,
            "auxiliary_budget_evaluations": 24,
        },
        "primary_encoder_specs": [{"run_id": deterministic_run_id(spec), "spec": spec} for spec in encoder_specs],
        "selection_candidate_specs": [{"run_id": deterministic_run_id(spec), "spec": spec} for spec in selection_specs],
        "primary_defense_specs": [{"run_id": deterministic_run_id(spec), "spec": spec} for spec in primary_specs],
        "semantic_probe_specs": [{"run_id": deterministic_run_id(spec), "spec": spec} for spec in semantic_specs],
        "auxiliary_budget_specs": [{"run_id": deterministic_run_id(spec), "spec": spec} for spec in budget_specs],
        "metric_versions": metric_versions(),
        "status": "planned",
        "submission_ready": False,
    }
    manifest_path = root / "manifests" / "study_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("study_id") != study_manifest["study_id"]:
            raise ValueError(f"Revision directory already belongs to study {existing.get('study_id')}")
        specs_changed = (
            existing.get("primary_encoder_specs") != study_manifest["primary_encoder_specs"]
            or existing.get("primary_defense_specs") != study_manifest["primary_defense_specs"]
        )
        if specs_changed and any((root / "manifests" / "primary").glob("*.json")):
            raise ValueError("Existing study manifest conflicts with registered primary jobs")
        study_manifest["status"] = existing.get("status", study_manifest["status"])
        study_manifest["submission_ready"] = existing.get("submission_ready", False)
        if "selection_provenance" in existing:
            study_manifest["selection_provenance"] = existing["selection_provenance"]
    if (root / LOCK_RELATIVE_PATH).is_file():
        _, selection_provenance, _ = load_sealed_selection(
            contract=contract,
            revision_root=root,
            contract_path=contract_path,
        )
        study_manifest["selection_provenance"] = selection_provenance
    _json_write(manifest_path, study_manifest)
    for directory in (
        root / "manifests" / "encoders",
        root / "manifests" / "primary",
        root / "manifests" / "selection",
        root / "manifests" / "semantic",
        root / "manifests" / BUDGET_REGISTRY_DIRECTORY,
        root / "checkpoints",
        root / "logs",
        root / "metrics",
        root / "selection",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for directory, specs in (
        (root / "manifests" / "primary", primary_specs),
        (root / "manifests" / "semantic", semantic_specs),
        (root / "manifests" / BUDGET_REGISTRY_DIRECTORY, budget_specs),
    ):
        registry = RunRegistry(directory)
        for spec in specs:
            registry.register(spec, resume=True)
    return study_manifest


def initialize_corrected_budget_stage(
    contract: dict[str, Any],
    revision_root: str | Path,
) -> dict[str, Any]:
    """Register the corrected stage without modifying historical budget records."""
    root = Path(revision_root)
    historical_audit = root.parent.parent / "audit" / "budget_result_audit_20260810" / "budget_result_audit.json"
    if not historical_audit.is_file():
        raise RuntimeError(f"Missing historical budget audit to supersede: {historical_audit}")
    historical_payload = json.loads(historical_audit.read_text(encoding="utf-8"))
    if historical_payload.get("verification_counts", {}).get("budget_artifacts_hashed") != 216:
        raise RuntimeError("Historical budget preservation boundary is not the expected 216 artifacts")
    specs = planned_budget_specs(contract)
    run_ids = [deterministic_run_id(spec) for spec in specs]
    stage_manifest = {
        "schema_version": 2,
        "stage": BUDGET_STAGE,
        "protocol_version": BUDGET_PROTOCOL_VERSION,
        "status": "planned",
        "expected_cells": 24,
        "expected_attacker_rows": 48,
        "expected_artifacts": 216,
        "run_id_digest": _budget_run_id_digest(run_ids),
        "historical_evidence": {
            "status": "superseded_for_manuscript_claims_preserved_as_historical_evidence",
            "audit_path": str(historical_audit),
            "audit_sha256": _sha256_file(historical_audit),
            "artifact_count": 216,
        },
        "specs": [{"run_id": deterministic_run_id(spec), "spec": spec} for spec in specs],
    }
    manifest_path = root / "manifests" / "auxiliary_budget_v2_stage.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != stage_manifest:
            raise RuntimeError(f"Corrected budget stage manifest conflicts with {manifest_path}")
    else:
        _json_write_exclusive(manifest_path, stage_manifest)
    registry = RunRegistry(root / "manifests" / BUDGET_REGISTRY_DIRECTORY)
    for spec in specs:
        registry.register(spec, resume=True)
    return stage_manifest


def _config(contract: dict[str, Any], split_point: str, method: str, alpha: float = 0.0) -> ModelConfig:
    feature_shape = tuple(contract["split_points"][split_point])
    return ModelConfig.from_mapping(
        {
            "split_point": split_point,
            "method": method,
            "alpha": alpha,
            "bottleneck_channels": feature_shape[0],
            "feature_shape": feature_shape,
            "lr": contract["encoder"]["learning_rate"],
            "batch_size": contract["encoder"]["batch_size"],
        }
    )


def _encoder_path(root: Path, dataset: str, split_point: str, seed: int, method: str) -> Path:
    return root / "checkpoints" / "encoders" / dataset / split_point / f"seed_{seed}" / f"{method}.pt"


def _encoder_log_path(root: Path, dataset: str, split_point: str, seed: int, method: str) -> Path:
    return root / "logs" / "encoders" / dataset / split_point / f"seed_{seed}" / f"{method}.json"


def _selection_encoder_path(root: Path, dataset: str, split_point: str, alpha: float) -> Path:
    return root / "checkpoints" / "selection" / dataset / split_point / f"afd_alpha_{alpha:g}.pt"


def _selection_standard_path(root: Path, dataset: str, split_point: str) -> Path:
    return root / "checkpoints" / "selection" / dataset / split_point / "standard.pt"


def _learned_path(root: Path, stage: str, dataset: str, split_point: str, seed: int, max_l2: float) -> Path:
    return root / "checkpoints" / stage / dataset / split_point / f"seed_{seed}" / f"learned_l2_{max_l2:g}.pt"


def _attacker_paths(
    root: Path,
    stage: str,
    dataset: str,
    split_point: str,
    seed: int,
    method_label: str,
    architecture: str,
) -> tuple[Path, Path]:
    base = Path(stage) / dataset / split_point / f"seed_{seed}" / method_label
    return root / "checkpoints" / base / f"attacker_{architecture}.pt", root / "logs" / base / f"attacker_{architecture}.json"


def _defense(method: str, value: float | None, learned_path: Path | None, feature_shape: tuple[int, int, int]) -> FeatureDefense:
    if method in {"standard", "afd"}:
        return FeatureDefense(method)
    if method == "laplace":
        if value is None:
            raise ValueError("Laplace defense requires a selected scale")
        return FeatureDefense("laplace", laplace_scale=float(value))
    if method == "learned":
        if learned_path is None:
            raise ValueError("Learned defense requires a perturbation checkpoint")
        return load_universal_perturbation(learned_path, feature_shape)
    raise ValueError(method)


def _selection_candidates(contract: dict[str, Any]) -> dict[str, list[float]]:
    return {
        "afd": list(map(float, contract["afd_alphas"])),
        "laplace": list(map(float, contract["laplace_scales"])),
        "learned": list(map(float, contract["learned_l2_limits"])),
    }


def _evaluate_candidate(
    *,
    contract: dict[str, Any],
    root: Path,
    dataset: str,
    split_point: str,
    method: str,
    value: float,
    model: AFDSplitModel,
    train_loader: Iterable,
    val_loader: Iterable,
    defense: FeatureDefense,
    seed: int,
    device: torch.device,
    attacker_max_epochs: int,
    attacker_patience: int,
) -> dict[str, Any]:
    label = f"{method}_{value:g}"
    utility = classifier_accuracy(
        model,
        val_loader,
        device,
        defense,
        TEST_DRAW_SEEDS if defense.stochastic else TEST_DRAW_SEEDS[:1],
    )
    attacker_records: dict[str, Any] = {}
    for architecture in ATTACKER_ARCHITECTURES:
        checkpoint_path, log_path = _attacker_paths(
            root, "selection", dataset, split_point, seed, label, architecture
        )
        training = train_adaptive_attacker(
            encoder=model.edge_encoder,
            feature_shape=model.config.feature_shape,
            train_loader=train_loader,
            val_loader=val_loader,
            defense=defense,
            architecture=architecture,
            seed=seed,
            device=device,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            max_epochs=attacker_max_epochs,
            patience=attacker_patience,
            learning_rate=contract["attackers"]["learning_rate"],
        )
        attacker_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        attacker = __import__("src.models", fromlist=["build_attacker"]).build_attacker(
            architecture, model.config.feature_shape
        ).to(device)
        attacker.load_state_dict(attacker_payload["model_state_dict"])
        metrics = evaluate_adaptive_attacker(
            encoder=model.edge_encoder,
            attacker=attacker,
            loader=val_loader,
            defense=defense,
            device=device,
        )
        attacker_records[architecture] = {"training": training, "validation_metrics": metrics}
    worst_ssim = max(record["validation_metrics"]["summary"]["ssim"] for record in attacker_records.values())
    strongest = max(
        attacker_records,
        key=lambda name: attacker_records[name]["validation_metrics"]["summary"]["ssim"],
    )
    return {
        "dataset": dataset,
        "split_point": split_point,
        "seed": seed,
        "method": method,
        "candidate_value": value,
        "utility": utility,
        "defense": defense.to_dict(),
        "attackers": attacker_records,
        "worst_case_validation_ssim": worst_ssim,
        "strongest_validation_attacker": strongest,
        "evaluation_partition": "validation",
        "test_loader_constructed": False,
    }


def run_selection(
    *,
    contract: dict[str, Any],
    revision_root: str | Path,
    data_dir: str,
    device_name: str,
    num_workers: int,
    attacker_max_epochs: int | None = None,
    attacker_patience: int | None = None,
    max_samples: int | None = None,
    freeze: bool = True,
) -> dict[str, Any]:
    root = Path(revision_root)
    if freeze:
        existing_selection_files = [
            path for path in (root / FROZEN_RELATIVE_PATH, root / LOCK_RELATIVE_PATH) if path.exists()
        ]
        if existing_selection_files:
            rendered = ", ".join(str(path) for path in existing_selection_files)
            raise RuntimeError(f"Production selection is immutable and already exists: {rendered}")
    if freeze and (max_samples is not None or attacker_max_epochs is not None or attacker_patience is not None):
        raise ValueError("Reduced smoke settings cannot freeze production operating points")
    device = resolve_device(device_name)
    attacker_max_epochs = attacker_max_epochs or int(contract["attackers"]["max_epochs"])
    attacker_patience = attacker_patience or int(contract["attackers"]["patience"])
    seed = int(contract["selection_seed"])
    frozen: dict[str, Any] = {
        "schema_version": 2,
        "selection_seed": seed,
        "partition": "validation_only",
        "test_access": False,
        "utility_loss_limit_percentage_points": contract["utility_loss_limit_percentage_points"],
        "settings": {},
    }
    for dataset in contract["datasets"]:
        frozen["settings"][dataset] = {}
        for split_point in contract["split_points"]:
            (train_loader, val_loader), num_classes, dataset_metadata = get_selection_loaders(
                dataset,
                data_dir,
                contract["encoder"]["batch_size"],
                seed,
                num_workers=num_workers,
                max_samples=max_samples,
            )
            standard_config = _config(contract, split_point, "standard", 0.0)
            standard_path = _selection_standard_path(root, dataset, split_point)
            if not standard_path.exists():
                train_encoder(
                    dataset=dataset,
                    data_dir=data_dir,
                    config=standard_config,
                    seed=seed,
                    epochs=contract["encoder"]["epochs"],
                    checkpoint_path=standard_path,
                    log_path=root / "logs" / "selection" / dataset / split_point / "standard_encoder.json",
                    device_name=device_name,
                    num_workers=num_workers,
                    max_samples=max_samples,
                    selection_stage=True,
                )
            standard_model, _ = load_split_model(standard_path, num_classes=num_classes, device=device)
            standard_utility = classifier_accuracy(standard_model, val_loader, device, FeatureDefense("standard"))
            standard_adaptive = _evaluate_candidate(
                contract=contract,
                root=root,
                dataset=dataset,
                split_point=split_point,
                method="standard",
                value=0.0,
                model=standard_model,
                train_loader=train_loader,
                val_loader=val_loader,
                defense=FeatureDefense("standard"),
                seed=seed,
                device=device,
                attacker_max_epochs=attacker_max_epochs,
                attacker_patience=attacker_patience,
            )
            setting = {
                "dataset_metadata": asdict(dataset_metadata),
                "standard_validation_accuracy": standard_utility["accuracy"],
                "standard_adaptive_evaluation": standard_adaptive,
                "methods": {},
            }
            for method, values in _selection_candidates(contract).items():
                candidates: list[dict[str, Any]] = []
                for value in values:
                    learned_path = None
                    if method == "afd":
                        encoder_path = _selection_encoder_path(root, dataset, split_point, value)
                        if not encoder_path.exists():
                            train_encoder(
                                dataset=dataset,
                                data_dir=data_dir,
                                config=_config(contract, split_point, "afd", value),
                                seed=seed,
                                epochs=contract["encoder"]["epochs"],
                                checkpoint_path=encoder_path,
                                log_path=root / "logs" / "selection" / dataset / split_point / f"afd_alpha_{value:g}_encoder.json",
                                device_name=device_name,
                                num_workers=num_workers,
                                max_samples=max_samples,
                                selection_stage=True,
                            )
                        model, _ = load_split_model(encoder_path, num_classes=num_classes, device=device)
                        defense = FeatureDefense("afd")
                    else:
                        model = standard_model
                        if method == "laplace":
                            defense = FeatureDefense("laplace", laplace_scale=value)
                        else:
                            learned_path = _learned_path(root, "selection", dataset, split_point, seed, value)
                            if not learned_path.exists():
                                train_universal_perturbation(
                                    model=standard_model,
                                    train_loader=train_loader,
                                    device=device,
                                    max_l2=value,
                                    epochs=contract["learned_perturbation"]["epochs"],
                                    learning_rate=contract["learned_perturbation"]["learning_rate"],
                                    output_path=learned_path,
                                    metadata={"stage": "selection", "dataset": dataset, "split_point": split_point, "seed": seed},
                                )
                            defense = load_universal_perturbation(learned_path, model.config.feature_shape)
                    candidates.append(
                        _evaluate_candidate(
                            contract=contract,
                            root=root,
                            dataset=dataset,
                            split_point=split_point,
                            method=method,
                            value=value,
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            defense=defense,
                            seed=seed,
                            device=device,
                            attacker_max_epochs=attacker_max_epochs,
                            attacker_patience=attacker_patience,
                        )
                    )
                utility_floor = standard_utility["accuracy"] - contract["utility_loss_limit_percentage_points"] / 100.0
                compatible = [candidate for candidate in candidates if candidate["utility"]["accuracy"] >= utility_floor]
                if compatible:
                    selected = min(compatible, key=lambda candidate: candidate["worst_case_validation_ssim"])
                    status = "utility_compatible"
                    utility_wording_allowed = True
                else:
                    selected = max(candidates, key=lambda candidate: candidate["utility"]["accuracy"])
                    status = "exploratory_no_utility_compatible_candidate"
                    utility_wording_allowed = False
                setting["methods"][method] = {
                    "status": status,
                    "utility_compatible_wording_allowed": utility_wording_allowed,
                    "selected_value": selected["candidate_value"],
                    "selected_worst_case_validation_ssim": selected["worst_case_validation_ssim"],
                    "selected_strongest_attacker": selected["strongest_validation_attacker"],
                    "candidates": candidates,
                }
            frozen["settings"][dataset][split_point] = setting
    output = root / "selection" / ("frozen_operating_points.json" if freeze else "smoke_operating_points.json")
    _json_write(output, frozen)
    return frozen


def _selected(frozen: dict[str, Any], dataset: str, split_point: str, method: str) -> float:
    return float(frozen["settings"][dataset][split_point]["methods"][method]["selected_value"])


def _load_or_train_primary_encoder(
    *,
    contract: dict[str, Any],
    root: Path,
    frozen: dict[str, Any],
    dataset: str,
    split_point: str,
    seed: int,
    method: str,
    data_dir: str,
    device_name: str,
    num_workers: int,
) -> Path:
    path = _encoder_path(root, dataset, split_point, seed, method)
    if path.exists():
        return path
    alpha = _selected(frozen, dataset, split_point, "afd") if method == "afd" else 0.0
    train_encoder(
        dataset=dataset,
        data_dir=data_dir,
        config=_config(contract, split_point, method, alpha),
        seed=seed,
        epochs=contract["encoder"]["epochs"],
        checkpoint_path=path,
        log_path=_encoder_log_path(root, dataset, split_point, seed, method),
        device_name=device_name,
        num_workers=num_workers,
    )
    return path


def run_primary_job(
    *,
    contract: dict[str, Any],
    revision_root: str | Path,
    frozen: dict[str, Any],
    selection_provenance: dict[str, str],
    spec: dict[str, Any],
    data_dir: str,
    device_name: str,
    num_workers: int,
) -> dict[str, Any]:
    root = Path(revision_root)
    registry = RunRegistry(root / "manifests" / "primary")
    existing = registry.register(spec, resume=True)
    if existing["status"] == "complete":
        require_selection_provenance(existing, selection_provenance)
        return existing
    run_id = existing["run_id"]
    registry.update(run_id, "running", selection_provenance=selection_provenance)
    try:
        dataset = spec["dataset"]
        split_point = spec["split_point"]
        seed = int(spec["seed"])
        method = spec["method"]
        device = resolve_device(device_name)
        (train_loader, val_loader, test_loader), num_classes, metadata = get_dataset_loaders(
            dataset,
            data_dir,
            contract["encoder"]["batch_size"],
            seed,
            num_workers=num_workers,
            include_attributes=False,
        )
        source_method = "afd" if method == "afd" else "standard"
        encoder_path = _load_or_train_primary_encoder(
            contract=contract,
            root=root,
            frozen=frozen,
            dataset=dataset,
            split_point=split_point,
            seed=seed,
            method=source_method,
            data_dir=data_dir,
            device_name=device_name,
            num_workers=num_workers,
        )
        model, encoder_payload = load_split_model(encoder_path, num_classes=num_classes, device=device)
        learned_path = None
        selected_value = None
        if method == "laplace":
            selected_value = _selected(frozen, dataset, split_point, "laplace")
        elif method == "learned":
            selected_value = _selected(frozen, dataset, split_point, "learned")
            learned_path = _learned_path(root, "primary", dataset, split_point, seed, selected_value)
            if not learned_path.exists():
                train_universal_perturbation(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    max_l2=selected_value,
                    epochs=contract["learned_perturbation"]["epochs"],
                    learning_rate=contract["learned_perturbation"]["learning_rate"],
                    output_path=learned_path,
                    metadata={"stage": "primary", "dataset": dataset, "split_point": split_point, "seed": seed},
                )
        elif method == "afd":
            selected_value = _selected(frozen, dataset, split_point, "afd")
        defense = _defense(method, selected_value, learned_path, model.config.feature_shape)
        utility = classifier_accuracy(
            model,
            test_loader,
            device,
            defense,
            TEST_DRAW_SEEDS if defense.stochastic else TEST_DRAW_SEEDS[:1],
        )
        attackers: dict[str, Any] = {}
        for architecture in ATTACKER_ARCHITECTURES:
            checkpoint_path, log_path = _attacker_paths(
                root, "primary", dataset, split_point, seed, method, architecture
            )
            training = train_adaptive_attacker(
                encoder=model.edge_encoder,
                feature_shape=model.config.feature_shape,
                train_loader=train_loader,
                val_loader=val_loader,
                defense=defense,
                architecture=architecture,
                seed=seed,
                device=device,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
                max_epochs=contract["attackers"]["max_epochs"],
                patience=contract["attackers"]["patience"],
                learning_rate=contract["attackers"]["learning_rate"],
            )
            attacker_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
            from .models import build_attacker

            attacker = build_attacker(architecture, model.config.feature_shape).to(device)
            attacker.load_state_dict(attacker_payload["model_state_dict"])
            metrics_path = root / "metrics" / dataset / split_point / f"seed_{seed}" / method / f"{architecture}.json"
            sample_path = root / "metrics" / dataset / split_point / f"seed_{seed}" / method / f"{architecture}_samples.pt"
            metrics = evaluate_adaptive_attacker(
                encoder=model.edge_encoder,
                attacker=attacker,
                loader=test_loader,
                defense=defense,
                device=device,
                sample_bundle_path=sample_path,
            )
            _json_write(metrics_path, metrics)
            attackers[architecture] = {
                "training": training,
                "metrics_path": str(metrics_path),
                "test_summary": metrics["summary"],
                "sample_bundle_path": str(sample_path),
            }
        strongest = max(attackers, key=lambda name: attackers[name]["test_summary"]["ssim"])
        selection_setting = frozen["settings"][dataset][split_point]
        validation_selected_attacker = (
            selection_setting["standard_adaptive_evaluation"]["strongest_validation_attacker"]
            if method == "standard"
            else selection_setting["methods"][method]["selected_strongest_attacker"]
        )
        result = {
            "dataset": dataset,
            "dataset_metadata": asdict(metadata),
            "split_point": split_point,
            "seed": seed,
            "method": method,
            "selected_defense_value": selected_value,
            "defense": defense.to_dict(),
            "encoder_checkpoint": str(encoder_path),
            "encoder_selected_epoch": encoder_payload["selected_epoch"],
            "encoder_hyperparameters": model.config.to_dict(),
            "utility": utility,
            "attackers": attackers,
            "strongest_attacker": strongest,
            "validation_selected_attacker": validation_selected_attacker,
            "worst_case_test_ssim": attackers[strongest]["test_summary"]["ssim"],
            "metric_versions": metric_versions(),
            "evaluation_partition": "test_after_frozen_selection",
        }
        validate_primary_result(result)
        return registry.update(
            run_id,
            "complete",
            selection_provenance=selection_provenance,
            result=result,
        )
    except Exception as error:
        registry.update(run_id, "failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise


def run_primary(
    *,
    contract: dict[str, Any],
    revision_root: str | Path,
    data_dir: str,
    device_name: str,
    num_workers: int,
    only_run_id: str | None = None,
) -> None:
    root = Path(revision_root)
    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract,
        revision_root=root,
    )
    for spec in planned_primary_specs(contract):
        if only_run_id is not None and deterministic_run_id(spec) != only_run_id:
            continue
        run_primary_job(
            contract=contract,
            revision_root=root,
            frozen=frozen,
            selection_provenance=selection_provenance,
            spec=spec,
            data_dir=data_dir,
            device_name=device_name,
            num_workers=num_workers,
        )


def _primary_record(
    root: Path,
    contract: dict[str, Any],
    dataset: str,
    split_point: str,
    seed: int,
    method: str,
    selection_provenance: dict[str, str],
) -> dict[str, Any]:
    spec = next(
        spec
        for spec in planned_primary_specs(contract)
        if spec["dataset"] == dataset
        and spec["split_point"] == split_point
        and spec["seed"] == seed
        and spec["method"] == method
    )
    path = root / "manifests" / "primary" / f"{deterministic_run_id(spec)}.json"
    if not path.is_file():
        raise RuntimeError(f"Required primary job is not registered: {deterministic_run_id(spec)}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] != "complete":
        raise RuntimeError(f"Required primary job is not complete: {record['run_id']}")
    require_selection_provenance(record, selection_provenance)
    return record


def run_semantic_probes(
    *,
    contract: dict[str, Any],
    revision_root: str | Path,
    data_dir: str,
    device_name: str,
    num_workers: int,
    only_run_id: str | None = None,
) -> None:
    root = Path(revision_root)
    expected_specs = planned_semantic_specs(contract)
    expected_by_id = {deterministic_run_id(spec): spec for spec in expected_specs}
    registry = RunRegistry(root / "manifests" / "semantic")
    records_by_id = {record["run_id"]: record for record in registry.records()}
    if set(records_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(records_by_id))
        unexpected = sorted(set(records_by_id) - set(expected_by_id))
        raise RuntimeError(f"Semantic registry identity mismatch; missing={missing}, unexpected={unexpected}")
    for run_id, record in records_by_id.items():
        if record.get("spec") != expected_by_id[run_id] or deterministic_run_id(record.get("spec", {})) != run_id:
            raise RuntimeError(f"Semantic registry configuration changed: {run_id}")
    if only_run_id is not None and only_run_id not in expected_by_id:
        raise ValueError(f"Unknown semantic --only-run-id: {only_run_id}")

    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract,
        revision_root=root,
    )
    device = resolve_device(device_name)
    selected_specs = expected_specs if only_run_id is None else [expected_by_id[only_run_id]]
    for spec in selected_specs:
        split_point = str(spec["split_point"])
        seed = int(spec["seed"])
        method = str(spec["method"])
        record = registry.register(spec, resume=True)
        if record["status"] == "complete":
            require_selection_provenance(record, selection_provenance)
            _semantic_event("semantic_job_immutable_skip", run_id=record["run_id"])
            continue
        if only_run_id is None and record["status"] in {"failed", "running"}:
            raise RuntimeError(
                f"Semantic record {record['run_id']} is {record['status']}; "
                "preserve partial evidence and recover only with --only-run-id"
            )
        run_id = record["run_id"]
        checkpoint = (
            root
            / "checkpoints"
            / "semantic"
            / "celeba"
            / split_point
            / f"seed_{seed}"
            / f"{method}.pt"
        )
        raw_bundle = checkpoint.with_suffix(".predictions.npz")
        existing_artifacts = [str(path) for path in (checkpoint, raw_bundle) if path.exists()]
        if existing_artifacts:
            raise RuntimeError(
                f"Semantic recovery requires quarantining existing partial/conflicting artifacts "
                f"for {run_id}: {existing_artifacts}"
            )
        registry.update(run_id, "running", selection_provenance=selection_provenance)
        complete_before = sum(item.get("status") == "complete" for item in registry.records())
        _semantic_event(
            "semantic_job_start",
            run_id=run_id,
            split_point=split_point,
            seed=seed,
            method=method,
            completed=complete_before,
            expected=40,
            progress=f"{complete_before}/40",
        )
        try:
            primary = _primary_record(
                root,
                contract,
                "celeba",
                split_point,
                seed,
                method,
                selection_provenance,
            )["result"]
            loaders, num_classes, metadata = get_dataset_loaders(
                "celeba",
                data_dir,
                contract["encoder"]["batch_size"],
                seed,
                num_workers=num_workers,
                include_attributes=True,
            )
            model, _ = load_split_model(
                primary["encoder_checkpoint"], num_classes=num_classes, device=device
            )
            learned_path = None
            if method == "learned":
                selected_l2 = _selected(frozen, "celeba", split_point, "learned")
                learned_path = _learned_path(
                    root, "primary", "celeba", split_point, seed, selected_l2
                )
            defense = _defense(
                method,
                primary["selected_defense_value"],
                learned_path,
                model.config.feature_shape,
            )
            encoder_path = Path(primary["encoder_checkpoint"])
            primary_checkpoint_hashes: dict[str, dict[str, str]] = {
                "encoder": {"path": str(encoder_path), "sha256": _sha256_file(encoder_path)}
            }
            if learned_path is not None:
                primary_checkpoint_hashes["learned_perturbation"] = {
                    "path": str(learned_path),
                    "sha256": _sha256_file(learned_path),
                }
            result = train_semantic_probe(
                model=model,
                defense=defense,
                loaders=loaders,
                device=device,
                seed=seed,
                checkpoint_path=checkpoint,
                raw_bundle_path=raw_bundle,
                artifact_metadata={
                    "dataset": "celeba",
                    "dataset_metadata": asdict(metadata),
                    "split_point": split_point,
                    "seed": seed,
                    "method": method,
                    "selected_defense_value": primary["selected_defense_value"],
                    "defense": defense.to_dict(),
                    "selection_provenance": selection_provenance,
                    "primary_checkpoint_hashes": primary_checkpoint_hashes,
                },
                max_epochs=contract["attackers"]["max_epochs"],
                patience=contract["attackers"]["patience"],
            )
            registry.update(
                run_id,
                "complete",
                selection_provenance=selection_provenance,
                result={
                    "dataset": "celeba",
                    "dataset_metadata": asdict(metadata),
                    "split_point": split_point,
                    "seed": seed,
                    "method": method,
                    "selected_defense_value": primary["selected_defense_value"],
                    "defense": defense.to_dict(),
                    "primary_checkpoint_hashes": primary_checkpoint_hashes,
                    **result,
                },
            )
            complete_after = sum(item.get("status") == "complete" for item in registry.records())
            _semantic_event(
                "semantic_job_complete",
                run_id=run_id,
                completed=complete_after,
                expected=40,
                progress=f"{complete_after}/40",
            )
        except Exception as error:
            registry.update(
                run_id,
                "failed",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            complete_now = sum(item.get("status") == "complete" for item in registry.records())
            _semantic_event(
                "semantic_job_failure",
                run_id=run_id,
                error_type=type(error).__name__,
                error_message=str(error),
                completed=complete_now,
                expected=40,
                progress=f"{complete_now}/40",
            )
            raise


def budget_subset_identity(dataset_size: int, fraction: float, seed: int = BUDGET_SUBSET_SEED) -> dict[str, Any]:
    if dataset_size <= 0 or not 0 < fraction <= 1:
        raise ValueError(f"Invalid budget subset request: size={dataset_size}, fraction={fraction}")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    count = max(1, int(round(dataset_size * fraction)))
    if count < 10:
        raise ValueError("Corrected budget caps require at least ten pairs for internal validation")
    indices = permutation[:count]
    validation_indices = indices[9::10]
    fitting_indices = [index for position, index in enumerate(indices, start=1) if position % 10]

    def digest(values: list[int]) -> str:
        return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()

    return {
        "dataset_size": dataset_size,
        "auxiliary_fraction": float(fraction),
        "auxiliary_examples": count,
        "auxiliary_fitting_examples": len(fitting_indices),
        "auxiliary_validation_examples": len(validation_indices),
        "auxiliary_index_seed": seed,
        "auxiliary_split_policy": BUDGET_SPLIT_POLICY,
        "auxiliary_index_digest": digest(indices),
        "auxiliary_fitting_index_digest": digest(fitting_indices),
        "auxiliary_validation_index_digest": digest(validation_indices),
        "auxiliary_index_digest_algorithm": "sha256-int64le-v1",
        "ordinary_validation_access": False,
        "indices": indices,
        "fitting_indices": fitting_indices,
        "validation_indices": validation_indices,
    }


def _budget_loaders(
    fitting_pool_loader: DataLoader,
    validation_pool_loader: DataLoader,
    fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    if len(fitting_pool_loader.dataset) != len(validation_pool_loader.dataset):
        raise RuntimeError("Budget fitting and validation views do not have identical membership")
    identity = budget_subset_identity(len(fitting_pool_loader.dataset), fraction, seed)
    fitting_subset = Subset(fitting_pool_loader.dataset, identity["fitting_indices"])
    validation_subset = Subset(
        validation_pool_loader.dataset,
        identity["validation_indices"],
    )
    common = {
        "batch_size": fitting_pool_loader.batch_size,
        "num_workers": fitting_pool_loader.num_workers,
        "pin_memory": fitting_pool_loader.pin_memory,
    }
    fitting_loader = DataLoader(
        fitting_subset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(validation_subset, shuffle=False, **common)
    public_identity = {
        key: value
        for key, value in identity.items()
        if key not in {"indices", "fitting_indices", "validation_indices"}
    }
    return fitting_loader, validation_loader, public_identity


def budget_artifact_paths(root: Path, spec: dict[str, Any], architecture: str) -> dict[str, Path]:
    fraction = float(spec["auxiliary_fraction"])
    label = f"{spec['method']}_budget_{fraction:g}"
    checkpoint, training_log = _attacker_paths(
        root,
        BUDGET_REGISTRY_DIRECTORY,
        str(spec["dataset"]),
        str(spec["split_point"]),
        int(spec["seed"]),
        label,
        architecture,
    )
    metric_base = (
        root
        / "metrics"
        / BUDGET_REGISTRY_DIRECTORY
        / str(spec["dataset"])
        / str(spec["split_point"])
        / f"seed_{spec['seed']}"
        / label
    )
    return {
        "checkpoint": checkpoint,
        "training_log": training_log,
        "metric_json": metric_base / f"{architecture}.json",
        "sample_bundle": metric_base / f"{architecture}_samples.pt",
    }


def _budget_run_id_digest(run_ids: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(run_ids)) + "\n").encode("utf-8")).hexdigest()


def run_auxiliary_budget_sweep(
    *,
    contract: dict[str, Any],
    revision_root: str | Path,
    data_dir: str,
    device_name: str,
    num_workers: int,
    only_run_id: str | None = None,
) -> None:
    root = Path(revision_root)
    from .semantic_result_audit import require_canonical_semantic_audit

    require_canonical_semantic_audit(
        contract,
        root,
        root.parent.parent / "audit" / "semantic_result_audit_20260809" / "semantic_result_audit.json",
    )
    expected_specs = planned_budget_specs(contract)
    expected_by_id = {deterministic_run_id(spec): spec for spec in expected_specs}
    if len(expected_by_id) != 24 or _budget_run_id_digest(expected_by_id) != EXPECTED_BUDGET_RUN_ID_DIGEST:
        raise RuntimeError("Budget contract no longer defines the frozen 24-cell run-ID matrix")
    registry = RunRegistry(root / "manifests" / BUDGET_REGISTRY_DIRECTORY)
    records_by_id = {record["run_id"]: record for record in registry.records()}
    if set(records_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(records_by_id))
        unexpected = sorted(set(records_by_id) - set(expected_by_id))
        raise RuntimeError(f"Budget registry identity mismatch; missing={missing}, unexpected={unexpected}")
    for run_id, record in records_by_id.items():
        if record.get("spec") != expected_by_id[run_id] or deterministic_run_id(record.get("spec", {})) != run_id:
            raise RuntimeError(f"Budget registry configuration changed: {run_id}")
    if only_run_id is not None and only_run_id not in expected_by_id:
        raise ValueError(f"Unknown budget --only-run-id: {only_run_id}")
    if only_run_id is None:
        unsafe = sorted(
            (run_id, record.get("status"))
            for run_id, record in records_by_id.items()
            if record.get("status") in {"running", "failed"}
        )
        if unsafe:
            raise RuntimeError(
                f"Budget bulk execution refuses running/failed records {unsafe}; "
                "preserve evidence and recover exactly one cell with --only-run-id"
            )
    selected_specs = expected_specs if only_run_id is None else [expected_by_id[only_run_id]]
    for spec in selected_specs:
        record = records_by_id[deterministic_run_id(spec)]
        if record.get("status") == "complete":
            continue
        collisions = [
            str(path)
            for architecture in ATTACKER_ARCHITECTURES
            for path in budget_artifact_paths(root, spec, architecture).values()
            if path.exists()
        ]
        if collisions:
            raise RuntimeError(
                f"Budget recovery requires quarantining existing partial/conflicting artifacts "
                f"for {record['run_id']}: {collisions}"
            )

    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract,
        revision_root=root,
    )
    for spec in selected_specs:
        record = records_by_id[deterministic_run_id(spec)]
        if record.get("status") == "complete":
            require_selection_provenance(record, selection_provenance)
            _budget_event("budget_job_immutable_skip", run_id=record["run_id"])
    if all(records_by_id[deterministic_run_id(spec)].get("status") == "complete" for spec in selected_specs):
        return
    device = resolve_device(device_name)
    loaders_by_dataset: dict[str, tuple[tuple[DataLoader, DataLoader, DataLoader], int, Any]] = {}
    current_base: tuple[str, str, str] | None = None
    primary: dict[str, Any]
    model: AFDSplitModel
    defense: FeatureDefense
    for spec in selected_specs:
        dataset = str(spec["dataset"])
        split_point = str(spec["split_point"])
        method = str(spec["method"])
        fraction = float(spec["auxiliary_fraction"])
        seed = int(spec["seed"])
        record = registry.register(spec, resume=True)
        if record["status"] == "complete":
            continue
        base = (dataset, split_point, method)
        if dataset not in loaders_by_dataset:
            loaders_by_dataset[dataset] = get_budget_pool_and_test_loaders(
                dataset,
                data_dir,
                contract["encoder"]["batch_size"],
                seed,
                num_workers=num_workers,
            )
        (
            fitting_pool_loader,
            validation_pool_loader,
            test_loader,
        ), num_classes, metadata = loaders_by_dataset[dataset]
        if base != current_base:
            primary = _primary_record(
                root, contract, dataset, split_point, seed, method, selection_provenance
            )["result"]
            model, _ = load_split_model(
                primary["encoder_checkpoint"], num_classes=num_classes, device=device
            )
            defense = FeatureDefense(method)
            current_base = base
        budget_train, budget_validation, subset_identity = _budget_loaders(
            fitting_pool_loader,
            validation_pool_loader,
            fraction,
            BUDGET_SUBSET_SEED,
        )
        encoder_path = Path(primary["encoder_checkpoint"])
        encoder_identity = {"path": str(encoder_path), "sha256": _sha256_file(encoder_path)}
        selected_defense_value = primary.get("selected_defense_value")
        common_identity = {
            "budget_artifact_schema_version": BUDGET_ARTIFACT_SCHEMA_VERSION,
            "budget_protocol_version": BUDGET_PROTOCOL_VERSION,
            "run_id": record["run_id"],
            "dataset": dataset,
            "split_point": split_point,
            "seed": seed,
            "method": method,
            "selected_defense_value": selected_defense_value,
            "defense": defense.to_dict(),
            "auxiliary_fraction": fraction,
            **subset_identity,
            "primary_encoder": encoder_identity,
            "selection_provenance": selection_provenance,
            "evaluation_partition": "test_after_frozen_selection",
            "metric_versions": metric_versions(),
            "model_seeds_are_independent_replicates": False,
            "attackers_are_independent_replicates": False,
            "images_are_independent_replicates": False,
            "auxiliary_examples_are_independent_replicates": False,
        }
        run_id = record["run_id"]
        registry.update(run_id, "running", selection_provenance=selection_provenance)
        complete_before = sum(item.get("status") == "complete" for item in registry.records())
        _budget_event(
            "budget_job_start",
            **common_identity,
            completed=complete_before,
            expected=24,
            progress=f"{complete_before}/24",
        )
        try:
            attackers: dict[str, Any] = {}
            for architecture in ATTACKER_ARCHITECTURES:
                paths = budget_artifact_paths(root, spec, architecture)
                artifact_identity = {
                    **common_identity,
                    "architecture": architecture,
                }
                training = train_adaptive_attacker(
                    encoder=model.edge_encoder,
                    feature_shape=model.config.feature_shape,
                    train_loader=budget_train,
                    val_loader=budget_validation,
                    defense=defense,
                    architecture=architecture,
                    seed=seed,
                    device=device,
                    checkpoint_path=paths["checkpoint"],
                    log_path=paths["training_log"],
                    max_epochs=contract["attackers"]["max_epochs"],
                    patience=contract["attackers"]["patience"],
                    learning_rate=contract["attackers"]["learning_rate"],
                    artifact_metadata={"artifact_type": "budget_attacker_checkpoint", **artifact_identity},
                    exclusive=True,
                )
                payload = torch.load(paths["checkpoint"], map_location=device, weights_only=True)
                from .models import build_attacker

                attacker = build_attacker(architecture, model.config.feature_shape).to(device)
                attacker.load_state_dict(payload["model_state_dict"], strict=True)
                metrics = evaluate_adaptive_attacker(
                    encoder=model.edge_encoder,
                    attacker=attacker,
                    loader=test_loader,
                    defense=defense,
                    device=device,
                    sample_bundle_path=paths["sample_bundle"],
                    sample_metadata={"artifact_type": "budget_sample_bundle", **artifact_identity},
                    exclusive=True,
                )
                metric_payload = {
                    "artifact_type": "budget_per_image_metrics",
                    **artifact_identity,
                    **metrics,
                    "images_are_independent_replicates": False,
                }
                _json_write_exclusive(paths["metric_json"], metric_payload)
                attackers[architecture] = {
                    "training": training,
                    "test_summary": metrics["summary"],
                    "checkpoint_path": str(paths["checkpoint"]),
                    "checkpoint_sha256": _sha256_file(paths["checkpoint"]),
                    "training_log_path": str(paths["training_log"]),
                    "training_log_sha256": _sha256_file(paths["training_log"]),
                    "metric_path": str(paths["metric_json"]),
                    "metric_sha256": _sha256_file(paths["metric_json"]),
                    "sample_bundle_path": str(paths["sample_bundle"]),
                    "sample_bundle_sha256": _sha256_file(paths["sample_bundle"]),
                }
            strongest = max(attackers, key=lambda name: attackers[name]["test_summary"]["ssim"])
            result = {
                **common_identity,
                "dataset_metadata": asdict(metadata),
                "attackers": attackers,
                "strongest_attacker": strongest,
                "worst_case_test_ssim": attackers[strongest]["test_summary"]["ssim"],
            }
            registry.update(
                run_id,
                "complete",
                selection_provenance=selection_provenance,
                result=result,
            )
            complete_after = sum(item.get("status") == "complete" for item in registry.records())
            _budget_event(
                "budget_job_complete",
                run_id=run_id,
                completed=complete_after,
                expected=24,
                progress=f"{complete_after}/24",
                strongest_attacker=strongest,
                worst_case_test_ssim=result["worst_case_test_ssim"],
            )
        except Exception as error:
            registry.update(
                run_id,
                "failed",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            complete_now = sum(item.get("status") == "complete" for item in registry.records())
            _budget_event(
                "budget_job_failure",
                run_id=run_id,
                error_type=type(error).__name__,
                error_message=str(error),
                completed=complete_now,
                expected=24,
                progress=f"{complete_now}/24",
            )
            raise


def run_smoke_matrix(
    *,
    contract: dict[str, Any],
    revision_root: str | Path,
    data_dir: str,
    device_name: str,
    num_workers: int,
    max_samples: int = 16,
) -> dict[str, Any]:
    """Exercise every dataset/split/method/attacker combination on real subsets."""
    root = Path(revision_root)
    device = resolve_device(device_name)
    results: list[dict[str, Any]] = []
    for dataset in contract["datasets"]:
        (train_loader, val_loader), num_classes, metadata = get_selection_loaders(
            dataset,
            data_dir,
            batch_size=min(8, max_samples),
            seed=42,
            num_workers=num_workers,
            max_samples=max_samples,
        )
        for split_point in contract["split_points"]:
            for method in contract["methods"]:
                alpha = 0.25 if method == "afd" else 0.0
                model = AFDSplitModel(num_classes=num_classes, config=_config(contract, split_point, "afd" if method == "afd" else "standard", alpha)).to(device)
                learned_path = None
                if method == "learned":
                    learned_path = root / "tmp" / "smoke" / dataset / split_point / "learned.pt"
                    train_universal_perturbation(
                        model=model,
                        train_loader=train_loader,
                        device=device,
                        max_l2=0.5,
                        epochs=1,
                        learning_rate=0.01,
                        output_path=learned_path,
                        metadata={"test_only": True},
                    )
                defense = _defense(method, 0.025 if method == "laplace" else None, learned_path, model.config.feature_shape)
                for architecture in ATTACKER_ARCHITECTURES:
                    checkpoint_path, log_path = _attacker_paths(
                        root, "tmp/smoke", dataset, split_point, 42, method, architecture
                    )
                    training = train_adaptive_attacker(
                        encoder=model.edge_encoder,
                        feature_shape=model.config.feature_shape,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        defense=defense,
                        architecture=architecture,
                        seed=42,
                        device=device,
                        checkpoint_path=checkpoint_path,
                        log_path=log_path,
                        max_epochs=1,
                        patience=1,
                    )
                    results.append(
                        {
                            "dataset": dataset,
                            "dataset_metadata": asdict(metadata),
                            "split_point": split_point,
                            "method": method,
                            "attacker": architecture,
                            "feature_shape": list(model.config.feature_shape),
                            "attacker_selected_epoch": training["selected_epoch"],
                            "status": "pass",
                        }
                    )
    report = {"combination_count": len(results), "expected_count": 32, "all_passed": len(results) == 32, "results": results}
    _json_write(root / "logs" / "smoke_matrix.json", report)
    return report


def consolidate_primary(contract: dict[str, Any], revision_root: str | Path) -> dict[str, Any]:
    root = Path(revision_root)
    _, selection_provenance, _ = load_sealed_selection(
        contract=contract,
        revision_root=root,
    )
    records = RunRegistry(root / "manifests" / "primary").records()
    expected = len(planned_primary_specs(contract))
    complete = [record for record in records if record["status"] == "complete"]
    for record in complete:
        require_selection_provenance(record, selection_provenance)
        validate_primary_result(record.get("result", {}))
    failed = [record for record in records if record["status"] == "failed"]
    summary = {
        "expected_runs": expected,
        "registered_runs": len(records),
        "complete_runs": len(complete),
        "failed_runs": len(failed),
        "all_planned_runs_complete": len(complete) == expected and len(records) == expected,
        "submission_ready": False,
        "selection_provenance": selection_provenance,
        "records": complete,
    }
    _json_write(root / "metrics" / "consolidated_primary.json", summary)
    csv_path = root / "metrics" / "consolidated_primary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_id", "dataset", "split_point", "seed", "method", "accuracy", "worst_case_ssim", "strongest_attacker",
            "deconv_mse", "deconv_psnr", "deconv_ssim", "deconv_lpips",
            "residual_mse", "residual_psnr", "residual_ssim", "residual_lpips",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in complete:
            result = record["result"]
            row = {
                "run_id": record["run_id"],
                "dataset": result["dataset"],
                "split_point": result["split_point"],
                "seed": result["seed"],
                "method": result["method"],
                "accuracy": result["utility"]["accuracy"],
                "worst_case_ssim": result["worst_case_test_ssim"],
                "strongest_attacker": result["strongest_attacker"],
            }
            for architecture, prefix in (("deconv_mse", "deconv"), ("residual_lpips", "residual")):
                for metric in ("mse", "psnr", "ssim", "lpips"):
                    row[f"{prefix}_{metric}"] = result["attackers"][architecture]["test_summary"][metric]
            writer.writerow(row)
    return summary


def seal_and_record_selection(
    contract: dict[str, Any],
    revision_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    root = Path(revision_root)
    _, selection_provenance, lock = seal_selection(
        contract=contract,
        revision_root=root,
        contract_path=contract_path,
    )
    manifest_path = root / "manifests" / "study_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selection_provenance"] = selection_provenance
        manifest["status"] = "selection_sealed"
        _json_write(manifest_path, manifest)
    return {
        "sealed": True,
        "frozen_path": str(root / FROZEN_RELATIVE_PATH),
        "lock_path": str(root / LOCK_RELATIVE_PATH),
        "selection_provenance": selection_provenance,
        "selected_defense_setting_count": len(
            lock["selection_contract"]["selected_defense_settings"]
        ),
    }


def _read_registry_status(directory: Path) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ] if directory.is_dir() else []
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "registered": len(records),
        "status_counts": status_counts,
        "running_run_ids": [record["run_id"] for record in records if record.get("status") == "running"],
        "failed_run_ids": [record["run_id"] for record in records if record.get("status") == "failed"],
    }


def pipeline_status(contract: dict[str, Any], revision_root: str | Path) -> dict[str, Any]:
    """Return selection and matrix state without mutating any artifact."""
    root = Path(revision_root)
    try:
        _, selection_provenance, lock = load_sealed_selection(
            contract=contract,
            revision_root=root,
        )
        selection_status: dict[str, Any] = {
            "sealed": True,
            "selection_provenance": selection_provenance,
            "selected_defense_settings": lock["selection_contract"]["selected_defense_settings"],
        }
    except SelectionProvenanceError as error:
        selection_status = {"sealed": False, "error": str(error)}
    readiness_path = root / "readiness" / "submission_readiness.json"
    submission_ready = None
    if readiness_path.is_file():
        submission_ready = bool(json.loads(readiness_path.read_text(encoding="utf-8")).get("submission_ready"))
    return {
        "selection": selection_status,
        "primary": _read_registry_status(root / "manifests" / "primary"),
        "semantic": _read_registry_status(root / "manifests" / "semantic"),
        "budget": _read_registry_status(root / "manifests" / BUDGET_REGISTRY_DIRECTORY),
        "historical_budget_v1": _read_registry_status(root / "manifests" / "budget"),
        "submission_ready": submission_ready,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive reconstruction revision pipeline")
    parser.add_argument(
        "command",
        choices=(
            "init",
            "smoke",
            "selection",
            "seal-selection",
            "status",
            "primary",
            "semantic",
            "init-budget-v2",
            "budget",
            "budget-v2",
            "init-benchmark-v1",
            "benchmark-budget-v1",
            "attacker-restarts-v1",
            "benchmark-status",
            "consolidate",
        ),
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--revision-root", default=str(DEFAULT_REVISION_ROOT))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--only-run-id", default=None)
    parser.add_argument("--smoke-max-samples", type=int, default=16)
    parser.add_argument(
        "--benchmark-config",
        default=str(PROJECT_ROOT / "config" / "benchmark_extension_v1.json"),
    )
    parser.add_argument(
        "--benchmark-preflight",
        default=str(PROJECT_ROOT / "audit" / "benchmark_revision_20260822" / "compute_preflight_p2.json"),
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="With benchmark-budget-v1, run the four full-data compute-timing scratch trainings only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    contract = load_contract(args.contract)
    if args.command == "init":
        print(json.dumps(initialize_study(args.contract, args.revision_root), indent=2))
    elif args.command == "smoke":
        print(
            json.dumps(
                run_smoke_matrix(
                    contract=contract,
                    revision_root=args.revision_root,
                    data_dir=args.data_dir,
                    device_name=args.device,
                    num_workers=args.num_workers,
                    max_samples=args.smoke_max_samples,
                ),
                indent=2,
            )
        )
    elif args.command == "selection":
        print(
            json.dumps(
                run_selection(
                    contract=contract,
                    revision_root=args.revision_root,
                    data_dir=args.data_dir,
                    device_name=args.device,
                    num_workers=args.num_workers,
                ),
                indent=2,
            )
        )
    elif args.command == "seal-selection":
        print(
            json.dumps(
                seal_and_record_selection(
                    contract,
                    args.revision_root,
                    args.contract,
                ),
                indent=2,
            )
        )
    elif args.command == "status":
        print(json.dumps(pipeline_status(contract, args.revision_root), indent=2))
    elif args.command == "primary":
        run_primary(
            contract=contract,
            revision_root=args.revision_root,
            data_dir=args.data_dir,
            device_name=args.device,
            num_workers=args.num_workers,
            only_run_id=args.only_run_id,
        )
    elif args.command == "semantic":
        run_semantic_probes(
            contract=contract,
            revision_root=args.revision_root,
            data_dir=args.data_dir,
            device_name=args.device,
            num_workers=args.num_workers,
            only_run_id=args.only_run_id,
        )
    elif args.command == "init-budget-v2":
        print(json.dumps(initialize_corrected_budget_stage(contract, args.revision_root), indent=2))
    elif args.command in {"budget", "budget-v2"}:
        run_auxiliary_budget_sweep(
            contract=contract,
            revision_root=args.revision_root,
            data_dir=args.data_dir,
            device_name=args.device,
            num_workers=args.num_workers,
            only_run_id=args.only_run_id,
        )
    elif args.command in {
        "init-benchmark-v1",
        "benchmark-budget-v1",
        "attacker-restarts-v1",
        "benchmark-status",
    }:
        from .benchmark_v1 import (
            benchmark_status,
            initialize_benchmark_v1,
            run_attacker_restarts_v1,
            run_benchmark_budget_v1,
            time_representative_full_data_runs,
        )

        common = {
            "config_path": args.benchmark_config,
            "contract_path": args.contract,
            "revision_root": args.revision_root,
        }
        if args.command == "init-benchmark-v1":
            print(json.dumps(initialize_benchmark_v1(**common), indent=2))
        elif args.command == "benchmark-status":
            print(
                json.dumps(
                    benchmark_status(**common, preflight_path=args.benchmark_preflight),
                    indent=2,
                )
            )
        elif args.command == "benchmark-budget-v1" and args.timing_only:
            print(
                json.dumps(
                    time_representative_full_data_runs(
                        **common,
                        data_dir=args.data_dir,
                        device_name=args.device,
                        num_workers=args.num_workers,
                        output_path=args.benchmark_preflight,
                    ),
                    indent=2,
                )
            )
        elif args.command == "benchmark-budget-v1":
            run_benchmark_budget_v1(
                **common,
                data_dir=args.data_dir,
                device_name=args.device,
                num_workers=args.num_workers,
                only_run_id=args.only_run_id,
                preflight_path=args.benchmark_preflight,
            )
        else:
            if args.timing_only:
                raise ValueError("--timing-only is valid only with benchmark-budget-v1")
            run_attacker_restarts_v1(
                **common,
                data_dir=args.data_dir,
                device_name=args.device,
                num_workers=args.num_workers,
                only_run_id=args.only_run_id,
                preflight_path=args.benchmark_preflight,
            )
    elif args.command == "consolidate":
        print(json.dumps(consolidate_primary(contract, args.revision_root), indent=2))


if __name__ == "__main__":
    main()
