"""Fail-closed, outcome-neutral audit of completed primary study artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .checkpointing import CheckpointCompatibilityError, load_split_model
from .experiment import TEST_DRAW_SEEDS
from .manifest import deterministic_run_id
from .metrics import METRIC_FIELDS, metric_versions
from .models import build_attacker
from .pipeline import planned_primary_specs
from .selection_provenance import SelectionProvenanceError, load_sealed_selection


EXPECTED_DATASET_METADATA: dict[str, dict[str, Any]] = {
    "celeba": {
        "name": "celeba",
        "partition_source": "canonical_boundaries",
        "synthetic": False,
        "train_size": 162_770,
        "val_size": 19_867,
        "test_size": 19_962,
    },
    "cifar10": {
        "name": "cifar10",
        "partition_source": "fixed_stratified_seed_314159",
        "synthetic": False,
        "train_size": 45_000,
        "val_size": 5_000,
        "test_size": 10_000,
    },
}
EXPECTED_RAW_COUNTS = {
    "primary_manifest": 80,
    "encoder_checkpoint": 40,
    "attacker_checkpoint": 160,
    "learned_perturbation_checkpoint": 20,
    "metric_json": 160,
    "sample_bundle": 160,
}


class ResultAuditError(RuntimeError):
    """Raised for a single raw-evidence contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultAuditError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ResultAuditError(f"{label} is non-finite")
    return number


def _project_root(revision_root: Path) -> Path:
    if revision_root.parent.name == "revisions":
        return revision_root.parent.parent
    return revision_root.parent


def resolve_artifact_path(declared: str, revision_root: Path) -> Path:
    """Resolve a stored path and reject every path outside the revision root."""
    raw = Path(declared)
    project_root = _project_root(revision_root)
    if raw.is_absolute():
        candidate = raw
    elif len(raw.parts) >= 2 and raw.parts[0] == "revisions" and raw.parts[1] == revision_root.name:
        candidate = project_root / raw
    else:
        candidate = revision_root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(revision_root)
    except ValueError as error:
        raise ResultAuditError(f"Referenced path escapes revision root: {declared}") from error
    return resolved


def validate_registry_records(
    records: list[Mapping[str, Any]], expected_specs: list[Mapping[str, Any]]
) -> list[str]:
    """Return registry errors for missing, duplicate, extra, or changed run cells."""
    errors: list[str] = []
    expected = {deterministic_run_id(spec): json.loads(json.dumps(spec, sort_keys=True)) for spec in expected_specs}
    seen: Counter[str] = Counter(str(record.get("run_id", "")) for record in records)
    duplicates = sorted(run_id for run_id, count in seen.items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate run IDs: {duplicates}")
    actual_ids = set(seen)
    missing = sorted(set(expected) - actual_ids)
    extra = sorted(actual_ids - set(expected))
    if missing:
        errors.append(f"Missing registered run IDs: {missing}")
    if extra:
        errors.append(f"Unexpected registered run IDs: {extra}")
    for record in records:
        run_id = str(record.get("run_id", ""))
        spec = record.get("spec")
        if run_id in expected and spec != expected[run_id]:
            errors.append(f"{run_id}: registered spec differs from the contract cell")
        if isinstance(spec, Mapping) and deterministic_run_id(spec) != run_id:
            errors.append(f"{run_id}: deterministic run ID does not match its spec")
        if record.get("status") != "complete":
            errors.append(f"{run_id}: status is {record.get('status')!r}, not 'complete'")
    return errors


def validate_selection_provenance(record: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    if record.get("selection_provenance") != expected:
        raise ResultAuditError("record provenance does not exactly match the sealed selection")


def validate_matched_original(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    if not torch.equal(reference, candidate):
        raise ResultAuditError("sample originals do not match across methods and attackers")


def validate_attacker_specific_reconstructions(reconstructions: Mapping[str, torch.Tensor]) -> None:
    required = {"deconv_mse", "residual_lpips"}
    if set(reconstructions) != required:
        raise ResultAuditError("both attacker reconstructions are required for comparison")
    if torch.equal(reconstructions["deconv_mse"], reconstructions["residual_lpips"]):
        raise ResultAuditError("attacker-specific reconstructions are identical")


def describe_test_utility(rows: list[dict[str, Any]], limit: float) -> dict[str, Any]:
    utility_by_cell = {
        (row["dataset"], row["split_point"], row["seed"], row["method"]): row["accuracy_percentage_points"]
        for row in rows
        if row.get("accuracy_percentage_points") is not None
    }
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        standard = utility_by_cell.get((row["dataset"], row["split_point"], row["seed"], "standard"))
        row_accuracy = row.get("accuracy_percentage_points")
        loss = None if standard is None or row_accuracy is None else standard - float(row_accuracy)
        row["utility_loss_vs_standard_percentage_points"] = loss
        row["test_utility_loss_exceeds_validation_limit"] = bool(loss is not None and loss > limit)
        if row["method"] != "standard":
            diagnostics.append(
                {
                    "dataset": row["dataset"],
                    "split_point": row["split_point"],
                    "seed": row["seed"],
                    "method": row["method"],
                    "loss_percentage_points": loss,
                    "validation_limit_percentage_points": limit,
                    "exceeds_limit": row["test_utility_loss_exceeds_validation_limit"],
                    "role": "descriptive_test_behavior_only_no_reselection",
                }
            )
    return {
        "no_reselection_from_test": True,
        "per_seed_rows": diagnostics,
        "exceedance_count": sum(row["exceeds_limit"] for row in diagnostics),
    }


def validate_training_history(training: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    history = training.get("history")
    if not isinstance(history, list) or not history:
        raise ResultAuditError("attacker early-stopping history is missing or empty")
    expected_epochs = list(range(1, len(history) + 1))
    observed_epochs = [entry.get("epoch") for entry in history if isinstance(entry, Mapping)]
    if observed_epochs != expected_epochs:
        raise ResultAuditError("attacker history epochs are not contiguous from one")
    validation_losses = [
        _require_finite_number(entry.get("validation_loss"), f"history[{index}].validation_loss")
        for index, entry in enumerate(history)
    ]
    for index, entry in enumerate(history):
        _require_finite_number(entry.get("train_loss"), f"history[{index}].train_loss")
    selected_epoch = int(training.get("selected_epoch", -1))
    expected_selected_epoch = min(range(len(validation_losses)), key=validation_losses.__getitem__) + 1
    if selected_epoch != expected_selected_epoch:
        raise ResultAuditError(
            f"selected attacker epoch {selected_epoch} does not match minimum validation-loss epoch {expected_selected_epoch}"
        )
    if training.get("epochs_completed") != len(history):
        raise ResultAuditError("epochs_completed does not match history length")
    best = _require_finite_number(training.get("best_validation_loss"), "best_validation_loss")
    if best != validation_losses[selected_epoch - 1]:
        raise ResultAuditError("best_validation_loss does not exactly match the selected history entry")
    max_epochs = int(contract["attackers"]["max_epochs"])
    patience = int(contract["attackers"]["patience"])
    if len(history) > max_epochs:
        raise ResultAuditError("attacker history exceeds the declared maximum epochs")
    if int(training.get("patience", -1)) != patience:
        raise ResultAuditError("attacker history patience differs from the contract")
    if len(history) < max_epochs and len(history) - selected_epoch < patience:
        raise ResultAuditError("early stopping occurred before the declared patience was exhausted")


def validate_metric_payload(
    payload: Mapping[str, Any],
    *,
    example_count: int,
    draw_seeds: list[int],
    manifest_summary: Mapping[str, Any],
    expected_versions: Mapping[str, Any],
) -> None:
    if payload.get("example_count") != example_count:
        raise ResultAuditError("metric example_count differs from the real test partition")
    if payload.get("mc_draw_seeds") != draw_seeds or payload.get("mc_draw_count") != len(draw_seeds):
        raise ResultAuditError("metric Monte Carlo draw policy differs from the contract")
    if payload.get("draws_are_independent_replicates") is not False:
        raise ResultAuditError("metric payload treats Monte Carlo draws as replicates")
    if payload.get("metric_versions") != expected_versions:
        raise ResultAuditError("metric implementation versions differ from the manifest")
    summary = payload.get("summary")
    per_image = payload.get("per_image")
    if not isinstance(summary, Mapping) or set(summary) != set(METRIC_FIELDS):
        raise ResultAuditError("metric summary fields are incomplete or unexpected")
    if not isinstance(per_image, Mapping) or set(per_image) != set(METRIC_FIELDS):
        raise ResultAuditError("per-image metric fields are incomplete or unexpected")
    for metric in METRIC_FIELDS:
        values = per_image[metric]
        if not isinstance(values, list):
            raise ResultAuditError(f"{metric} per-image values are not a JSON list")
        array = np.asarray(values)
        if array.ndim != 1 or array.size != example_count or array.dtype.kind not in "iuf":
            raise ResultAuditError(f"{metric} per-image vector has an invalid type, rank, or length")
        array = array.astype(np.float64, copy=False)
        if not np.isfinite(array).all():
            raise ResultAuditError(f"{metric} per-image vector contains a non-finite value")
        recomputed = float(np.mean(array))
        reported = _require_finite_number(summary[metric], f"summary.{metric}")
        manifest_value = _require_finite_number(manifest_summary.get(metric), f"manifest_summary.{metric}")
        if recomputed != reported or reported != manifest_value:
            raise ResultAuditError(
                f"{metric} mean drift: recomputed={recomputed!r}, metric={reported!r}, manifest={manifest_value!r}"
            )


def validate_sample_bundle(payload: Mapping[str, Any], *, draw_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if set(payload) != {"original", "reconstruction", "draw_seed"}:
        raise ResultAuditError("sample bundle fields are incomplete or unexpected")
    original = payload.get("original")
    reconstruction = payload.get("reconstruction")
    if not isinstance(original, torch.Tensor) or not isinstance(reconstruction, torch.Tensor):
        raise ResultAuditError("sample bundle images are not tensors")
    expected_shape = (5, 3, 32, 32)
    if tuple(original.shape) != expected_shape or tuple(reconstruction.shape) != expected_shape:
        raise ResultAuditError("sample bundle tensors are not 5x3x32x32")
    if not torch.isfinite(original).all() or not torch.isfinite(reconstruction).all():
        raise ResultAuditError("sample bundle contains non-finite values")
    if payload.get("draw_seed") != draw_seed:
        raise ResultAuditError("sample bundle draw seed differs from the first fixed test draw")
    return original, reconstruction


def validate_encoder_checkpoint(
    path: Path,
    *,
    dataset: str,
    split_point: str,
    seed: int,
    source_method: str,
    feature_shape: tuple[int, int, int],
    selected_epoch: int,
    hyperparameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ResultAuditError(f"missing encoder checkpoint: {path}")
    try:
        model, payload = load_split_model(
            path,
            num_classes=2 if dataset == "celeba" else 10,
            device="cpu",
            expected_feature_shape=feature_shape,
        )
        del model
    except (CheckpointCompatibilityError, KeyError, RuntimeError, ValueError) as error:
        raise ResultAuditError(f"incompatible encoder checkpoint: {error}") from error
    checks = {
        "schema_version": 2,
        "dataset": dataset,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "epoch": selected_epoch,
        "feature_shape": list(feature_shape),
        "config": dict(hyperparameters),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ResultAuditError(f"encoder checkpoint {key} metadata mismatch")
    if payload["config"].get("split_point") != split_point or payload["config"].get("method") != source_method:
        raise ResultAuditError("encoder checkpoint split/method metadata mismatch")
    return payload


def validate_attacker_checkpoint(
    path: Path,
    *,
    architecture: str,
    feature_shape: tuple[int, int, int],
    seed: int,
    training: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ResultAuditError(f"missing attacker checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ResultAuditError("attacker checkpoint is not a mapping")
    expected = {
        "schema_version": 2,
        "architecture": architecture,
        "feature_shape": list(feature_shape),
        "seed": seed,
        "max_epochs": int(contract["attackers"]["max_epochs"]),
        "patience": int(contract["attackers"]["patience"]),
        "selected_epoch": int(training["selected_epoch"]),
        "best_validation_loss": float(training["best_validation_loss"]),
        "loss": contract["attackers"]["losses"][architecture],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ResultAuditError(f"attacker checkpoint {key} metadata mismatch")
    try:
        attacker = build_attacker(architecture, feature_shape)
        attacker.load_state_dict(payload["model_state_dict"], strict=True)
        del attacker
    except (KeyError, RuntimeError, ValueError) as error:
        raise ResultAuditError(f"incompatible attacker checkpoint tensors: {error}") from error
    return payload


def validate_learned_checkpoint(
    path: Path,
    *,
    dataset: str,
    split_point: str,
    seed: int,
    feature_shape: tuple[int, int, int],
    selected_value: float,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ResultAuditError(f"missing learned-perturbation checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
        raise ResultAuditError("learned-perturbation checkpoint schema is incompatible")
    noise = payload.get("noise")
    if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (1, *feature_shape):
        raise ResultAuditError("learned perturbation tensor has an incompatible shape")
    if not torch.isfinite(noise).all():
        raise ResultAuditError("learned perturbation tensor contains non-finite values")
    realized = float(noise.norm(p=2))
    declared_max = _require_finite_number(payload.get("max_l2"), "learned max_l2")
    declared_realized = _require_finite_number(payload.get("realized_l2"), "learned realized_l2")
    if declared_max != selected_value:
        raise ResultAuditError("learned perturbation selected L2 limit metadata mismatch")
    if not math.isclose(declared_realized, realized, rel_tol=0.0, abs_tol=1e-6):
        raise ResultAuditError("learned perturbation realized L2 metadata differs from the tensor beyond float32 tolerance")
    if max(realized, declared_realized) > selected_value + 1e-6:
        raise ResultAuditError("learned perturbation exceeds its selected global L2 limit")
    expected_metadata = {"stage": "primary", "dataset": dataset, "split_point": split_point, "seed": seed}
    if payload.get("metadata") != expected_metadata:
        raise ResultAuditError("learned perturbation provenance metadata mismatch")
    return payload


def validate_manifest_learned_l2(manifest_value: Any, checkpoint_value: Any) -> None:
    manifest_realized = _require_finite_number(manifest_value, "manifest learned realized L2")
    checkpoint_realized = _require_finite_number(checkpoint_value, "checkpoint learned realized L2")
    if not math.isclose(manifest_realized, checkpoint_realized, rel_tol=0.0, abs_tol=1e-6):
        raise ResultAuditError("manifest learned L2 differs from its checkpoint beyond float32 tolerance")


def _selected_setting(frozen: Mapping[str, Any], dataset: str, split_point: str, method: str) -> tuple[Any, str]:
    setting = frozen["settings"][dataset][split_point]
    if method == "standard":
        return None, str(setting["standard_adaptive_evaluation"]["strongest_validation_attacker"])
    method_setting = setting["methods"][method]
    return method_setting["selected_value"], str(method_setting["selected_strongest_attacker"])


def audit_validation_utility_selection(
    contract: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    limit = float(contract["utility_loss_limit_percentage_points"])
    for dataset in contract["datasets"]:
        for split_point in contract["split_points"]:
            setting = frozen["settings"][dataset][split_point]
            standard_accuracy = float(setting["standard_validation_accuracy"])
            for method in ("afd", "laplace", "learned"):
                method_setting = setting["methods"][method]
                selected_value = method_setting["selected_value"]
                candidates = [
                    candidate
                    for candidate in method_setting["candidates"]
                    if candidate["candidate_value"] == selected_value
                ]
                if len(candidates) != 1:
                    errors.append(f"{dataset}/{split_point}/{method}: selected candidate is not unique")
                    continue
                accuracy = float(candidates[0]["utility"]["accuracy"])
                loss = 100.0 * (standard_accuracy - accuracy)
                compatible = loss <= limit + 1e-12
                declared = method_setting.get("status") == "utility_compatible"
                if compatible != declared:
                    errors.append(f"{dataset}/{split_point}/{method}: validation utility status mismatch")
                rows.append(
                    {
                        "dataset": dataset,
                        "split_point": split_point,
                        "method": method,
                        "selected_value": selected_value,
                        "standard_validation_accuracy_percentage_points": 100.0 * standard_accuracy,
                        "selected_validation_accuracy_percentage_points": 100.0 * accuracy,
                        "validation_utility_loss_percentage_points": loss,
                        "limit_percentage_points": limit,
                        "within_limit": compatible,
                    }
                )
    return {"pass": len(rows) == 12 and not errors, "rows": rows, "errors": errors}


def _discover_raw_artifacts(root: Path) -> dict[Path, str]:
    artifacts: dict[Path, str] = {}
    for path in (root / "manifests" / "primary").glob("*.json"):
        artifacts[path.resolve()] = "primary_manifest"
    for path in (root / "checkpoints" / "encoders").rglob("*.pt"):
        artifacts[path.resolve()] = "encoder_checkpoint"
    for path in (root / "checkpoints" / "primary").rglob("*.pt"):
        kind = "attacker_checkpoint" if path.name.startswith("attacker_") else "learned_perturbation_checkpoint"
        artifacts[path.resolve()] = kind
    for dataset in ("celeba", "cifar10"):
        for path in (root / "metrics" / dataset).rglob("*"):
            if path.is_file():
                artifacts[path.resolve()] = "sample_bundle" if path.name.endswith("_samples.pt") else "metric_json"
    return artifacts


def _learned_checkpoint_path(root: Path, dataset: str, split_point: str, seed: int, value: float) -> Path:
    value_label = f"{value:g}"
    return (root / "checkpoints" / "primary" / dataset / split_point / f"seed_{seed}" / f"learned_l2_{value_label}.pt").resolve()


def audit_primary_results(contract: Mapping[str, Any], revision_root: str | Path) -> dict[str, Any]:
    """Audit primary raw evidence without mutating the revision or audit outputs."""
    root = Path(revision_root).resolve()
    issues: list[dict[str, Any]] = []
    expected_specs = planned_primary_specs(dict(contract))
    expected_spec_by_id = {deterministic_run_id(spec): spec for spec in expected_specs}

    def issue(gate: str, message: str, run_id: str | None = None) -> None:
        payload: dict[str, Any] = {"gate": gate, "message": message}
        if run_id is not None:
            payload["run_id"] = run_id
        issues.append(payload)

    try:
        frozen, provenance, lock = load_sealed_selection(
            contract=dict(contract),
            revision_root=root,
            contract_path=root / "study_contract.json",
        )
    except (SelectionProvenanceError, OSError, ValueError) as error:
        frozen, provenance, lock = {}, {}, {}
        issue("selection_provenance", str(error))

    manifest_paths = sorted((root / "manifests" / "primary").glob("*.json"))
    records: list[dict[str, Any]] = []
    record_paths: dict[str, Path] = {}
    for path in manifest_paths:
        try:
            record = strict_json_load(path)
            if not isinstance(record, dict):
                raise ResultAuditError("manifest root is not an object")
            records.append(record)
            run_id = str(record.get("run_id", ""))
            if run_id in record_paths:
                issue("registry", f"duplicate run ID also appears in {record_paths[run_id]}", run_id)
            else:
                record_paths[run_id] = path.resolve()
            if path.stem != run_id:
                issue("registry", f"manifest filename {path.stem!r} differs from run_id", run_id)
        except (OSError, ValueError, ResultAuditError) as error:
            issue("registry", f"{path}: {error}")
    for message in validate_registry_records(records, expected_specs):
        issue("registry", message)

    expected_artifacts: dict[Path, str] = {path.resolve(): "primary_manifest" for path in manifest_paths}
    rows: list[dict[str, Any]] = []
    originals_by_cell: dict[tuple[str, str, int], torch.Tensor] = {}
    encoder_cache: dict[Path, Mapping[str, Any]] = {}
    attacker_cache: dict[Path, Mapping[str, Any]] = {}
    learned_cache: dict[Path, Mapping[str, Any]] = {}
    metric_verified = 0
    sample_verified = 0
    training_history_verified = 0

    for record in records:
        run_id = str(record.get("run_id", ""))
        record_issue_count = len(issues)
        spec = record.get("spec")
        result = record.get("result")
        if run_id not in expected_spec_by_id or not isinstance(spec, Mapping) or not isinstance(result, Mapping):
            issue("record_structure", "record is not a complete declared contract cell", run_id)
            continue
        try:
            validate_selection_provenance(record, provenance)
        except ResultAuditError as error:
            issue("selection_provenance", str(error), run_id)
        dataset = str(spec["dataset"])
        split_point = str(spec["split_point"])
        method = str(spec["method"])
        seed = int(spec["seed"])
        feature_shape = tuple(int(value) for value in contract["split_points"][split_point])
        expected_count = int(EXPECTED_DATASET_METADATA[dataset]["test_size"])
        source_method = "afd" if method == "afd" else "standard"

        exact_fields = {
            "dataset": dataset,
            "split_point": split_point,
            "method": method,
            "seed": seed,
            "dataset_metadata": EXPECTED_DATASET_METADATA[dataset],
            "evaluation_partition": "test_after_frozen_selection",
            "metric_versions": metric_versions(),
        }
        for key, expected in exact_fields.items():
            if result.get(key) != expected:
                issue("record_metadata", f"result.{key} differs from the declared value", run_id)

        try:
            selected_value, validation_attacker = _selected_setting(frozen, dataset, split_point, method)
            if result.get("selected_defense_value") != selected_value:
                raise ResultAuditError("selected defense value differs from the frozen validation choice")
            if result.get("validation_selected_attacker") != validation_attacker:
                raise ResultAuditError("validation-selected attacker differs from the frozen selection")
            defense = result.get("defense")
            if not isinstance(defense, Mapping) or defense.get("method") != method:
                raise ResultAuditError("defense metadata does not match the method")
            if defense.get("stochastic") is not (method == "laplace"):
                raise ResultAuditError("defense stochastic flag does not match the method")
            if method == "afd" and result.get("encoder_hyperparameters", {}).get("alpha") != selected_value:
                raise ResultAuditError("AFD encoder alpha differs from the selected value")
            if method == "laplace" and defense.get("laplace_scale") != selected_value:
                raise ResultAuditError("Laplace scale differs from the selected value")
            if method == "learned" and defense.get("learned_max_l2") != selected_value:
                raise ResultAuditError("learned perturbation limit differs from the selected value")
        except (KeyError, TypeError, ResultAuditError) as error:
            issue("selected_defense", str(error), run_id)
            selected_value = result.get("selected_defense_value")

        utility = result.get("utility")
        try:
            if not isinstance(utility, Mapping):
                raise ResultAuditError("utility payload is missing")
            draw_seeds = list(contract["test_mc_draw_seeds"]) if method == "laplace" else [int(contract["test_mc_draw_seeds"][0])]
            if utility.get("draw_seeds") != draw_seeds:
                raise ResultAuditError("utility Monte Carlo draw seeds differ from the fixed policy")
            if utility.get("draws_are_independent_replicates") is not False:
                raise ResultAuditError("utility payload treats Monte Carlo draws as replicates")
            draw_accuracies = np.asarray(utility.get("draw_accuracies"))
            if draw_accuracies.ndim != 1 or draw_accuracies.size != len(draw_seeds) or draw_accuracies.dtype.kind not in "iuf":
                raise ResultAuditError("utility draw accuracies have an invalid type or count")
            draw_accuracies = draw_accuracies.astype(np.float64, copy=False)
            if not np.isfinite(draw_accuracies).all():
                raise ResultAuditError("utility draw accuracies contain a non-finite value")
            accuracy = _require_finite_number(utility.get("accuracy"), "utility.accuracy")
            if accuracy != float(np.mean(draw_accuracies)):
                raise ResultAuditError("utility accuracy does not exactly match the draw mean")
            if utility.get("example_count") != expected_count:
                raise ResultAuditError("utility example count differs from the real test partition")
        except (TypeError, ValueError, ResultAuditError) as error:
            issue("monte_carlo_policy", str(error), run_id)
            draw_seeds = [int(contract["test_mc_draw_seeds"][0])]
            accuracy = None

        try:
            encoder_path = resolve_artifact_path(str(result.get("encoder_checkpoint", "")), root)
            expected_artifacts[encoder_path] = "encoder_checkpoint"
            if encoder_path not in encoder_cache:
                encoder_cache[encoder_path] = validate_encoder_checkpoint(
                    encoder_path,
                    dataset=dataset,
                    split_point=split_point,
                    seed=seed,
                    source_method=source_method,
                    feature_shape=feature_shape,
                    selected_epoch=int(result.get("encoder_selected_epoch", -1)),
                    hyperparameters=result.get("encoder_hyperparameters", {}),
                )
        except (OSError, TypeError, ValueError, ResultAuditError) as error:
            issue("encoder_checkpoint", str(error), run_id)

        if method == "learned" and isinstance(selected_value, (int, float)):
            learned_path = _learned_checkpoint_path(root, dataset, split_point, seed, float(selected_value))
            expected_artifacts[learned_path] = "learned_perturbation_checkpoint"
            try:
                if learned_path not in learned_cache:
                    learned_cache[learned_path] = validate_learned_checkpoint(
                        learned_path,
                        dataset=dataset,
                        split_point=split_point,
                        seed=seed,
                        feature_shape=feature_shape,
                        selected_value=float(selected_value),
                    )
                validate_manifest_learned_l2(
                    result.get("defense", {}).get("learned_realized_l2"),
                    learned_cache[learned_path]["realized_l2"],
                )
            except (OSError, TypeError, ValueError, ResultAuditError) as error:
                issue("learned_perturbation_checkpoint", str(error), run_id)

        attackers = result.get("attackers")
        if not isinstance(attackers, Mapping) or set(attackers) != set(contract["attackers"]["architectures"]):
            issue("attacker_structure", "attacker result set is incomplete or unexpected", run_id)
            continue
        attacker_ssim: dict[str, float] = {}
        reconstructions: dict[str, torch.Tensor] = {}
        for architecture in contract["attackers"]["architectures"]:
            attacker = attackers[architecture]
            if not isinstance(attacker, Mapping):
                issue("attacker_structure", f"{architecture} result is not an object", run_id)
                continue
            training = attacker.get("training")
            try:
                if not isinstance(training, Mapping):
                    raise ResultAuditError("attacker training metadata is missing")
                if training.get("architecture") != architecture or training.get("seed") != seed:
                    raise ResultAuditError("attacker training identity metadata mismatch")
                if training.get("feature_shape") != list(feature_shape):
                    raise ResultAuditError("attacker training feature shape mismatch")
                validate_training_history(training, contract)
                training_history_verified += 1
            except (KeyError, TypeError, ValueError, ResultAuditError) as error:
                issue("early_stopping_history", f"{architecture}: {error}", run_id)

            try:
                checkpoint_path = resolve_artifact_path(str(training.get("checkpoint_path", "")), root)
                expected_artifacts[checkpoint_path] = "attacker_checkpoint"
                if checkpoint_path not in attacker_cache:
                    attacker_cache[checkpoint_path] = validate_attacker_checkpoint(
                        checkpoint_path,
                        architecture=architecture,
                        feature_shape=feature_shape,
                        seed=seed,
                        training=training,
                        contract=contract,
                    )
            except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
                issue("attacker_checkpoint", f"{architecture}: {error}", run_id)

            try:
                metric_path = resolve_artifact_path(str(attacker.get("metrics_path", "")), root)
                expected_artifacts[metric_path] = "metric_json"
                if not metric_path.is_file():
                    raise ResultAuditError(f"missing metric JSON: {metric_path}")
                metric_payload = strict_json_load(metric_path)
                if not isinstance(metric_payload, Mapping):
                    raise ResultAuditError("metric JSON root is not an object")
                validate_metric_payload(
                    metric_payload,
                    example_count=expected_count,
                    draw_seeds=draw_seeds,
                    manifest_summary=attacker.get("test_summary", {}),
                    expected_versions=result.get("metric_versions", {}),
                )
                attacker_ssim[architecture] = float(metric_payload["summary"]["ssim"])
                metric_verified += 1
            except (OSError, TypeError, ValueError, ResultAuditError) as error:
                issue("metric_integrity", f"{architecture}: {error}", run_id)

            try:
                sample_path = resolve_artifact_path(str(attacker.get("sample_bundle_path", "")), root)
                expected_artifacts[sample_path] = "sample_bundle"
                if not sample_path.is_file():
                    raise ResultAuditError(f"missing sample bundle: {sample_path}")
                sample_payload = torch.load(sample_path, map_location="cpu", weights_only=True)
                if not isinstance(sample_payload, Mapping):
                    raise ResultAuditError("sample bundle root is not a mapping")
                original, reconstruction = validate_sample_bundle(sample_payload, draw_seed=draw_seeds[0])
                cell = (dataset, split_point, seed)
                if cell in originals_by_cell:
                    validate_matched_original(originals_by_cell[cell], original)
                originals_by_cell.setdefault(cell, original.clone())
                reconstructions[architecture] = reconstruction
                sample_verified += 1
            except (OSError, TypeError, ValueError, ResultAuditError) as error:
                issue("sample_bundle", f"{architecture}: {error}", run_id)

        if set(reconstructions) == set(contract["attackers"]["architectures"]):
            try:
                validate_attacker_specific_reconstructions(reconstructions)
            except ResultAuditError as error:
                issue("sample_bundle", str(error), run_id)

        if set(attacker_ssim) == set(contract["attackers"]["architectures"]):
            strongest = max(attacker_ssim, key=attacker_ssim.get)
            if result.get("strongest_attacker") != strongest:
                issue("strongest_attacker", "strongest attacker does not maximize test SSIM", run_id)
            if result.get("worst_case_test_ssim") != attacker_ssim[strongest]:
                issue("strongest_attacker", "worst-case SSIM does not exactly match the strongest attacker", run_id)

        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "split_point": split_point,
                "seed": seed,
                "method": method,
                "selected_defense_value": selected_value,
                "accuracy_percentage_points": None if accuracy is None else 100.0 * accuracy,
                "worst_case_ssim": result.get("worst_case_test_ssim"),
                "strongest_attacker": result.get("strongest_attacker"),
                "validation_selected_attacker": result.get("validation_selected_attacker"),
                "record_passed": len(issues) == record_issue_count,
            }
        )

    limit = float(contract["utility_loss_limit_percentage_points"])
    descriptive_test_utility = describe_test_utility(rows, limit)

    validation_utility = (
        audit_validation_utility_selection(contract, frozen)
        if frozen
        else {"pass": False, "rows": [], "errors": ["sealed selection unavailable"]}
    )
    for message in validation_utility["errors"]:
        issue("validation_utility_selection", message)

    actual_artifacts = _discover_raw_artifacts(root)
    actual_counts = Counter(actual_artifacts.values())
    expected_counts = Counter(expected_artifacts.values())
    missing_artifacts = sorted(str(path) for path in set(expected_artifacts) - set(actual_artifacts))
    unexpected_artifacts = sorted(str(path) for path in set(actual_artifacts) - set(expected_artifacts))
    if actual_counts != Counter(EXPECTED_RAW_COUNTS):
        issue("artifact_inventory", f"raw artifact counts are {dict(sorted(actual_counts.items()))}, expected {EXPECTED_RAW_COUNTS}")
    if expected_counts != Counter(EXPECTED_RAW_COUNTS):
        issue("artifact_inventory", f"referenced artifact counts are {dict(sorted(expected_counts.items()))}, expected {EXPECTED_RAW_COUNTS}")
    if missing_artifacts:
        issue("artifact_inventory", f"missing referenced artifacts: {missing_artifacts}")
    if unexpected_artifacts:
        issue("artifact_inventory", f"unexpected raw artifacts: {unexpected_artifacts}")

    raw_evidence = []
    for path, kind in sorted(actual_artifacts.items(), key=lambda item: str(item[0])):
        raw_evidence.append(
            {
                "artifact_type": kind,
                "path": str(path.relative_to(_project_root(root))),
                "size_bytes": path.stat().st_size,
                "mode": f"{path.stat().st_mode & 0o777:04o}",
                "sha256": sha256_file(path),
            }
        )

    gate_names = (
        "registry",
        "selection_provenance",
        "record_structure",
        "record_metadata",
        "selected_defense",
        "validation_utility_selection",
        "monte_carlo_policy",
        "encoder_checkpoint",
        "learned_perturbation_checkpoint",
        "attacker_structure",
        "early_stopping_history",
        "attacker_checkpoint",
        "metric_integrity",
        "sample_bundle",
        "strongest_attacker",
        "artifact_inventory",
    )
    gates = {
        name: {
            "pass": not any(entry["gate"] == name for entry in issues),
            "error_count": sum(entry["gate"] == name for entry in issues),
        }
        for name in gate_names
    }
    quarantined = sorted({str(entry["run_id"]) for entry in issues if entry.get("run_id")})
    rows.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], row["seed"]))
    report = {
        "schema_version": 1,
        "audit": "completed_primary_results",
        "passed": not issues,
        "outcome_neutrality": "Integrity gates do not require any method to win, any effect to be large, or any p-value to cross a threshold.",
        "contract_matrix": {
            "expected_records": len(expected_specs),
            "registered_records": len(records),
            "required_seeds": list(contract["seeds"]),
            "datasets": list(contract["datasets"]),
            "split_points": list(contract["split_points"]),
            "methods": list(contract["methods"]),
            "attackers": list(contract["attackers"]["architectures"]),
        },
        "selection_provenance": provenance,
        "selection_lock_summary": {
            "partition": lock.get("selection_contract", {}).get("partition"),
            "test_access": lock.get("selection_contract", {}).get("test_access"),
            "selected_defense_setting_count": len(lock.get("selection_contract", {}).get("selected_defense_settings", [])),
        },
        "gates": gates,
        "verification_counts": {
            "metric_json_verified": metric_verified,
            "sample_bundles_verified": sample_verified,
            "training_histories_verified": training_history_verified,
            "encoder_checkpoints_verified": len(encoder_cache),
            "attacker_checkpoints_verified": len(attacker_cache),
            "learned_perturbation_checkpoints_verified": len(learned_cache),
            "raw_artifacts_hashed": len(raw_evidence),
        },
        "artifact_counts": dict(sorted(actual_counts.items())),
        "validation_utility_selection": validation_utility,
        "descriptive_test_utility": descriptive_test_utility,
        "quarantined_run_ids": quarantined,
        "issues": issues,
        "primary_seed_results": rows,
        "raw_evidence": raw_evidence,
    }
    return report
