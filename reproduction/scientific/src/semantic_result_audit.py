"""Fail-closed, outcome-neutral audit of completed semantic-probe evidence."""

from __future__ import annotations

import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .experiment import (
    SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
    SEMANTIC_RAW_SCHEMA_VERSION,
    TEST_DRAW_SEEDS,
    MultilabelProbe,
)
from .pipeline import planned_semantic_specs
from .result_audit import (
    ResultAuditError,
    resolve_artifact_path,
    sha256_file,
    strict_json_load,
    validate_registry_records,
)
from .selection_provenance import load_sealed_selection


CELEBA_ATTRIBUTE_NAMES = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)
SMILING_INDEX = CELEBA_ATTRIBUTE_NAMES.index("Smiling")
SEMANTIC_TARGET_INDICES = tuple(index for index in range(40) if index != SMILING_INDEX)
SEMANTIC_TARGET_NAMES = tuple(CELEBA_ATTRIBUTE_NAMES[index] for index in SEMANTIC_TARGET_INDICES)
CANONICAL_TEST_EXAMPLE_IDS = tuple(f"{index:06d}.jpg" for index in range(182_638, 202_600))
EXPECTED_DATASET_METADATA = {
    "name": "celeba",
    "partition_source": "canonical_boundaries",
    "synthetic": False,
    "train_size": 162_770,
    "val_size": 19_867,
    "test_size": 19_962,
}
RAW_KEYS = {
    "schema_version",
    "probabilities_by_draw",
    "binary_targets",
    "example_ids",
    "target_names",
    "target_indices",
    "draw_seeds",
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultAuditError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResultAuditError(f"{label} is non-finite")
    return result


def validate_semantic_history(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    history = payload.get("history")
    if not isinstance(history, list) or not 1 <= len(history) <= int(contract["attackers"]["max_epochs"]):
        raise ResultAuditError("semantic history length is outside 1--max_epochs")
    epochs = [entry.get("epoch") for entry in history if isinstance(entry, Mapping)]
    if epochs != list(range(1, len(history) + 1)):
        raise ResultAuditError("semantic history epochs are not contiguous from one")
    train_losses = [
        _finite_number(entry.get("train_loss"), f"history[{index}].train_loss")
        for index, entry in enumerate(history)
    ]
    del train_losses
    validation_losses = [
        _finite_number(entry.get("validation_loss"), f"history[{index}].validation_loss")
        for index, entry in enumerate(history)
    ]
    selected_epoch = int(payload.get("selected_epoch", -1))
    expected_epoch = min(range(len(validation_losses)), key=validation_losses.__getitem__) + 1
    if selected_epoch != expected_epoch:
        raise ResultAuditError(
            f"semantic selected epoch {selected_epoch} is not minimum validation-loss epoch {expected_epoch}"
        )
    if payload.get("epochs_completed") != len(history):
        raise ResultAuditError("semantic epochs_completed does not match history length")
    if _finite_number(payload.get("best_validation_loss"), "best_validation_loss") != validation_losses[selected_epoch - 1]:
        raise ResultAuditError("semantic best_validation_loss does not match the selected epoch")
    max_epochs = int(contract["attackers"]["max_epochs"])
    patience = int(contract["attackers"]["patience"])
    if payload.get("max_epochs") != max_epochs or payload.get("patience") != patience:
        raise ResultAuditError("semantic maximum epochs or patience differs from the contract")
    if len(history) < max_epochs and len(history) - selected_epoch < patience:
        raise ResultAuditError("semantic early stopping occurred before patience was exhausted")


def load_semantic_raw_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ResultAuditError(f"missing semantic raw bundle: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ResultAuditError("semantic NPZ contains duplicate members")
            expected_members = {f"{key}.npy" for key in RAW_KEYS}
            if set(members) != expected_members:
                raise ResultAuditError("semantic NPZ members are incomplete or unexpected")
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != RAW_KEYS:
                raise ResultAuditError("semantic NPZ arrays are incomplete or unexpected")
            return {name: np.asarray(bundle[name]) for name in bundle.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ResultAuditError(f"malformed semantic NPZ: {error}") from error


def validate_semantic_raw_arrays(
    arrays: Mapping[str, np.ndarray], *, method: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = np.asarray(arrays["schema_version"])
    if schema.shape != () or int(schema) != SEMANTIC_RAW_SCHEMA_VERSION:
        raise ResultAuditError("semantic raw schema version is incompatible")
    expected_draws = TEST_DRAW_SEEDS if method == "laplace" else TEST_DRAW_SEEDS[:1]
    probabilities = np.asarray(arrays["probabilities_by_draw"])
    targets = np.asarray(arrays["binary_targets"])
    example_ids = np.asarray(arrays["example_ids"])
    target_names = np.asarray(arrays["target_names"])
    target_indices = np.asarray(arrays["target_indices"])
    draw_seeds = np.asarray(arrays["draw_seeds"])
    expected_shape = (len(expected_draws), len(CANONICAL_TEST_EXAMPLE_IDS), len(SEMANTIC_TARGET_NAMES))
    if probabilities.shape != expected_shape or probabilities.dtype.kind != "f":
        raise ResultAuditError(f"semantic probabilities have shape/dtype {probabilities.shape}/{probabilities.dtype}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ResultAuditError("semantic probabilities are non-finite or outside [0,1]")
    if targets.shape != expected_shape[1:] or targets.dtype.kind not in "biu":
        raise ResultAuditError(f"semantic targets have shape/dtype {targets.shape}/{targets.dtype}")
    if not np.isin(targets, (0, 1)).all():
        raise ResultAuditError("semantic targets are not binary")
    if example_ids.ndim != 1 or tuple(example_ids.tolist()) != CANONICAL_TEST_EXAMPLE_IDS:
        raise ResultAuditError("semantic example IDs do not match the canonical test partition")
    if target_names.ndim != 1 or tuple(target_names.tolist()) != SEMANTIC_TARGET_NAMES:
        raise ResultAuditError("semantic target names or order differ from the 39 non-Smiling attributes")
    if target_indices.ndim != 1 or tuple(map(int, target_indices.tolist())) != SEMANTIC_TARGET_INDICES:
        raise ResultAuditError("semantic target indices differ from the canonical non-Smiling indices")
    if draw_seeds.ndim != 1 or tuple(map(int, draw_seeds.tolist())) != expected_draws:
        raise ResultAuditError("semantic draw seeds/count differ from the declared method policy")

    averaged = probabilities.mean(axis=0, dtype=np.float64)
    attribute_rows: list[dict[str, Any]] = []
    for column, (target_index, target_name) in enumerate(zip(SEMANTIC_TARGET_INDICES, SEMANTIC_TARGET_NAMES, strict=True)):
        truth = targets[:, column]
        probability = averaged[:, column]
        attribute_rows.append(
            {
                "attribute_index": target_index,
                "attribute_name": target_name,
                "auroc": float(roc_auc_score(truth, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, probability >= 0.5)),
            }
        )
    aurocs = np.asarray([row["auroc"] for row in attribute_rows], dtype=np.float64)
    balanced = np.asarray([row["balanced_accuracy"] for row in attribute_rows], dtype=np.float64)
    summary = {
        "macro_auroc": float(aurocs.mean()),
        "macro_balanced_accuracy": float(balanced.mean()),
        "per_attribute_auroc": aurocs.tolist(),
        "per_attribute_balanced_accuracy": balanced.tolist(),
        "auroc_sample_sd": float(aurocs.std(ddof=1)),
        "balanced_accuracy_sample_sd": float(balanced.std(ddof=1)),
        "attribute_count": len(attribute_rows),
        "test_example_count": len(CANONICAL_TEST_EXAMPLE_IDS),
        "semantic_draw_seeds": list(expected_draws),
        "semantic_draw_count": len(expected_draws),
    }
    return summary, attribute_rows


def _require_equal_metric(recomputed: Any, reported: Any, label: str) -> None:
    if isinstance(recomputed, list):
        reported_array = np.asarray(reported, dtype=np.float64)
        recomputed_array = np.asarray(recomputed, dtype=np.float64)
        if reported_array.shape != recomputed_array.shape or not np.allclose(
            reported_array, recomputed_array, rtol=0.0, atol=1e-12
        ):
            raise ResultAuditError(f"semantic metric drift for {label}")
        return
    if not math.isclose(_finite_number(reported, label), float(recomputed), rel_tol=0.0, abs_tol=1e-12):
        raise ResultAuditError(f"semantic metric drift for {label}")


def validate_cross_bundle_identity(
    reference_targets: np.ndarray,
    reference_example_ids: np.ndarray,
    candidate_targets: np.ndarray,
    candidate_example_ids: np.ndarray,
) -> None:
    if not np.array_equal(reference_targets, candidate_targets):
        raise ResultAuditError("semantic target arrays differ across registered cells")
    if not np.array_equal(reference_example_ids, candidate_example_ids):
        raise ResultAuditError("semantic example arrays differ across registered cells")


def _selected_value(frozen: Mapping[str, Any], split_point: str, method: str) -> Any:
    if method == "standard":
        return None
    return frozen["settings"]["celeba"][split_point]["methods"][method]["selected_value"]


def validate_semantic_checkpoint(
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    provenance: Mapping[str, str],
    frozen: Mapping[str, Any],
    revision_root: Path,
) -> None:
    result = record["result"]
    identity = {
        "schema_version": SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "semantic_probe_checkpoint",
        "dataset": "celeba",
        "dataset_metadata": EXPECTED_DATASET_METADATA,
        "split_point": result["split_point"],
        "seed": result["seed"],
        "method": result["method"],
        "feature_shape": list(contract["split_points"][result["split_point"]]),
        "selected_defense_value": _selected_value(frozen, result["split_point"], result["method"]),
        "selection_provenance": dict(provenance),
        "target_names": list(SEMANTIC_TARGET_NAMES),
        "target_indices": list(SEMANTIC_TARGET_INDICES),
        "test_example_count": len(CANONICAL_TEST_EXAMPLE_IDS),
        "semantic_draw_seeds": list(TEST_DRAW_SEEDS if result["method"] == "laplace" else TEST_DRAW_SEEDS[:1]),
    }
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise ResultAuditError(f"semantic checkpoint metadata mismatch for {key}")
    if payload.get("defense") != result.get("defense"):
        raise ResultAuditError("semantic checkpoint defense metadata differs from the manifest")
    if payload.get("primary_checkpoint_hashes") != result.get("primary_checkpoint_hashes"):
        raise ResultAuditError("semantic checkpoint primary dependency metadata differs from the manifest")
    if payload.get("raw_bundle_path") != result.get("raw_bundle_path"):
        raise ResultAuditError("semantic checkpoint raw-bundle path differs from the manifest")
    for key in ("selected_epoch", "best_validation_loss", "history"):
        if payload.get(key) != result.get(key):
            raise ResultAuditError(f"semantic checkpoint training metadata differs from the manifest for {key}")
    validate_semantic_history(payload, contract)
    try:
        probe = MultilabelProbe(tuple(identity["feature_shape"]))
        probe.load_state_dict(payload["model_state_dict"], strict=True)
        del probe
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ResultAuditError(f"incompatible semantic checkpoint state: {error}") from error
    dependencies = payload.get("primary_checkpoint_hashes")
    expected_roles = {"encoder", "learned_perturbation"} if result["method"] == "learned" else {"encoder"}
    if not isinstance(dependencies, Mapping) or set(dependencies) != expected_roles:
        raise ResultAuditError("semantic primary dependency roles are incomplete or unexpected")
    for role, dependency in dependencies.items():
        if not isinstance(dependency, Mapping):
            raise ResultAuditError(f"semantic {role} dependency metadata is malformed")
        path = resolve_artifact_path(str(dependency.get("path", "")), revision_root)
        if not path.is_file() or sha256_file(path) != dependency.get("sha256"):
            raise ResultAuditError(f"semantic {role} dependency is missing or hash-inconsistent")


def _discover_semantic_artifacts(root: Path) -> dict[Path, str]:
    artifacts: dict[Path, str] = {}
    for path in (root / "manifests" / "semantic").glob("*.json"):
        artifacts[path.resolve()] = "semantic_manifest"
    for path in (root / "checkpoints" / "semantic").rglob("*.pt") if (root / "checkpoints" / "semantic").is_dir() else []:
        artifacts[path.resolve()] = "semantic_checkpoint"
    for path in (root / "checkpoints" / "semantic").rglob("*.npz") if (root / "checkpoints" / "semantic").is_dir() else []:
        artifacts[path.resolve()] = "semantic_raw_bundle"
    return artifacts


def audit_semantic_results(contract: Mapping[str, Any], revision_root: str | Path) -> dict[str, Any]:
    root = Path(revision_root).resolve()
    frozen, provenance, _ = load_sealed_selection(contract=dict(contract), revision_root=root)
    expected_specs = planned_semantic_specs(dict(contract))
    manifest_paths = sorted((root / "manifests" / "semantic").glob("*.json"))
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def issue(gate: str, message: str, run_id: str | None = None) -> None:
        entry: dict[str, Any] = {"gate": gate, "message": message}
        if run_id is not None:
            entry["run_id"] = run_id
        issues.append(entry)

    for path in manifest_paths:
        try:
            payload = strict_json_load(path)
            if not isinstance(payload, dict):
                raise ResultAuditError("semantic manifest root is not an object")
            records.append(payload)
        except (OSError, TypeError, ValueError, ResultAuditError) as error:
            issue("registry", f"{path}: {error}")
    for message in validate_registry_records(records, expected_specs):
        issue("registry", message)

    expected_artifacts: dict[Path, str] = {
        path.resolve(): "semantic_manifest" for path in manifest_paths
    }
    seed_rows: list[dict[str, Any]] = []
    attribute_rows: list[dict[str, Any]] = []
    targets_reference: np.ndarray | None = None
    example_ids_reference: np.ndarray | None = None
    checkpoint_verified = 0
    raw_verified = 0
    history_verified = 0
    dependency_paths: set[Path] = set()

    for record in records:
        run_id = str(record.get("run_id", ""))
        if record.get("status") != "complete" or not isinstance(record.get("result"), Mapping):
            continue
        result = record["result"]
        try:
            if record.get("selection_provenance") != provenance:
                raise ResultAuditError("semantic manifest provenance differs from the sealed selection")
            expected_identity = {
                "dataset": "celeba",
                "dataset_metadata": EXPECTED_DATASET_METADATA,
                "split_point": record["spec"]["split_point"],
                "seed": record["spec"]["seed"],
                "method": record["spec"]["method"],
                "selected_defense_value": _selected_value(
                    frozen, record["spec"]["split_point"], record["spec"]["method"]
                ),
            }
            for key, expected in expected_identity.items():
                if result.get(key) != expected:
                    raise ResultAuditError(f"semantic manifest result metadata mismatch for {key}")
            if result.get("attribute_count") != 39 or result.get("test_example_count") != 19_962:
                raise ResultAuditError("semantic manifest reports the wrong target/example count")
            if result.get("target_names") != list(SEMANTIC_TARGET_NAMES) or result.get("target_indices") != list(SEMANTIC_TARGET_INDICES):
                raise ResultAuditError("semantic manifest target identity/order is wrong")
            expected_draws = list(TEST_DRAW_SEEDS if result["method"] == "laplace" else TEST_DRAW_SEEDS[:1])
            if result.get("semantic_draw_seeds") != expected_draws or result.get("semantic_draw_count") != len(expected_draws):
                raise ResultAuditError("semantic manifest draw policy is wrong")
            if any(
                result.get(field) is not False
                for field in (
                    "mc_draws_are_independent_replicates",
                    "attributes_are_independent_replicates",
                    "images_are_independent_replicates",
                )
            ):
                raise ResultAuditError("semantic manifest promotes a non-seed dimension to a replicate")
        except (KeyError, TypeError, ValueError, ResultAuditError) as error:
            issue("record_metadata", str(error), run_id)

        checkpoint_path: Path | None = None
        raw_path: Path | None = None
        try:
            checkpoint_path = resolve_artifact_path(str(result.get("checkpoint_path", "")), root)
            expected_artifacts[checkpoint_path] = "semantic_checkpoint"
            if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != result.get("checkpoint_sha256"):
                raise ResultAuditError("semantic checkpoint is missing or its manifest hash differs")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, Mapping):
                raise ResultAuditError("semantic checkpoint root is not a mapping")
            validate_semantic_checkpoint(
                checkpoint,
                contract=contract,
                record=record,
                provenance=provenance,
                frozen=frozen,
                revision_root=root,
            )
            checkpoint_verified += 1
            history_verified += 1
            for dependency in checkpoint["primary_checkpoint_hashes"].values():
                dependency_paths.add(resolve_artifact_path(str(dependency["path"]), root))
        except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
            issue("checkpoint_integrity", str(error), run_id)

        try:
            raw_path = resolve_artifact_path(str(result.get("raw_bundle_path", "")), root)
            expected_artifacts[raw_path] = "semantic_raw_bundle"
            if not raw_path.is_file() or sha256_file(raw_path) != result.get("raw_bundle_sha256"):
                raise ResultAuditError("semantic raw bundle is missing or its manifest hash differs")
            arrays = load_semantic_raw_bundle(raw_path)
            recomputed, per_attribute = validate_semantic_raw_arrays(arrays, method=str(result["method"]))
            for name, value in recomputed.items():
                _require_equal_metric(value, result.get(name), name)
            current_targets = np.asarray(arrays["binary_targets"])
            current_ids = np.asarray(arrays["example_ids"])
            if targets_reference is not None and example_ids_reference is not None:
                validate_cross_bundle_identity(
                    targets_reference,
                    example_ids_reference,
                    current_targets,
                    current_ids,
                )
            targets_reference = current_targets.copy() if targets_reference is None else targets_reference
            example_ids_reference = current_ids.copy() if example_ids_reference is None else example_ids_reference
            raw_verified += 1
            for row in per_attribute:
                attribute_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": "celeba",
                        "split_point": result["split_point"],
                        "seed": result["seed"],
                        "method": result["method"],
                        **row,
                    }
                )
            seed_rows.append(
                {
                    "run_id": run_id,
                    "dataset": "celeba",
                    "split_point": result["split_point"],
                    "seed": result["seed"],
                    "method": result["method"],
                    "selected_defense_value": result["selected_defense_value"],
                    "macro_auroc": recomputed["macro_auroc"],
                    "macro_balanced_accuracy": recomputed["macro_balanced_accuracy"],
                    "attribute_auroc_sample_sd": recomputed["auroc_sample_sd"],
                    "attribute_balanced_accuracy_sample_sd": recomputed["balanced_accuracy_sample_sd"],
                    "semantic_draw_count": recomputed["semantic_draw_count"],
                    "semantic_draw_seeds": ",".join(map(str, recomputed["semantic_draw_seeds"])),
                    "record_passed": True,
                }
            )
        except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
            issue("raw_integrity", str(error), run_id)

    actual_artifacts = _discover_semantic_artifacts(root)
    missing = sorted(str(path) for path in set(expected_artifacts) - set(actual_artifacts))
    unexpected = sorted(str(path) for path in set(actual_artifacts) - set(expected_artifacts))
    expected_counts = Counter(expected_artifacts.values())
    actual_counts = Counter(actual_artifacts.values())
    required_counts = Counter(
        {"semantic_manifest": 40, "semantic_checkpoint": 40, "semantic_raw_bundle": 40}
    )
    if expected_counts != required_counts:
        issue("artifact_inventory", f"referenced semantic artifact counts are {dict(expected_counts)}")
    if actual_counts != required_counts:
        issue("artifact_inventory", f"discovered semantic artifact counts are {dict(actual_counts)}")
    if missing:
        issue("artifact_inventory", f"missing referenced semantic artifacts: {missing}")
    if unexpected:
        issue("artifact_inventory", f"unreferenced semantic artifacts: {unexpected}")

    raw_evidence = [
        {
            "artifact_type": kind,
            "path": str(path.relative_to(root.parent.parent)),
            "size_bytes": path.stat().st_size,
            "mode": f"{path.stat().st_mode & 0o777:04o}",
            "sha256": sha256_file(path),
        }
        for path, kind in sorted(actual_artifacts.items(), key=lambda item: str(item[0]))
    ]
    seed_rows.sort(key=lambda row: (row["split_point"], row["method"], row["seed"]))
    attribute_rows.sort(
        key=lambda row: (row["split_point"], row["method"], row["seed"], row["attribute_index"])
    )
    if len(seed_rows) != 40:
        issue("metric_cardinality", f"semantic seed row count is {len(seed_rows)}, expected 40")
    if len(attribute_rows) != 1_560:
        issue("metric_cardinality", f"semantic attribute row count is {len(attribute_rows)}, expected 1560")
    if len(dependency_paths) != 30:
        issue("primary_dependencies", f"semantic dependency path count is {len(dependency_paths)}, expected 30")

    gates = {
        name: {
            "pass": not any(entry["gate"] == name for entry in issues),
            "error_count": sum(entry["gate"] == name for entry in issues),
        }
        for name in (
            "registry",
            "record_metadata",
            "checkpoint_integrity",
            "raw_integrity",
            "artifact_inventory",
            "metric_cardinality",
            "primary_dependencies",
        )
    }
    quarantined = sorted({entry["run_id"] for entry in issues if entry.get("run_id")})
    return {
        "schema_version": 1,
        "audit": "completed_semantic_results",
        "passed": not issues,
        "outcome_neutrality": (
            "Integrity does not depend on leakage magnitude, method ordering, effect size, or significance."
        ),
        "contract_matrix": {
            "expected_records": 40,
            "registered_records": len(records),
            "dataset": "celeba",
            "split_points": list(contract["split_points"]),
            "seeds": list(contract["seeds"]),
            "methods": list(contract["methods"]),
            "target_count": 39,
            "test_example_count": 19_962,
        },
        "selection_provenance": provenance,
        "gates": gates,
        "verification_counts": {
            "semantic_manifests_verified": sum(record.get("status") == "complete" for record in records),
            "semantic_checkpoints_verified": checkpoint_verified,
            "semantic_histories_verified": history_verified,
            "semantic_raw_bundles_verified": raw_verified,
            "semantic_seed_rows": len(seed_rows),
            "semantic_attribute_rows": len(attribute_rows),
            "semantic_artifacts_hashed": len(raw_evidence),
            "unique_primary_dependencies_verified": len(dependency_paths),
            "laplace_five_draw_bundles": sum(row["method"] == "laplace" and row["semantic_draw_count"] == 5 for row in seed_rows),
            "single_draw_bundles": sum(row["method"] != "laplace" and row["semantic_draw_count"] == 1 for row in seed_rows),
        },
        "artifact_counts": dict(sorted(actual_counts.items())),
        "quarantined_run_ids": quarantined,
        "issues": issues,
        "semantic_seed_results": seed_rows,
        "semantic_attribute_results": attribute_rows,
        "raw_evidence": raw_evidence,
    }


def require_canonical_semantic_audit(
    contract: Mapping[str, Any], revision_root: str | Path, audit_path: str | Path
) -> dict[str, Any]:
    """Verify the canonical audit and its 120 live evidence hashes before budget setup."""
    path = Path(audit_path)
    if not path.is_file():
        raise RuntimeError(f"Budget is blocked: missing canonical semantic audit {path}")
    canonical = strict_json_load(path)
    fresh = audit_semantic_results(contract, revision_root)
    required_counts = {
        "semantic_manifests_verified": 40,
        "semantic_checkpoints_verified": 40,
        "semantic_raw_bundles_verified": 40,
        "semantic_artifacts_hashed": 120,
    }
    if (
        canonical.get("passed") is not True
        or fresh.get("passed") is not True
        or canonical.get("issues")
        or fresh.get("issues")
        or canonical.get("quarantined_run_ids")
        or fresh.get("quarantined_run_ids")
        or any(canonical.get("verification_counts", {}).get(key) != value for key, value in required_counts.items())
        or canonical.get("raw_evidence") != fresh.get("raw_evidence")
    ):
        raise RuntimeError("Budget is blocked: semantic audit is failed, incomplete, quarantined, or hash-stale")
    return fresh
