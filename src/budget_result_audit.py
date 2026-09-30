"""Fail-closed, outcome-neutral audit of the fixed auxiliary-budget evidence."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .experiment import ATTACKER_ARCHITECTURES, TEST_DRAW_SEEDS
from .manifest import deterministic_run_id
from .metrics import METRIC_FIELDS, metric_versions
from .pipeline import (
    BUDGET_ARTIFACT_SCHEMA_VERSION,
    BUDGET_PROTOCOL_VERSION,
    BUDGET_REGISTRY_DIRECTORY,
    BUDGET_SPLIT_POLICY,
    BUDGET_SUBSET_SEED,
    EXPECTED_BUDGET_RUN_ID_DIGEST,
    _budget_run_id_digest,
    budget_artifact_paths,
    planned_budget_specs,
    planned_primary_specs,
)
from .result_audit import (
    EXPECTED_DATASET_METADATA,
    ResultAuditError,
    resolve_artifact_path,
    sha256_file,
    strict_json_load,
    validate_attacker_checkpoint,
    validate_metric_payload,
    validate_registry_records,
    validate_training_history,
)
from .selection_provenance import load_sealed_selection


EXPECTED_ARTIFACT_COUNTS = {
    "budget_manifest": 24,
    "budget_attacker_checkpoint": 48,
    "budget_training_log": 48,
    "budget_metric_json": 48,
    "budget_sample_bundle": 48,
}
EXPECTED_AUXILIARY_IDENTITIES: dict[str, dict[float, dict[str, Any]]] = {
    "celeba": {
        0.1: {
            "auxiliary_examples": 16_277,
            "auxiliary_fitting_examples": 14_650,
            "auxiliary_validation_examples": 1_627,
            "auxiliary_index_digest": "156900f031dedf0d1022783577d358779f09c6ff67e87052f95c25540739b349",
            "auxiliary_fitting_index_digest": "adac474174f8e1ab475a35a9973c83593254c8c2adf511abc60ffab1b090efaf",
            "auxiliary_validation_index_digest": "7bd1f5606ca409718ec20fc416a5a0aa7fe116bc4d765b2ddb366d40a6dca6a5",
        },
        0.5: {
            "auxiliary_examples": 81_385,
            "auxiliary_fitting_examples": 73_247,
            "auxiliary_validation_examples": 8_138,
            "auxiliary_index_digest": "84fe6e82193dffabaaa452e5270d0f02069ee248eca1511ed93a07afad614b13",
            "auxiliary_fitting_index_digest": "ace9d1f949a8a66371ee5a930695aa47e60bb1fdd7becf64bae359d550007b5f",
            "auxiliary_validation_index_digest": "61db04f3d5bbfb57b0199d7af34e5fc5d53d9f37616112613be0dfed25315bbf",
        },
        1.0: {
            "auxiliary_examples": 162_770,
            "auxiliary_fitting_examples": 146_493,
            "auxiliary_validation_examples": 16_277,
            "auxiliary_index_digest": "edfa55bb0923c44f0c597626c32a393bb3fe7da3294e275754f10d90ed71abda",
            "auxiliary_fitting_index_digest": "e4cff83b8a271b0010ce06007f30074cd3e40a96cabc9d9ff80fd708695319af",
            "auxiliary_validation_index_digest": "bd15f010564007b562b6c1b1dd10ab31a8bde53c5c18e30dc03cdb60b987f249",
        },
    },
    "cifar10": {
        0.1: {
            "auxiliary_examples": 4_500,
            "auxiliary_fitting_examples": 4_050,
            "auxiliary_validation_examples": 450,
            "auxiliary_index_digest": "0ed7cbbeefe70c074e2fe86db11312e23241845ecd458285380336d6edb4f839",
            "auxiliary_fitting_index_digest": "375cd5d2bcb5179da7dfd60ce23f0e7e0a8a7d9152a51f21ce8122109ed83151",
            "auxiliary_validation_index_digest": "63064c622295499de47ee7071ce445508d4fca52b67817488ddbae984b369c4a",
        },
        0.5: {
            "auxiliary_examples": 22_500,
            "auxiliary_fitting_examples": 20_250,
            "auxiliary_validation_examples": 2_250,
            "auxiliary_index_digest": "a84854c93554fe947a9706247765b94d745e0e5f1017c36af07cc7a0711f7fba",
            "auxiliary_fitting_index_digest": "c87d917e9b4051f9d0716b2ab5659e68f4921f4bd963b1c911ad4ad659dcad69",
            "auxiliary_validation_index_digest": "80fb430ec3a1ae4b996352c02a21f9360429dc7744388a0c91c0923b49aae384",
        },
        1.0: {
            "auxiliary_examples": 45_000,
            "auxiliary_fitting_examples": 40_500,
            "auxiliary_validation_examples": 4_500,
            "auxiliary_index_digest": "7f200f123476452d699baec781de16672c7723cba6ef109bf204453ebaffc052",
            "auxiliary_fitting_index_digest": "15283f73605ea4aac3829c42f994b91d1ffd2dfe3526df38510df84bd45b00db",
            "auxiliary_validation_index_digest": "3ad326bc7676a33dfa5c77e27904ca716a0a4a4df834da197625edc34eeed2b4",
        },
    },
}
COMMON_IDENTITY_KEYS = (
    "budget_artifact_schema_version",
    "budget_protocol_version",
    "run_id",
    "dataset",
    "split_point",
    "seed",
    "method",
    "selected_defense_value",
    "defense",
    "auxiliary_fraction",
    "dataset_size",
    "auxiliary_examples",
    "auxiliary_fitting_examples",
    "auxiliary_validation_examples",
    "auxiliary_index_seed",
    "auxiliary_split_policy",
    "auxiliary_index_digest",
    "auxiliary_fitting_index_digest",
    "auxiliary_validation_index_digest",
    "auxiliary_index_digest_algorithm",
    "ordinary_validation_access",
    "primary_encoder",
    "selection_provenance",
    "evaluation_partition",
    "metric_versions",
    "model_seeds_are_independent_replicates",
    "attackers_are_independent_replicates",
    "images_are_independent_replicates",
    "auxiliary_examples_are_independent_replicates",
)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultAuditError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ResultAuditError(f"{label} is non-finite")
    return number


def expected_auxiliary_identity(dataset: str, fraction: float) -> dict[str, Any]:
    if dataset not in EXPECTED_AUXILIARY_IDENTITIES or fraction not in EXPECTED_AUXILIARY_IDENTITIES[dataset]:
        raise ResultAuditError(f"unexpected auxiliary-budget cell {dataset}/{fraction}")
    return {
        "dataset_size": EXPECTED_DATASET_METADATA[dataset]["train_size"],
        "auxiliary_fraction": fraction,
        **EXPECTED_AUXILIARY_IDENTITIES[dataset][fraction],
        "auxiliary_index_seed": BUDGET_SUBSET_SEED,
        "auxiliary_split_policy": BUDGET_SPLIT_POLICY,
        "auxiliary_index_digest_algorithm": "sha256-int64le-v1",
        "ordinary_validation_access": False,
    }


def validate_budget_common_identity(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in COMMON_IDENTITY_KEYS:
        if payload.get(key) != expected.get(key):
            raise ResultAuditError(f"budget artifact metadata mismatch for {key}")
    if payload.get("budget_artifact_schema_version") != BUDGET_ARTIFACT_SCHEMA_VERSION:
        raise ResultAuditError("budget artifact schema version is incompatible")
    if payload.get("budget_protocol_version") != BUDGET_PROTOCOL_VERSION:
        raise ResultAuditError("budget protocol version is incompatible")
    if payload.get("ordinary_validation_access") is not False:
        raise ResultAuditError("budget artifact reports ordinary validation access")
    if payload.get("auxiliary_examples") != (
        payload.get("auxiliary_fitting_examples", 0) + payload.get("auxiliary_validation_examples", 0)
    ):
        raise ResultAuditError("budget fitting and validation counts do not sum to the total cap")
    for field in (
        "model_seeds_are_independent_replicates",
        "attackers_are_independent_replicates",
        "images_are_independent_replicates",
        "auxiliary_examples_are_independent_replicates",
    ):
        if payload.get(field) is not False:
            raise ResultAuditError(f"budget artifact promotes a non-replicate dimension: {field}")


def validate_budget_sample_bundle(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any], architecture: str
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_budget_common_identity(payload, expected)
    if payload.get("artifact_type") != "budget_sample_bundle" or payload.get("architecture") != architecture:
        raise ResultAuditError("budget sample-bundle identity is wrong")
    original = payload.get("original")
    reconstruction = payload.get("reconstruction")
    if not isinstance(original, torch.Tensor) or not isinstance(reconstruction, torch.Tensor):
        raise ResultAuditError("budget sample bundle images are not tensors")
    if tuple(original.shape) != (5, 3, 32, 32) or tuple(reconstruction.shape) != (5, 3, 32, 32):
        raise ResultAuditError("budget sample bundle tensors are not 5x3x32x32")
    if not torch.isfinite(original).all() or not torch.isfinite(reconstruction).all():
        raise ResultAuditError("budget sample bundle contains non-finite values")
    if payload.get("draw_seed") != TEST_DRAW_SEEDS[0]:
        raise ResultAuditError("budget sample bundle uses the wrong fixed draw seed")
    return original, reconstruction


def _discover_budget_artifacts(root: Path) -> dict[Path, str]:
    artifacts: dict[Path, str] = {}
    for path in (root / "manifests" / BUDGET_REGISTRY_DIRECTORY).glob("*.json"):
        artifacts[path.resolve()] = "budget_manifest"
    for path in (root / "checkpoints" / BUDGET_REGISTRY_DIRECTORY).rglob("*") if (root / "checkpoints" / BUDGET_REGISTRY_DIRECTORY).is_dir() else []:
        if path.is_file():
            artifacts[path.resolve()] = (
                "budget_attacker_checkpoint" if path.suffix == ".pt" else "unexpected_budget_artifact"
            )
    for path in (root / "logs" / BUDGET_REGISTRY_DIRECTORY).rglob("*") if (root / "logs" / BUDGET_REGISTRY_DIRECTORY).is_dir() else []:
        if path.is_file():
            artifacts[path.resolve()] = (
                "budget_training_log" if path.suffix == ".json" else "unexpected_budget_artifact"
            )
    for path in (root / "metrics" / BUDGET_REGISTRY_DIRECTORY).rglob("*") if (root / "metrics" / BUDGET_REGISTRY_DIRECTORY).is_dir() else []:
        if not path.is_file():
            continue
        if path.name.endswith("_samples.pt"):
            kind = "budget_sample_bundle"
        elif path.suffix == ".json":
            kind = "budget_metric_json"
        else:
            kind = "unexpected_budget_artifact"
        artifacts[path.resolve()] = kind
    return artifacts


def _load_primary_records(root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    records = [strict_json_load(path) for path in sorted((root / "manifests" / "primary").glob("*.json"))]
    return {
        (
            str(record["spec"]["dataset"]),
            str(record["spec"]["split_point"]),
            int(record["spec"]["seed"]),
            str(record["spec"]["method"]),
        ): record
        for record in records
    }


def audit_budget_results(contract: Mapping[str, Any], revision_root: str | Path) -> dict[str, Any]:
    root = Path(revision_root).resolve()
    frozen, provenance, _ = load_sealed_selection(contract=dict(contract), revision_root=root)
    expected_specs = planned_budget_specs(dict(contract))
    expected_ids = [deterministic_run_id(spec) for spec in expected_specs]
    manifest_paths = sorted((root / "manifests" / BUDGET_REGISTRY_DIRECTORY).glob("*.json"))
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def issue(gate: str, message: str, run_id: str | None = None) -> None:
        entry: dict[str, Any] = {"gate": gate, "message": message}
        if run_id:
            entry["run_id"] = run_id
        issues.append(entry)

    if len(expected_specs) != 24 or _budget_run_id_digest(expected_ids) != EXPECTED_BUDGET_RUN_ID_DIGEST:
        issue("registry", "the current contract does not reproduce the frozen 24-cell run-ID digest")
    for path in manifest_paths:
        try:
            record = strict_json_load(path)
            if not isinstance(record, dict):
                raise ResultAuditError("budget manifest root is not an object")
            records.append(record)
        except (OSError, TypeError, ValueError, ResultAuditError) as error:
            issue("registry", f"{path}: {error}")
    for message in validate_registry_records(records, expected_specs):
        issue("registry", message)

    primary_records = _load_primary_records(root)
    primary_expected_ids = {deterministic_run_id(spec) for spec in planned_primary_specs(dict(contract))}
    if len(primary_records) != 80 or {
        record.get("run_id") for record in primary_records.values()
    } != primary_expected_ids:
        issue("primary_dependencies", "primary registry identity is not the exact fixed 80-cell matrix")

    expected_artifacts: dict[Path, str] = {
        path.resolve(): "budget_manifest" for path in manifest_paths
    }
    cell_rows: list[dict[str, Any]] = []
    attacker_rows: list[dict[str, Any]] = []
    checkpoint_verified = 0
    log_verified = 0
    metric_verified = 0
    sample_verified = 0
    history_verified = 0
    dependency_paths: set[Path] = set()

    for record in records:
        run_id = str(record.get("run_id", ""))
        if record.get("status") != "complete" or not isinstance(record.get("result"), Mapping):
            continue
        result = record["result"]
        try:
            spec = record["spec"]
            dataset = str(spec["dataset"])
            split_point = str(spec["split_point"])
            seed = int(spec["seed"])
            method = str(spec["method"])
            fraction = float(spec["auxiliary_fraction"])
            if seed != int(contract["selection_seed"]) or seed != 42:
                raise ResultAuditError("budget seed differs from the single predeclared seed 42")
            primary = primary_records[(dataset, split_point, seed, method)]
            if primary.get("status") != "complete" or primary.get("selection_provenance") != provenance:
                raise ResultAuditError("referenced primary cell is incomplete or provenance-stale")
            primary_result = primary["result"]
            encoder_path = resolve_artifact_path(str(primary_result["encoder_checkpoint"]), root)
            encoder_identity = {"path": str(Path(primary_result["encoder_checkpoint"])), "sha256": sha256_file(encoder_path)}
            subset = expected_auxiliary_identity(dataset, fraction)
            selected_value = primary_result.get("selected_defense_value")
            expected_common = {
                "budget_artifact_schema_version": BUDGET_ARTIFACT_SCHEMA_VERSION,
                "budget_protocol_version": BUDGET_PROTOCOL_VERSION,
                "run_id": run_id,
                "dataset": dataset,
                "split_point": split_point,
                "seed": seed,
                "method": method,
                "selected_defense_value": selected_value,
                "defense": {"method": method, "stochastic": False},
                "auxiliary_fraction": fraction,
                **subset,
                "primary_encoder": encoder_identity,
                "selection_provenance": provenance,
                "evaluation_partition": "test_after_frozen_selection",
                "metric_versions": metric_versions(),
                "model_seeds_are_independent_replicates": False,
                "attackers_are_independent_replicates": False,
                "images_are_independent_replicates": False,
                "auxiliary_examples_are_independent_replicates": False,
            }
            validate_budget_common_identity(result, expected_common)
            if record.get("selection_provenance") != provenance:
                raise ResultAuditError("budget manifest provenance differs from the sealed selection")
            if result.get("dataset_metadata") != EXPECTED_DATASET_METADATA[dataset]:
                raise ResultAuditError("budget dataset metadata differs from the real fixed partitions")
            dependency_paths.add(encoder_path)
        except (KeyError, TypeError, ValueError, ResultAuditError) as error:
            issue("record_metadata", str(error), run_id)
            continue

        attacker_summaries: dict[str, dict[str, float]] = {}
        attackers = result.get("attackers")
        if not isinstance(attackers, Mapping) or set(attackers) != set(ATTACKER_ARCHITECTURES):
            issue("record_metadata", "budget attacker family is incomplete or unexpected", run_id)
            continue
        originals_reference: torch.Tensor | None = None
        for architecture in ATTACKER_ARCHITECTURES:
            payload = attackers[architecture]
            if not isinstance(payload, Mapping):
                issue("record_metadata", f"{architecture}: attacker record is malformed", run_id)
                continue
            paths = budget_artifact_paths(root, record["spec"], architecture)
            for role, path in paths.items():
                expected_artifacts[path.resolve()] = {
                    "checkpoint": "budget_attacker_checkpoint",
                    "training_log": "budget_training_log",
                    "metric_json": "budget_metric_json",
                    "sample_bundle": "budget_sample_bundle",
                }[role]
                declared_key = {
                    "checkpoint": "checkpoint_path",
                    "training_log": "training_log_path",
                    "metric_json": "metric_path",
                    "sample_bundle": "sample_bundle_path",
                }[role]
                hash_key = declared_key.replace("_path", "_sha256")
                try:
                    declared = resolve_artifact_path(str(payload.get(declared_key, "")), root)
                    if declared != path.resolve() or not declared.is_file() or sha256_file(declared) != payload.get(hash_key):
                        raise ResultAuditError(f"{architecture} {role} path/hash mismatch")
                except (OSError, TypeError, ValueError, ResultAuditError) as error:
                    issue("artifact_hashes", str(error), run_id)
            training = payload.get("training")
            try:
                if not isinstance(training, Mapping):
                    raise ResultAuditError("training metadata is missing")
                validate_training_history(training, contract)
                log_payload = strict_json_load(paths["training_log"])
                if log_payload != training:
                    raise ResultAuditError("training log differs from the manifest training record")
                validate_budget_common_identity(log_payload, expected_common)
                if log_payload.get("architecture") != architecture:
                    raise ResultAuditError("training log architecture is wrong")
                log_verified += 1
                history_verified += 1
            except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
                issue("training_log", f"{architecture}: {error}", run_id)
            try:
                checkpoint = validate_attacker_checkpoint(
                    paths["checkpoint"],
                    architecture=architecture,
                    feature_shape=tuple(contract["split_points"][split_point]),
                    seed=seed,
                    training=training,
                    contract=contract,
                )
                validate_budget_common_identity(checkpoint, expected_common)
                if checkpoint.get("architecture") != architecture:
                    raise ResultAuditError("checkpoint architecture is wrong")
                checkpoint_verified += 1
            except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
                issue("checkpoint_integrity", f"{architecture}: {error}", run_id)
            try:
                metric = strict_json_load(paths["metric_json"])
                validate_budget_common_identity(metric, expected_common)
                if metric.get("artifact_type") != "budget_per_image_metrics" or metric.get("architecture") != architecture:
                    raise ResultAuditError("metric artifact identity is wrong")
                validate_metric_payload(
                    metric,
                    example_count=int(EXPECTED_DATASET_METADATA[dataset]["test_size"]),
                    draw_seeds=[TEST_DRAW_SEEDS[0]],
                    manifest_summary=payload.get("test_summary", {}),
                    expected_versions=metric_versions(),
                )
                metric_verified += 1
                attacker_summaries[architecture] = {
                    metric_name: _finite(metric["summary"][metric_name], f"{architecture}.{metric_name}")
                    for metric_name in METRIC_FIELDS
                }
            except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
                issue("metric_integrity", f"{architecture}: {error}", run_id)
            try:
                sample = torch.load(paths["sample_bundle"], map_location="cpu", weights_only=True)
                if not isinstance(sample, Mapping):
                    raise ResultAuditError("sample bundle root is not a mapping")
                original, _ = validate_budget_sample_bundle(
                    sample, expected=expected_common, architecture=architecture
                )
                if originals_reference is not None and not torch.equal(originals_reference, original):
                    raise ResultAuditError("matched sample originals differ across attackers")
                originals_reference = original if originals_reference is None else originals_reference
                sample_verified += 1
            except (OSError, KeyError, TypeError, ValueError, ResultAuditError) as error:
                issue("sample_integrity", f"{architecture}: {error}", run_id)
            summary = payload.get("test_summary", {})
            attacker_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split_point": split_point,
                    "seed": seed,
                    "method": method,
                    "auxiliary_fraction": fraction,
                    "auxiliary_examples": expected_common["auxiliary_examples"],
                    "auxiliary_fitting_examples": expected_common["auxiliary_fitting_examples"],
                    "auxiliary_validation_examples": expected_common["auxiliary_validation_examples"],
                    "auxiliary_index_digest": expected_common["auxiliary_index_digest"],
                    "auxiliary_fitting_index_digest": expected_common["auxiliary_fitting_index_digest"],
                    "auxiliary_validation_index_digest": expected_common["auxiliary_validation_index_digest"],
                    "architecture": architecture,
                    "selected_epoch": training.get("selected_epoch") if isinstance(training, Mapping) else None,
                    "epochs_completed": training.get("epochs_completed") if isinstance(training, Mapping) else None,
                    "best_validation_loss": training.get("best_validation_loss") if isinstance(training, Mapping) else None,
                    **{name: summary.get(name) for name in METRIC_FIELDS},
                    "record_passed": True,
                }
            )

        try:
            if set(attacker_summaries) != set(ATTACKER_ARCHITECTURES):
                raise ResultAuditError("both attacker metric payloads were not verified")
            strongest = max(attacker_summaries, key=lambda name: attacker_summaries[name]["ssim"])
            worst = attacker_summaries[strongest]["ssim"]
            if result.get("strongest_attacker") != strongest or _finite(
                result.get("worst_case_test_ssim"), "worst_case_test_ssim"
            ) != worst:
                raise ResultAuditError("strongest attacker or worst-case SSIM is inconsistent")
            cell_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split_point": split_point,
                    "seed": seed,
                    "method": method,
                    "auxiliary_fraction": fraction,
                    "auxiliary_examples": expected_common["auxiliary_examples"],
                    "auxiliary_fitting_examples": expected_common["auxiliary_fitting_examples"],
                    "auxiliary_validation_examples": expected_common["auxiliary_validation_examples"],
                    "auxiliary_index_digest": expected_common["auxiliary_index_digest"],
                    "auxiliary_fitting_index_digest": expected_common["auxiliary_fitting_index_digest"],
                    "auxiliary_validation_index_digest": expected_common["auxiliary_validation_index_digest"],
                    "primary_encoder_path": expected_common["primary_encoder"]["path"],
                    "primary_encoder_sha256": expected_common["primary_encoder"]["sha256"],
                    "deconv_mse_ssim": attacker_summaries["deconv_mse"]["ssim"],
                    "residual_lpips_ssim": attacker_summaries["residual_lpips"]["ssim"],
                    "strongest_attacker": strongest,
                    "worst_case_test_ssim": worst,
                    "record_passed": True,
                }
            )
        except (KeyError, TypeError, ValueError, ResultAuditError) as error:
            issue("strongest_attacker", str(error), run_id)

    actual_artifacts = _discover_budget_artifacts(root)
    expected_counts = Counter(expected_artifacts.values())
    actual_counts = Counter(actual_artifacts.values())
    required_counts = Counter(EXPECTED_ARTIFACT_COUNTS)
    missing = sorted(str(path) for path in set(expected_artifacts) - set(actual_artifacts))
    unexpected = sorted(str(path) for path in set(actual_artifacts) - set(expected_artifacts))
    if expected_counts != required_counts:
        issue("artifact_inventory", f"referenced budget artifact counts are {dict(expected_counts)}")
    if actual_counts != required_counts:
        issue("artifact_inventory", f"discovered budget artifact counts are {dict(actual_counts)}")
    if missing:
        issue("artifact_inventory", f"missing referenced budget artifacts: {missing}")
    if unexpected:
        issue("artifact_inventory", f"unreferenced budget artifacts: {unexpected}")
    if len(dependency_paths) != 8:
        issue("primary_dependencies", f"unique seed-42 primary encoder count is {len(dependency_paths)}, expected 8")

    cell_rows.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], row["auxiliary_fraction"]))
    attacker_rows.sort(
        key=lambda row: (
            row["dataset"], row["split_point"], row["method"], row["auxiliary_fraction"], row["architecture"]
        )
    )
    if len(cell_rows) != 24:
        issue("metric_cardinality", f"budget cell row count is {len(cell_rows)}, expected 24")
    if len(attacker_rows) != 48:
        issue("metric_cardinality", f"budget attacker row count is {len(attacker_rows)}, expected 48")

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
    gates = {
        name: {
            "pass": not any(entry["gate"] == name for entry in issues),
            "error_count": sum(entry["gate"] == name for entry in issues),
        }
        for name in (
            "registry",
            "record_metadata",
            "artifact_hashes",
            "training_log",
            "checkpoint_integrity",
            "metric_integrity",
            "sample_integrity",
            "strongest_attacker",
            "artifact_inventory",
            "metric_cardinality",
            "primary_dependencies",
        )
    }
    quarantined = sorted({str(entry["run_id"]) for entry in issues if entry.get("run_id")})
    return {
        "schema_version": 2,
        "audit": "completed_auxiliary_budget_results",
        "budget_protocol_version": BUDGET_PROTOCOL_VERSION,
        "passed": not issues,
        "outcome_neutrality": (
            "Integrity does not depend on monotonicity, method ordering, leakage direction, or favorable outcomes."
        ),
        "assurance_boundary": {
            "subset_identity": (
                "Counts and total/fit/validation index digests are compared with six frozen "
                "auditor constants and do not call the producer's subset function."
            ),
            "ordinary_validation_non_access": (
                "Artifact metadata is checked for ordinary_validation_access=false; actual loader "
                "non-access is supported by code-path tests and the accepted real-data preflight, "
                "not independently observable from completed output artifacts."
            ),
        },
        "analysis_boundary": {
            "seed": 42,
            "fixed_conditions_not_inferential_replicates": True,
            "attackers_are_an_adversary_family_not_replicates": True,
            "images_are_not_replicates": True,
            "auxiliary_examples_are_not_replicates": True,
            "inferential_statistics_allowed": False,
        },
        "contract_matrix": {
            "expected_records": 24,
            "registered_records": len(records),
            "datasets": list(contract["datasets"]),
            "split_points": list(contract["split_points"]),
            "methods": ["standard", "afd"],
            "auxiliary_fractions": list(contract["auxiliary_data_budgets"]),
            "seed": int(contract["selection_seed"]),
            "attacker_architectures": list(ATTACKER_ARCHITECTURES),
            "run_id_digest": _budget_run_id_digest(expected_ids),
        },
        "selection_provenance": provenance,
        "gates": gates,
        "verification_counts": {
            "budget_manifests_verified": sum(record.get("status") == "complete" for record in records),
            "budget_attacker_checkpoints_verified": checkpoint_verified,
            "budget_training_logs_verified": log_verified,
            "budget_training_histories_verified": history_verified,
            "budget_metric_json_verified": metric_verified,
            "budget_sample_bundles_verified": sample_verified,
            "budget_cell_rows": len(cell_rows),
            "budget_attacker_rows": len(attacker_rows),
            "budget_artifacts_hashed": len(raw_evidence),
            "unique_primary_encoders_verified": len(dependency_paths),
        },
        "artifact_counts": dict(sorted(actual_counts.items())),
        "quarantined_run_ids": quarantined,
        "issues": issues,
        "budget_cell_results": cell_rows,
        "budget_attacker_results": attacker_rows,
        "raw_evidence": raw_evidence,
    }


def require_canonical_budget_audit(
    contract: Mapping[str, Any], revision_root: str | Path, audit_path: str | Path
) -> dict[str, Any]:
    path = Path(audit_path)
    if not path.is_file():
        raise RuntimeError(f"Budget-derived outputs are blocked: missing canonical audit {path}")
    canonical = strict_json_load(path)
    fresh = audit_budget_results(contract, revision_root)
    required = {
        "budget_manifests_verified": 24,
        "budget_attacker_checkpoints_verified": 48,
        "budget_training_logs_verified": 48,
        "budget_metric_json_verified": 48,
        "budget_sample_bundles_verified": 48,
        "budget_cell_rows": 24,
        "budget_attacker_rows": 48,
        "budget_artifacts_hashed": 216,
        "unique_primary_encoders_verified": 8,
    }
    if (
        canonical.get("passed") is not True
        or fresh.get("passed") is not True
        or canonical.get("issues")
        or fresh.get("issues")
        or canonical.get("quarantined_run_ids")
        or fresh.get("quarantined_run_ids")
        or any(canonical.get("verification_counts", {}).get(key) != value for key, value in required.items())
        or canonical.get("raw_evidence") != fresh.get("raw_evidence")
        or canonical.get("budget_cell_results") != fresh.get("budget_cell_results")
        or canonical.get("budget_attacker_results") != fresh.get("budget_attacker_results")
    ):
        raise RuntimeError("Budget-derived outputs are blocked: audit is failed, incomplete, quarantined, or hash-stale")
    return fresh
