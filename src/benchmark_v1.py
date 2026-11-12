"""Fail-closed execution for the prospective benchmark-style extension."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .checkpointing import load_split_model
from .dataset import get_budget_pool_and_test_loaders, get_dataset_loaders
from .defenses import FeatureDefense, load_universal_perturbation
from .experiment import (
    ATTACKER_ARCHITECTURES,
    TEST_DRAW_SEEDS,
    evaluate_adaptive_attacker,
    resolve_device,
    train_adaptive_attacker,
)
from .implementation_provenance import local_import_closure
from .manifest import RunRegistry, deterministic_run_id
from .metrics import metric_versions
from .pipeline import (
    _budget_loaders,
    _learned_path,
    _primary_record,
    _sha256_file,
    _json_write_exclusive,
    load_contract,
)
from .selection_provenance import load_sealed_selection, require_selection_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "benchmark_extension_v1.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
DEFAULT_REVISION_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_PREFLIGHT = PROJECT_ROOT / "audit" / "benchmark_revision_20260822" / "compute_preflight_p2.json"
BUDGET_REGISTRY = "benchmark_budget_v1_p2"
RESTART_REGISTRY = "attacker_restarts_v1_p2"
STAGE_MANIFEST = "benchmark_extension_v1_p2_stage.json"
ARTIFACT_SCHEMA_VERSION = 1
REQUIRED_PROVENANCE_FIELDS = {
    "artifact_schema_version",
    "benchmark_protocol_id",
    "benchmark_config_sha256",
    "base_contract_sha256",
    "selection_artifact_sha256",
    "selection_lock_sha256",
    "implementation_closure_sha256",
    "encoder",
    "model_seed",
    "attacker_training_seed",
    "metric_versions",
    "artifact_hashes",
    "subset_identity",
    "images_are_independent_replicates",
    "attackers_are_independent_replicates",
    "attacker_restarts_are_independent_scientific_replicates",
}


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _project_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve(strict=False)
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"Benchmark path escapes the project root: {relative}") from error
    return path


def _require_hash(relative: str, expected: str) -> Path:
    path = _project_path(relative)
    if not path.is_file():
        raise RuntimeError(f"Missing benchmark-bound file: {relative}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Benchmark hash mismatch for {relative}: {actual} != {expected}")
    return path


def _expected_encoder_paths(config: Mapping[str, Any], revision_root: Path) -> set[str]:
    prefix = revision_root.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    return {
        f"{prefix}/checkpoints/encoders/{dataset}/{interface}/seed_{seed}/{method}.pt"
        for dataset in config["datasets"]
        for interface in config["interfaces"]
        for seed in config["model_seeds"]
        for method in ("standard", "afd")
    }


def load_benchmark_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT,
    revision_root: str | Path = DEFAULT_REVISION_ROOT,
) -> dict[str, Any]:
    target = Path(path).resolve()
    config = _strict_json(target)
    if config.get("schema_version") != 1 or config.get("protocol_id") != "benchmark_extension_v1_20260822":
        raise RuntimeError("Unsupported benchmark extension configuration")
    contract = load_contract(contract_path)
    if config.get("datasets") != contract.get("datasets"):
        raise RuntimeError("Benchmark datasets differ from the base contract")
    if config.get("interfaces") != list(contract.get("split_points", {})):
        raise RuntimeError("Benchmark interfaces differ from the base contract")
    if config.get("model_seeds") != contract.get("seeds"):
        raise RuntimeError("Benchmark model seeds differ from the base contract")
    if config.get("metric_versions") != metric_versions():
        raise RuntimeError("Benchmark metric versions differ from the active implementation")
    base = config.get("base_study", {})
    bindings = (
        ("contract_path", "contract_sha256"),
        ("implementation_protocol_path", "implementation_protocol_sha256"),
        ("selection_artifact_path", "selection_artifact_sha256"),
        ("selection_lock_path", "selection_lock_sha256"),
        ("primary_audit_path", "primary_audit_sha256"),
        ("corrected_budget_audit_path", "corrected_budget_audit_sha256"),
        ("pre_change_archive_manifest_path", "pre_change_archive_manifest_sha256"),
        ("pre_change_supplemental_manifest_path", "pre_change_supplemental_manifest_sha256"),
    )
    for path_key, hash_key in bindings:
        _require_hash(str(base.get(path_key, "")), str(base.get(hash_key, "")))
    if _project_path(str(base["contract_path"])) != Path(contract_path).resolve():
        raise RuntimeError("Benchmark base-contract path differs from the requested contract")
    revision = Path(revision_root).resolve()
    encoder_hashes = config.get("encoder_hashes", {})
    expected_encoders = _expected_encoder_paths(config, revision)
    if set(encoder_hashes) != expected_encoders or len(encoder_hashes) != 40:
        raise RuntimeError("Benchmark encoder binding is not the exact 40-checkpoint matrix")
    for relative, expected in sorted(encoder_hashes.items()):
        _require_hash(relative, expected)
    learned = config.get("restart_learned_perturbation_hashes", {})
    if len(learned) != 4:
        raise RuntimeError("Benchmark restart binding is not the exact four learned perturbations")
    for relative, expected in sorted(learned.items()):
        _require_hash(relative, expected)
    budget = config.get("budget_extension", {})
    if (
        budget.get("expected_conditions") != 120
        or budget.get("expected_attacker_rows") != 240
        or budget.get("new_conditions") != 96
        or budget.get("new_attacker_trainings") != 192
        or budget.get("inherited_model_seed") != 42
        or budget.get("attacker_architectures") != list(ATTACKER_ARCHITECTURES)
    ):
        raise RuntimeError("Benchmark budget matrix declaration is inconsistent")
    restarts = config.get("attacker_restarts", {})
    if (
        restarts.get("expected_rows") != 96
        or restarts.get("new_trainings") != 64
        or restarts.get("model_seed") != 42
        or restarts.get("attacker_training_seeds") != [42, 314159, 271828]
        or restarts.get("attacker_architectures") != list(ATTACKER_ARCHITECTURES)
        or restarts.get("subset_identity_policy") != "full_frozen_primary_train_validation_test_partitions_v1"
    ):
        raise RuntimeError("Benchmark attacker-restart matrix declaration is inconsistent")
    gate = config.get("compute_gate", {})
    if (
        gate.get("maximum_projected_gpu_hours") != 144.0
        or gate.get("maximum_projected_storage_gb") != 300.0
        or gate.get("protocol_shrinkage_allowed") is not False
    ):
        raise RuntimeError("Benchmark compute interlock differs from the frozen protocol")
    config["_config_path"] = str(target)
    config["_config_sha256"] = _sha256_file(target)
    return config


def benchmark_implementation_closure() -> dict[str, Any]:
    return local_import_closure(
        {
            "benchmark_execution": "src/benchmark_v1.py",
            "pipeline_cli": "src/pipeline.py",
            "entry_script": "scripts/run_revision_pipeline.py",
        },
        project_root=PROJECT_ROOT,
    )


def planned_budget_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    budget = config["budget_extension"]
    return [
        {
            "stage": "benchmark_budget_v1",
            "benchmark_protocol_id": config["protocol_id"],
            "dataset": dataset,
            "split_point": interface,
            "method": method,
            "model_seed": int(model_seed),
            "attacker_training_seed": int(model_seed),
            "auxiliary_fraction": float(fraction),
            "attacker_architectures": list(budget["attacker_architectures"]),
            "subset_seed": int(budget["subset_seed"]),
            "subset_policy": budget["subset_policy"],
        }
        for dataset in config["datasets"]
        for interface in config["interfaces"]
        for method in budget["methods"]
        for fraction in budget["auxiliary_fractions"]
        for model_seed in config["model_seeds"]
    ]


def planned_restart_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    restart = config["attacker_restarts"]
    return [
        {
            "stage": "attacker_restarts_v1",
            "benchmark_protocol_id": config["protocol_id"],
            "dataset": dataset,
            "split_point": interface,
            "method": method,
            "architecture": architecture,
            "model_seed": int(restart["model_seed"]),
            "attacker_training_seed": int(training_seed),
            "subset_policy": restart["subset_identity_policy"],
            "statistical_role": restart["statistical_role"],
        }
        for dataset in config["datasets"]
        for interface in config["interfaces"]
        for method in restart["methods"]
        for architecture in restart["attacker_architectures"]
        for training_seed in restart["attacker_training_seeds"]
    ]


def _run_id_digest(specs: Iterable[Mapping[str, Any]]) -> str:
    ids = sorted(deterministic_run_id(spec) for spec in specs)
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def _canonical_audits(config: Mapping[str, Any]) -> None:
    base = config["base_study"]
    primary = _strict_json(_project_path(base["primary_audit_path"]))
    primary_counts = primary.get("verification_counts", {})
    if (
        primary.get("passed") is not True
        or primary_counts.get("raw_artifacts_hashed") != 620
        or primary.get("issues")
        or primary.get("quarantined_run_ids")
    ):
        raise RuntimeError("Canonical primary 620-artifact audit is not clean")
    budget = _strict_json(_project_path(base["corrected_budget_audit_path"]))
    budget_counts = budget.get("verification_counts", {})
    if (
        budget.get("passed") is not True
        or budget_counts.get("budget_artifacts_hashed") != 216
        or budget_counts.get("budget_cell_rows") != 24
        or budget_counts.get("budget_attacker_rows") != 48
    ):
        raise RuntimeError("Canonical corrected-budget 24/48/216 audit is not clean")


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _artifact_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        target = _project_path(str(target))
    if not target.is_file():
        raise RuntimeError(f"Missing benchmark evidence artifact: {target}")
    return {
        "path": _relative(target),
        "sha256": _sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _encoder_identity(
    config: Mapping[str, Any],
    revision_root: Path,
    dataset: str,
    split_point: str,
    model_seed: int,
    method: str,
) -> dict[str, Any]:
    source_method = "afd" if method == "afd" else "standard"
    path = revision_root / "checkpoints" / "encoders" / dataset / split_point / f"seed_{model_seed}" / f"{source_method}.pt"
    identity = _artifact_identity(path)
    expected = config["encoder_hashes"].get(identity["path"])
    if identity["sha256"] != expected:
        raise RuntimeError(f"Encoder hash differs from benchmark config: {identity['path']}")
    return identity


def _base_identity(
    config: Mapping[str, Any],
    implementation_closure_sha256: str,
    encoder: Mapping[str, Any],
    model_seed: int,
    attacker_training_seed: int,
) -> dict[str, Any]:
    base = config["base_study"]
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "benchmark_protocol_id": config["protocol_id"],
        "benchmark_config_sha256": config["_config_sha256"],
        "base_contract_sha256": base["contract_sha256"],
        "selection_artifact_sha256": base["selection_artifact_sha256"],
        "selection_lock_sha256": base["selection_lock_sha256"],
        "implementation_closure_sha256": implementation_closure_sha256,
        "encoder": dict(encoder),
        "model_seed": int(model_seed),
        "attacker_training_seed": int(attacker_training_seed),
        "metric_versions": dict(config["metric_versions"]),
        "model_seeds_are_budget_analysis_units": True,
        "images_are_independent_replicates": False,
        "attackers_are_independent_replicates": False,
        "attacker_restarts_are_independent_scientific_replicates": False,
    }


def _full_partition_identity(
    config: Mapping[str, Any], dataset_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = dict(dataset_metadata)
    required = {"name", "partition_source", "synthetic", "train_size", "val_size", "test_size"}
    if set(metadata) != required:
        raise RuntimeError(
            f"Full-partition subset identity has unexpected dataset metadata keys: {sorted(metadata)}"
        )
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "policy": config["attacker_restarts"]["subset_identity_policy"],
        "dataset_metadata": metadata,
        "dataset_metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "fitting_partition": "train",
        "validation_partition": "validation",
        "evaluation_partition": "test",
        "full_partition_sizes": {
            "train": int(metadata["train_size"]),
            "validation": int(metadata["val_size"]),
            "test": int(metadata["test_size"]),
        },
    }


def _historical_budget_record(
    revision_root: Path,
    dataset: str,
    split_point: str,
    method: str,
    fraction: float,
) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in sorted((revision_root / "manifests" / "budget_v2").glob("*.json")):
        record = _strict_json(path)
        spec = record.get("spec", {})
        if (
            spec.get("dataset") == dataset
            and spec.get("split_point") == split_point
            and spec.get("method") == method
            and int(spec.get("seed", -1)) == 42
            and float(spec.get("auxiliary_fraction", -1)) == float(fraction)
        ):
            candidates.append((path, record))
    if len(candidates) != 1 or candidates[0][1].get("status") != "complete":
        raise RuntimeError(f"Expected one complete inherited budget record, found {len(candidates)}")
    return candidates[0]


def _inherited_budget_result(
    config: Mapping[str, Any],
    revision_root: Path,
    spec: Mapping[str, Any],
    closure_sha256: str,
) -> dict[str, Any]:
    path, historical = _historical_budget_record(
        revision_root,
        str(spec["dataset"]),
        str(spec["split_point"]),
        str(spec["method"]),
        float(spec["auxiliary_fraction"]),
    )
    result = historical["result"]
    encoder = _encoder_identity(
        config,
        revision_root,
        str(spec["dataset"]),
        str(spec["split_point"]),
        42,
        str(spec["method"]),
    )
    if result.get("primary_encoder", {}).get("sha256") != encoder["sha256"]:
        raise RuntimeError(f"Inherited budget encoder mismatch in {path}")
    attackers: dict[str, Any] = {}
    artifacts: dict[str, Any] = {"historical_registry": _artifact_identity(path)}
    for architecture in ATTACKER_ARCHITECTURES:
        row = result.get("attackers", {}).get(architecture, {})
        attacker_artifacts = {
            "checkpoint": _artifact_identity(row.get("checkpoint_path", "")),
            "training_log": _artifact_identity(row.get("training_log_path", "")),
            "metric_json": _artifact_identity(row.get("metric_path", "")),
            "sample_bundle": _artifact_identity(row.get("sample_bundle_path", "")),
        }
        for label, identity in attacker_artifacts.items():
            declared = row.get(f"{label.replace('metric_json', 'metric').replace('sample_bundle', 'sample_bundle').replace('training_log', 'training_log').replace('checkpoint', 'checkpoint')}_sha256")
            if declared is not None and declared != identity["sha256"]:
                raise RuntimeError(f"Inherited budget artifact hash mismatch: {identity['path']}")
        attackers[architecture] = {
            "architecture": architecture,
            "test_summary": row["test_summary"],
            "artifact_hashes": attacker_artifacts,
        }
        artifacts[architecture] = attacker_artifacts
    identity = _base_identity(config, closure_sha256, encoder, 42, 42)
    return {
        **identity,
        "origin": "inherited_auxiliary_budget_v2",
        "dataset": spec["dataset"],
        "split_point": spec["split_point"],
        "method": spec["method"],
        "auxiliary_fraction": float(spec["auxiliary_fraction"]),
        "subset_identity": {
            key: result[key]
            for key in (
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
            )
        },
        "attackers": attackers,
        "strongest_attacker": result["strongest_attacker"],
        "worst_case_test_ssim": result["worst_case_test_ssim"],
        "artifact_hashes": artifacts,
    }


def _inherited_restart_result(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    revision_root: Path,
    frozen: Mapping[str, Any],
    selection_provenance: Mapping[str, str],
    spec: Mapping[str, Any],
    closure_sha256: str,
) -> dict[str, Any]:
    dataset = str(spec["dataset"])
    split_point = str(spec["split_point"])
    method = str(spec["method"])
    architecture = str(spec["architecture"])
    primary_path_record = _primary_record(
        revision_root, dict(contract), dataset, split_point, 42, method, dict(selection_provenance)
    )
    primary = primary_path_record["result"]
    encoder = _encoder_identity(config, revision_root, dataset, split_point, 42, method)
    if _sha256_file(Path(primary["encoder_checkpoint"])) != encoder["sha256"]:
        raise RuntimeError("Inherited restart encoder differs from benchmark config")
    training = primary["attackers"][architecture]["training"]
    checkpoint = Path(training["checkpoint_path"])
    log = revision_root / "logs" / "primary" / dataset / split_point / "seed_42" / method / f"attacker_{architecture}.json"
    metric = Path(primary["attackers"][architecture]["metrics_path"])
    sample = Path(primary["attackers"][architecture]["sample_bundle_path"])
    artifacts: dict[str, Any] = {
        "primary_registry": _artifact_identity(
            revision_root / "manifests" / "primary" / f"{primary_path_record['run_id']}.json"
        ),
        "checkpoint": _artifact_identity(checkpoint),
        "training_log": _artifact_identity(log),
        "metric_json": _artifact_identity(metric),
        "sample_bundle": _artifact_identity(sample),
    }
    learned_identity = None
    if method == "learned":
        selected = float(frozen["settings"][dataset][split_point]["methods"]["learned"]["selected_value"])
        learned_path = _learned_path(revision_root, "primary", dataset, split_point, 42, selected)
        learned_identity = _artifact_identity(learned_path)
        if config["restart_learned_perturbation_hashes"].get(learned_identity["path"]) != learned_identity["sha256"]:
            raise RuntimeError("Inherited learned perturbation differs from benchmark config")
        artifacts["learned_perturbation"] = learned_identity
    identity = _base_identity(config, closure_sha256, encoder, 42, 42)
    dataset_metadata = primary.get("dataset_metadata")
    if not isinstance(dataset_metadata, Mapping):
        raise RuntimeError("Inherited restart lacks primary dataset metadata")
    subset_identity = _full_partition_identity(config, dataset_metadata)
    return {
        **identity,
        "origin": "inherited_primary_attacker",
        "dataset": dataset,
        "split_point": split_point,
        "method": method,
        "architecture": architecture,
        "selected_defense_value": primary.get("selected_defense_value"),
        "defense": primary.get("defense"),
        "learned_perturbation": learned_identity,
        "subset_identity": subset_identity,
        "test_summary": primary["attackers"][architecture]["test_summary"],
        "artifact_hashes": artifacts,
    }


def _write_stage_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _strict_json(path) != payload:
            raise RuntimeError(f"Benchmark stage manifest conflicts with {path}")
        return
    _json_write_exclusive(path, dict(payload))


def _register_or_require_complete(
    registry: RunRegistry,
    spec: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> None:
    record = registry.register(spec, resume=True)
    if result is None:
        return
    if record["status"] == "complete":
        if record.get("result") != result:
            raise RuntimeError(f"Inherited benchmark record conflicts with {record['run_id']}")
        return
    if record["status"] != "planned":
        raise RuntimeError(f"Cannot inherit into {record['status']} record {record['run_id']}")
    registry.update(record["run_id"], "complete", result=dict(result))


def initialize_benchmark_v1(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    contract_path: str | Path = DEFAULT_CONTRACT,
    revision_root: str | Path = DEFAULT_REVISION_ROOT,
) -> dict[str, Any]:
    root = Path(revision_root).resolve()
    config = load_benchmark_config(config_path, contract_path=contract_path, revision_root=root)
    contract = load_contract(contract_path)
    _canonical_audits(config)
    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract, revision_root=root, contract_path=contract_path
    )
    closure = benchmark_implementation_closure()
    budget_specs = planned_budget_specs(config)
    restart_specs = planned_restart_specs(config)
    if len(budget_specs) != 120 or len(restart_specs) != 96:
        raise RuntimeError("Benchmark logical matrices are not exactly 120 and 96")
    manifest = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "benchmark_config_path": _relative(Path(config["_config_path"])),
        "benchmark_config_sha256": config["_config_sha256"],
        "implementation_closure": closure,
        "budget": {
            "registry": f"manifests/{BUDGET_REGISTRY}",
            "expected_conditions": 120,
            "expected_attacker_rows": 240,
            "inherited_conditions": 24,
            "new_conditions": 96,
            "run_id_digest": _run_id_digest(budget_specs),
        },
        "attacker_restarts": {
            "registry": f"manifests/{RESTART_REGISTRY}",
            "expected_rows": 96,
            "inherited_rows": 32,
            "new_trainings": 64,
            "run_id_digest": _run_id_digest(restart_specs),
        },
        "statistical_units": {
            "budget": "paired_model_seed",
            "attacker_restarts": "optimizer_sensitivity_condition_not_scientific_replicate",
            "images": "not_independent_replicates",
            "attacker_architectures": "not_independent_replicates",
        },
    }
    _write_stage_manifest(root / "manifests" / STAGE_MANIFEST, manifest)
    budget_registry = RunRegistry(root / "manifests" / BUDGET_REGISTRY)
    for spec in budget_specs:
        inherited = int(spec["model_seed"]) == int(config["budget_extension"]["inherited_model_seed"])
        result = _inherited_budget_result(config, root, spec, closure["closure_sha256"]) if inherited else None
        _register_or_require_complete(budget_registry, spec, result)
    restart_registry = RunRegistry(root / "manifests" / RESTART_REGISTRY)
    for spec in restart_specs:
        inherited = int(spec["attacker_training_seed"]) == int(
            config["attacker_restarts"]["inherited_attacker_training_seed"]
        )
        result = (
            _inherited_restart_result(
                config, contract, root, frozen, selection_provenance, spec, closure["closure_sha256"]
            )
            if inherited
            else None
        )
        _register_or_require_complete(restart_registry, spec, result)
    return benchmark_status(
        config_path=config_path,
        contract_path=contract_path,
        revision_root=root,
    )


def _records_for_exact_specs(directory: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = RunRegistry(directory)
    records = registry.records()
    expected = {deterministic_run_id(spec): spec for spec in specs}
    actual = {record["run_id"]: record for record in records}
    if set(actual) != set(expected):
        raise RuntimeError(
            f"Benchmark registry identity mismatch at {directory}; "
            f"missing={sorted(set(expected) - set(actual))}, unexpected={sorted(set(actual) - set(expected))}"
        )
    for run_id, record in actual.items():
        if record.get("spec") != expected[run_id] or deterministic_run_id(record.get("spec", {})) != run_id:
            raise RuntimeError(f"Benchmark registry configuration changed: {run_id}")
    return [actual[deterministic_run_id(spec)] for spec in specs]


def _provenance_errors(record: Mapping[str, Any]) -> list[str]:
    result = record.get("result")
    if record.get("status") != "complete":
        return []
    if not isinstance(result, Mapping):
        return [f"{record.get('run_id')}: complete record has no result"]
    missing = REQUIRED_PROVENANCE_FIELDS - set(result)
    errors = [f"{record.get('run_id')}: missing provenance {sorted(missing)}"] if missing else []
    if result.get("images_are_independent_replicates") is not False:
        errors.append(f"{record.get('run_id')}: invalid image replicate boundary")
    if result.get("attackers_are_independent_replicates") is not False:
        errors.append(f"{record.get('run_id')}: invalid attacker replicate boundary")
    if result.get("attacker_restarts_are_independent_scientific_replicates") is not False:
        errors.append(f"{record.get('run_id')}: invalid restart replicate boundary")
    return errors


def benchmark_status(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    contract_path: str | Path = DEFAULT_CONTRACT,
    revision_root: str | Path = DEFAULT_REVISION_ROOT,
    preflight_path: str | Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    root = Path(revision_root).resolve()
    config = load_benchmark_config(config_path, contract_path=contract_path, revision_root=root)
    stage_path = root / "manifests" / STAGE_MANIFEST
    stage = _strict_json(stage_path) if stage_path.is_file() else None
    closure = benchmark_implementation_closure()
    closure_matches = bool(stage) and stage.get("implementation_closure", {}).get("closure_sha256") == closure["closure_sha256"]
    matrices: dict[str, Any] = {}
    all_provenance_errors: list[str] = []
    for label, directory, specs, expected in (
        ("budget", root / "manifests" / BUDGET_REGISTRY, planned_budget_specs(config), 120),
        ("attacker_restarts", root / "manifests" / RESTART_REGISTRY, planned_restart_specs(config), 96),
    ):
        if not directory.is_dir():
            matrices[label] = {"expected": expected, "registered": 0, "status_counts": {}}
            continue
        records = _records_for_exact_specs(directory, specs)
        counts: dict[str, int] = {}
        for record in records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            all_provenance_errors.extend(_provenance_errors(record))
        matrices[label] = {
            "expected": expected,
            "registered": len(records),
            "status_counts": counts,
            "complete": counts.get("complete", 0),
            "missing": expected - counts.get("complete", 0),
            "failed_run_ids": [record["run_id"] for record in records if record["status"] == "failed"],
            "running_run_ids": [record["run_id"] for record in records if record["status"] == "running"],
        }
    preflight_target = Path(preflight_path)
    preflight = _strict_json(preflight_target) if preflight_target.is_file() else None
    quarantine_files = list((root / "quarantine" / "benchmark_v1").rglob("*")) if (root / "quarantine" / "benchmark_v1").is_dir() else []
    return {
        "protocol_id": config["protocol_id"],
        "benchmark_config_sha256": config["_config_sha256"],
        "initialized": stage is not None,
        "implementation_closure_matches": closure_matches,
        "preflight": preflight,
        "matrices": matrices,
        "provenance_complete": not all_provenance_errors,
        "provenance_errors": all_provenance_errors,
        "quarantine_file_count": sum(path.is_file() for path in quarantine_files),
    }


def _stage_closure(config: Mapping[str, Any], revision_root: Path) -> str:
    stage_path = revision_root / "manifests" / STAGE_MANIFEST
    if not stage_path.is_file():
        raise RuntimeError("Run init-benchmark-v1 before benchmark execution")
    stage = _strict_json(stage_path)
    if stage.get("benchmark_config_sha256") != config["_config_sha256"]:
        raise RuntimeError("Benchmark stage manifest is bound to another configuration")
    declared = stage.get("implementation_closure", {}).get("closure_sha256")
    actual = benchmark_implementation_closure()["closure_sha256"]
    if declared != actual:
        raise RuntimeError(f"Benchmark implementation hash mismatch: {actual} != {declared}")
    return str(actual)


def require_compute_gate(
    config: Mapping[str, Any],
    closure_sha256: str,
    preflight_path: str | Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    path = Path(preflight_path)
    if not path.is_file():
        raise RuntimeError("Benchmark launch is blocked: compute preflight is missing")
    report = _strict_json(path)
    if report.get("benchmark_config_sha256") != config["_config_sha256"]:
        raise RuntimeError("Benchmark launch is blocked: compute preflight uses another config")
    if report.get("implementation_closure_sha256") != closure_sha256:
        raise RuntimeError("Benchmark launch is blocked: compute preflight uses another implementation")
    if report.get("scratch_run_count") != 4 or report.get("launch_allowed") is not True:
        raise RuntimeError("Benchmark launch is blocked by the frozen compute/storage interlock")
    if float(report.get("projected_gpu_hours", float("inf"))) > 144.0:
        raise RuntimeError("Benchmark launch is blocked: projection exceeds 144 GPU-hours")
    if float(report.get("projected_storage_gb", float("inf"))) > 300.0:
        raise RuntimeError("Benchmark launch is blocked: projection exceeds 300 GB")
    return report


def _budget_artifact_paths(root: Path, spec: Mapping[str, Any], architecture: str) -> dict[str, Path]:
    fraction = float(spec["auxiliary_fraction"])
    base = (
        Path("benchmark_v1")
        / "budget"
        / str(spec["dataset"])
        / str(spec["split_point"])
        / f"model_seed_{spec['model_seed']}"
        / f"{spec['method']}_budget_{fraction:g}"
    )
    suffix = f"attacker_{architecture}_train_seed_{spec['attacker_training_seed']}"
    return {
        "checkpoint": root / "checkpoints" / base / f"{suffix}.pt",
        "training_log": root / "logs" / base / f"{suffix}.json",
        "metric_json": root / "metrics" / base / f"{suffix}.json",
        "sample_bundle": root / "metrics" / base / f"{suffix}_samples.pt",
    }


def _restart_artifact_paths(root: Path, spec: Mapping[str, Any]) -> dict[str, Path]:
    base = (
        Path("benchmark_v1")
        / "restarts"
        / str(spec["dataset"])
        / str(spec["split_point"])
        / f"model_seed_{spec['model_seed']}"
        / str(spec["method"])
    )
    suffix = f"attacker_{spec['architecture']}_train_seed_{spec['attacker_training_seed']}"
    return {
        "checkpoint": root / "checkpoints" / base / f"{suffix}.pt",
        "training_log": root / "logs" / base / f"{suffix}.json",
        "metric_json": root / "metrics" / base / f"{suffix}.json",
        "sample_bundle": root / "metrics" / base / f"{suffix}_samples.pt",
    }


def _ensure_no_collision(record: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(
            f"Benchmark recovery requires quarantining partial/conflicting artifacts for "
            f"{record['run_id']}: {collisions}"
        )


def _load_defended_model(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    revision_root: Path,
    frozen: Mapping[str, Any],
    selection_provenance: Mapping[str, str],
    dataset: str,
    split_point: str,
    model_seed: int,
    method: str,
    num_classes: int,
    device: torch.device,
) -> tuple[Any, FeatureDefense, dict[str, Any], dict[str, Any] | None, float | None]:
    primary = _primary_record(
        revision_root,
        dict(contract),
        dataset,
        split_point,
        model_seed,
        method,
        dict(selection_provenance),
    )["result"]
    model, _ = load_split_model(primary["encoder_checkpoint"], num_classes=num_classes, device=device)
    encoder = _encoder_identity(config, revision_root, dataset, split_point, model_seed, method)
    if _sha256_file(Path(primary["encoder_checkpoint"])) != encoder["sha256"]:
        raise RuntimeError("Primary record encoder differs from benchmark configuration")
    learned_identity = None
    learned_path = None
    selected = primary.get("selected_defense_value")
    if method == "learned":
        learned_path = _learned_path(revision_root, "primary", dataset, split_point, model_seed, float(selected))
        learned_identity = _artifact_identity(learned_path)
        if model_seed == 42 and config["restart_learned_perturbation_hashes"].get(learned_identity["path"]) != learned_identity["sha256"]:
            raise RuntimeError("Learned perturbation differs from benchmark configuration")
    if method == "laplace":
        defense = FeatureDefense("laplace", laplace_scale=float(selected))
    elif method == "learned":
        defense = load_universal_perturbation(learned_path, model.config.feature_shape)
    else:
        defense = FeatureDefense(method)
    return model, defense, encoder, learned_identity, selected


def _train_evaluate_one(
    *,
    contract: Mapping[str, Any],
    model: Any,
    defense: FeatureDefense,
    architecture: str,
    attacker_training_seed: int,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    device: torch.device,
    paths: Mapping[str, Path],
    artifact_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training = train_adaptive_attacker(
        encoder=model.edge_encoder,
        feature_shape=model.config.feature_shape,
        train_loader=train_loader,
        val_loader=val_loader,
        defense=defense,
        architecture=architecture,
        seed=attacker_training_seed,
        device=device,
        checkpoint_path=paths["checkpoint"],
        log_path=paths["training_log"],
        max_epochs=int(contract["attackers"]["max_epochs"]),
        patience=int(contract["attackers"]["patience"]),
        learning_rate=float(contract["attackers"]["learning_rate"]),
        artifact_metadata=dict(artifact_identity),
        exclusive=True,
    )
    from .models import build_attacker

    checkpoint_payload = torch.load(paths["checkpoint"], map_location=device, weights_only=True)
    attacker = build_attacker(architecture, model.config.feature_shape).to(device)
    attacker.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    metrics = evaluate_adaptive_attacker(
        encoder=model.edge_encoder,
        attacker=attacker,
        loader=test_loader,
        defense=defense,
        device=device,
        stochastic_draw_seeds=TEST_DRAW_SEEDS,
        sample_bundle_path=paths["sample_bundle"],
        sample_metadata=dict(artifact_identity),
        exclusive=True,
    )
    _json_write_exclusive(
        paths["metric_json"],
        {"artifact_type": "benchmark_per_image_metrics", **dict(artifact_identity), **metrics},
    )
    hashes = {label: _artifact_identity(path) for label, path in paths.items()}
    return training, metrics, hashes


def run_benchmark_budget_v1(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    contract_path: str | Path = DEFAULT_CONTRACT,
    revision_root: str | Path = DEFAULT_REVISION_ROOT,
    data_dir: str = "data",
    device_name: str = "auto",
    num_workers: int = 0,
    only_run_id: str | None = None,
    preflight_path: str | Path = DEFAULT_PREFLIGHT,
) -> None:
    root = Path(revision_root).resolve()
    config = load_benchmark_config(config_path, contract_path=contract_path, revision_root=root)
    contract = load_contract(contract_path)
    _canonical_audits(config)
    closure_sha256 = _stage_closure(config, root)
    require_compute_gate(config, closure_sha256, preflight_path)
    specs = planned_budget_specs(config)
    records = _records_for_exact_specs(root / "manifests" / BUDGET_REGISTRY, specs)
    by_id = {record["run_id"]: record for record in records}
    expected = {deterministic_run_id(spec): spec for spec in specs}
    if only_run_id is not None and only_run_id not in expected:
        raise ValueError(f"Unknown benchmark budget --only-run-id: {only_run_id}")
    selected = specs if only_run_id is None else [expected[only_run_id]]
    if only_run_id is None:
        unsafe = [(record["run_id"], record["status"]) for record in records if record["status"] in {"running", "failed"}]
        if unsafe:
            raise RuntimeError(f"Benchmark budget bulk execution refuses running/failed records: {unsafe}")
    for spec in selected:
        record = by_id[deterministic_run_id(spec)]
        if record["status"] == "complete":
            continue
        for architecture in ATTACKER_ARCHITECTURES:
            _ensure_no_collision(record, _budget_artifact_paths(root, spec, architecture))
    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract, revision_root=root, contract_path=contract_path
    )
    device = resolve_device(device_name)
    registry = RunRegistry(root / "manifests" / BUDGET_REGISTRY)
    loaders: dict[tuple[str, int], Any] = {}
    for spec in selected:
        record = by_id[deterministic_run_id(spec)]
        if record["status"] == "complete":
            require_selection_provenance(record, selection_provenance) if "selection_provenance" in record else None
            continue
        dataset = str(spec["dataset"])
        split_point = str(spec["split_point"])
        method = str(spec["method"])
        model_seed = int(spec["model_seed"])
        attacker_seed = int(spec["attacker_training_seed"])
        loader_key = (dataset, model_seed)
        if loader_key not in loaders:
            loaders[loader_key] = get_budget_pool_and_test_loaders(
                dataset,
                data_dir,
                int(contract["encoder"]["batch_size"]),
                model_seed,
                num_workers=num_workers,
            )
        (pool_fit, pool_val, test_loader), num_classes, metadata = loaders[loader_key]
        model, defense, encoder, _, selected_value = _load_defended_model(
            config=config,
            contract=contract,
            revision_root=root,
            frozen=frozen,
            selection_provenance=selection_provenance,
            dataset=dataset,
            split_point=split_point,
            model_seed=model_seed,
            method=method,
            num_classes=num_classes,
            device=device,
        )
        budget_train, budget_validation, subset_identity = _budget_loaders(
            pool_fit,
            pool_val,
            float(spec["auxiliary_fraction"]),
            int(config["budget_extension"]["subset_seed"]),
        )
        base_identity = _base_identity(config, closure_sha256, encoder, model_seed, attacker_seed)
        condition_identity = {
            **base_identity,
            "origin": "prospective_benchmark_budget_v1",
            "dataset": dataset,
            "dataset_metadata": asdict(metadata),
            "split_point": split_point,
            "method": method,
            "selected_defense_value": selected_value,
            "defense": defense.to_dict(),
            "auxiliary_fraction": float(spec["auxiliary_fraction"]),
            "subset_identity": subset_identity,
        }
        registry.update(record["run_id"], "running", provenance=condition_identity)
        try:
            attackers: dict[str, Any] = {}
            all_hashes: dict[str, Any] = {}
            for architecture in ATTACKER_ARCHITECTURES:
                paths = _budget_artifact_paths(root, spec, architecture)
                artifact_identity = {
                    **condition_identity,
                    "artifact_type": "benchmark_budget_attacker",
                    "run_id": record["run_id"],
                    "architecture": architecture,
                }
                training, metrics, hashes = _train_evaluate_one(
                    contract=contract,
                    model=model,
                    defense=defense,
                    architecture=architecture,
                    attacker_training_seed=attacker_seed,
                    train_loader=budget_train,
                    val_loader=budget_validation,
                    test_loader=test_loader,
                    device=device,
                    paths=paths,
                    artifact_identity=artifact_identity,
                )
                attackers[architecture] = {
                    "architecture": architecture,
                    "training": training,
                    "test_summary": metrics["summary"],
                    "artifact_hashes": hashes,
                }
                all_hashes[architecture] = hashes
            strongest = max(attackers, key=lambda item: attackers[item]["test_summary"]["ssim"])
            registry.update(
                record["run_id"],
                "complete",
                result={
                    **condition_identity,
                    "attackers": attackers,
                    "strongest_attacker": strongest,
                    "worst_case_test_ssim": attackers[strongest]["test_summary"]["ssim"],
                    "artifact_hashes": all_hashes,
                },
            )
        except Exception as error:
            registry.update(
                record["run_id"],
                "failed",
                error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            )
            raise


def run_attacker_restarts_v1(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    contract_path: str | Path = DEFAULT_CONTRACT,
    revision_root: str | Path = DEFAULT_REVISION_ROOT,
    data_dir: str = "data",
    device_name: str = "auto",
    num_workers: int = 0,
    only_run_id: str | None = None,
    preflight_path: str | Path = DEFAULT_PREFLIGHT,
) -> None:
    root = Path(revision_root).resolve()
    config = load_benchmark_config(config_path, contract_path=contract_path, revision_root=root)
    contract = load_contract(contract_path)
    _canonical_audits(config)
    closure_sha256 = _stage_closure(config, root)
    require_compute_gate(config, closure_sha256, preflight_path)
    specs = planned_restart_specs(config)
    records = _records_for_exact_specs(root / "manifests" / RESTART_REGISTRY, specs)
    by_id = {record["run_id"]: record for record in records}
    expected = {deterministic_run_id(spec): spec for spec in specs}
    if only_run_id is not None and only_run_id not in expected:
        raise ValueError(f"Unknown attacker restart --only-run-id: {only_run_id}")
    selected = specs if only_run_id is None else [expected[only_run_id]]
    if only_run_id is None:
        unsafe = [(record["run_id"], record["status"]) for record in records if record["status"] in {"running", "failed"}]
        if unsafe:
            raise RuntimeError(f"Attacker-restart bulk execution refuses running/failed records: {unsafe}")
    for spec in selected:
        record = by_id[deterministic_run_id(spec)]
        if record["status"] != "complete":
            _ensure_no_collision(record, _restart_artifact_paths(root, spec))
    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract, revision_root=root, contract_path=contract_path
    )
    device = resolve_device(device_name)
    registry = RunRegistry(root / "manifests" / RESTART_REGISTRY)
    loaders: dict[str, Any] = {}
    models: dict[tuple[str, str, str], Any] = {}
    for spec in selected:
        record = by_id[deterministic_run_id(spec)]
        if record["status"] == "complete":
            continue
        dataset = str(spec["dataset"])
        split_point = str(spec["split_point"])
        method = str(spec["method"])
        architecture = str(spec["architecture"])
        model_seed = int(spec["model_seed"])
        attacker_seed = int(spec["attacker_training_seed"])
        if dataset not in loaders:
            loaders[dataset] = get_dataset_loaders(
                dataset,
                data_dir,
                int(contract["encoder"]["batch_size"]),
                model_seed,
                num_workers=num_workers,
            )
        (train_loader, val_loader, test_loader), num_classes, metadata = loaders[dataset]
        model_key = (dataset, split_point, method)
        if model_key not in models:
            models[model_key] = _load_defended_model(
                config=config,
                contract=contract,
                revision_root=root,
                frozen=frozen,
                selection_provenance=selection_provenance,
                dataset=dataset,
                split_point=split_point,
                model_seed=model_seed,
                method=method,
                num_classes=num_classes,
                device=device,
            )
        model, defense, encoder, learned_identity, selected_value = models[model_key]
        identity = {
            **_base_identity(config, closure_sha256, encoder, model_seed, attacker_seed),
            "origin": "prospective_attacker_restarts_v1",
            "dataset": dataset,
            "dataset_metadata": asdict(metadata),
            "split_point": split_point,
            "method": method,
            "architecture": architecture,
            "selected_defense_value": selected_value,
            "defense": defense.to_dict(),
            "learned_perturbation": learned_identity,
            "subset_identity": _full_partition_identity(config, asdict(metadata)),
        }
        paths = _restart_artifact_paths(root, spec)
        registry.update(record["run_id"], "running", provenance=identity)
        try:
            training, metrics, hashes = _train_evaluate_one(
                contract=contract,
                model=model,
                defense=defense,
                architecture=architecture,
                attacker_training_seed=attacker_seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                paths=paths,
                artifact_identity={
                    **identity,
                    "artifact_type": "attacker_restart",
                    "run_id": record["run_id"],
                },
            )
            registry.update(
                record["run_id"],
                "complete",
                result={
                    **identity,
                    "training": training,
                    "test_summary": metrics["summary"],
                    "artifact_hashes": hashes,
                },
            )
        except Exception as error:
            registry.update(
                record["run_id"],
                "failed",
                error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            )
            raise


def time_representative_full_data_runs(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    contract_path: str | Path = DEFAULT_CONTRACT,
    revision_root: str | Path = DEFAULT_REVISION_ROOT,
    data_dir: str = "data",
    device_name: str = "auto",
    num_workers: int = 0,
    output_path: str | Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    output = Path(output_path)
    root = Path(revision_root).resolve()
    config = load_benchmark_config(config_path, contract_path=contract_path, revision_root=root)
    contract = load_contract(contract_path)
    closure_sha256 = _stage_closure(config, root)
    if output.is_file():
        existing = _strict_json(output)
        if (
            existing.get("benchmark_config_sha256") != config["_config_sha256"]
            or existing.get("implementation_closure_sha256") != closure_sha256
            or existing.get("scratch_run_count") != 4
        ):
            raise RuntimeError(f"Existing compute preflight conflicts with {output}")
        return existing
    _canonical_audits(config)
    frozen, selection_provenance, _ = load_sealed_selection(
        contract=contract, revision_root=root, contract_path=contract_path
    )
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Full-data benchmark timing requires a CUDA device")
    scratch_root = output.parent / "timing_scratch_p2"
    rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        (pool_fit, pool_val, test_loader), num_classes, metadata = get_budget_pool_and_test_loaders(
            dataset,
            data_dir,
            int(contract["encoder"]["batch_size"]),
            7,
            num_workers=num_workers,
        )
        model, defense, encoder, _, _ = _load_defended_model(
            config=config,
            contract=contract,
            revision_root=root,
            frozen=frozen,
            selection_provenance=selection_provenance,
            dataset=dataset,
            split_point="early",
            model_seed=7,
            method="standard",
            num_classes=num_classes,
            device=device,
        )
        train_loader, val_loader, subset_identity = _budget_loaders(pool_fit, pool_val, 1.0, 2718)
        for offset, architecture in enumerate(ATTACKER_ARCHITECTURES):
            timing_seed = 900001 + len(rows)
            spec = {
                "dataset": dataset,
                "split_point": "early",
                "method": "standard",
                "model_seed": 7,
                "attacker_training_seed": timing_seed,
                "auxiliary_fraction": 1.0,
            }
            paths = _budget_artifact_paths(scratch_root, spec, architecture)
            fake_record = {"run_id": f"timing-{dataset}-{architecture}"}
            _ensure_no_collision(fake_record, paths)
            identity = {
                **_base_identity(config, closure_sha256, encoder, 7, timing_seed),
                "origin": "full_data_compute_timing_scratch",
                "dataset": dataset,
                "dataset_metadata": asdict(metadata),
                "split_point": "early",
                "method": "standard",
                "architecture": architecture,
                "auxiliary_fraction": 1.0,
                "subset_identity": subset_identity,
                "artifact_type": "compute_timing_scratch",
            }
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            training, metrics, hashes = _train_evaluate_one(
                contract=contract,
                model=model,
                defense=defense,
                architecture=architecture,
                attacker_training_seed=timing_seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                paths=paths,
                artifact_identity=identity,
            )
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            artifact_bytes = sum(int(item["size_bytes"]) for item in hashes.values())
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "full_data": True,
                    "split_point": "early",
                    "method": "standard",
                    "duration_seconds": duration,
                    "artifact_bytes": artifact_bytes,
                    "selected_epoch": training["selected_epoch"],
                    "epochs_completed": training["epochs_completed"],
                    "test_ssim": metrics["summary"]["ssim"],
                    "artifact_hashes": hashes,
                }
            )
    projected_seconds = sum(float(row["duration_seconds"]) * 64 for row in rows)
    projected_bytes = sum(int(row["artifact_bytes"]) * 64 for row in rows)
    projected_gpu_hours = projected_seconds / 3600.0
    projected_storage_gb = projected_bytes / 1_000_000_000.0
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    gate = config["compute_gate"]
    report = {
        "schema_version": 1,
        "benchmark_protocol_id": config["protocol_id"],
        "benchmark_config_sha256": config["_config_sha256"],
        "implementation_closure_sha256": closure_sha256,
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "scratch_run_count": len(rows),
        "scratch_runs": rows,
        "projection_multiplier_per_dataset_architecture": 64,
        "projected_new_trainings": 256,
        "projected_gpu_hours": projected_gpu_hours,
        "projected_storage_gb": projected_storage_gb,
        "free_storage_gb": free_bytes / 1_000_000_000.0,
        "gpu_hour_cap": float(gate["maximum_projected_gpu_hours"]),
        "storage_cap_gb": float(gate["maximum_projected_storage_gb"]),
        "launch_allowed": (
            len(rows) == 4
            and projected_gpu_hours <= float(gate["maximum_projected_gpu_hours"])
            and projected_storage_gb <= float(gate["maximum_projected_storage_gb"])
            and projected_bytes <= free_bytes
        ),
        "protocol_shrunk": False,
    }
    _json_write_exclusive(output, report)
    return report
