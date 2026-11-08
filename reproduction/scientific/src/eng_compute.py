"""Prospective MDPI Eng component-compute profiling contract and utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
COMPUTE_ROOT = PROJECT_ROOT / "revisions" / "2026-08-27_eng_compute_extension" / "eng_compute_v1"
CONTRACT_PATH = COMPUTE_ROOT / "contract.json"
CONDITIONS_PATH = COMPUTE_ROOT / "conditions.json"
ENVIRONMENT_PATH = COMPUTE_ROOT / "environment.json"
SEAL_PATH = COMPUTE_ROOT / "protocol_seal.json"
RAW_ROOT = COMPUTE_ROOT / "raw"

PROTOCOL_ID = "eng_compute_v1"
DATASETS = ("celeba", "cifar10")
INTERFACES = ("early", "current")
METHODS = ("standard", "afd", "laplace", "learned")
MODEL_SEEDS = (7, 42, 123, 2024, 2025)
BATCH_SIZES = (1, 32)
FEATURE_SHAPES = {"early": (6, 16, 16), "current": (6, 8, 8)}
CLASS_COUNTS = {"celeba": 2, "cifar10": 10}

WARMUP_ITERATIONS = 100
TECHNICAL_BLOCKS = 10
MIN_RUN_TIME_SECONDS = 0.5
STABILITY_THRESHOLD = 0.10
MAX_RETRIES = 2

MEASUREMENT_EXCLUSIONS = (
    "dataset_loading",
    "serialization",
    "host_to_device_transfer",
    "networking",
    "queueing",
    "energy",
)
COMPUTE_MACROS = (
    "ComputeConditionCount",
    "ComputeQuarantineCount",
    "ComputeCpuInterfaceDominanceStrataCount",
    "ComputeCpuInterfaceStrataCount",
    "ComputeTransformationDominanceStrataCount",
    "ComputeTransformationStrataCount",
    "ComputeAfdStandardCpuDifferenceMedianUs",
    "ComputeAfdStandardGpuDifferenceMedianUs",
    "ComputeInterfaceCpuDifferenceMedianUs",
    "ComputeInterfaceGpuDifferenceMedianUs",
    "ComputeLaplaceDefenseMedianUs",
    "ComputeLearnedDefenseMedianUs",
)
COMPUTE_CLAIMS = ("E01", "E02", "E03")
RUNTIME_SOURCE_PATHS = (
    Path("src/eng_compute.py"),
    Path("src/models.py"),
    Path("src/defenses.py"),
    Path("src/checkpointing.py"),
    Path("scripts/seal_eng_compute_protocol.py"),
    Path("scripts/profile_eng_compute.py"),
    Path("scripts/audit_eng_compute.py"),
    Path("scripts/analyze_eng_compute.py"),
)


class ComputeContractError(RuntimeError):
    """Raised when prospective compute evidence violates its sealed contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComputeContractError(f"Cannot read JSON {path}: {error}") from error


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write(path, canonical_json_bytes(payload))


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ComputeContractError(f"Path is outside the project root: {path}") from error


def _primary_records() -> dict[tuple[str, str, str, int], tuple[Path, dict[str, Any]]]:
    result: dict[tuple[str, str, str, int], tuple[Path, dict[str, Any]]] = {}
    registry_root = LEGACY_ROOT / "manifests" / "primary"
    for manifest_path in sorted(registry_root.glob("*.json")):
        payload = read_json(manifest_path)
        if payload.get("status") != "complete" or not isinstance(payload.get("result"), Mapping):
            raise ComputeContractError(f"Primary manifest is not completed: {manifest_path}")
        record = dict(payload["result"])
        key = (
            str(record.get("dataset")),
            str(record.get("split_point")),
            str(record.get("method")),
            int(record.get("seed")),
        )
        if key in result:
            raise ComputeContractError(f"Duplicate primary condition: {key}")
        result[key] = (manifest_path, payload)
    expected = {
        (dataset, interface, method, seed)
        for dataset in DATASETS
        for interface in INTERFACES
        for method in METHODS
        for seed in MODEL_SEEDS
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ComputeContractError(f"Primary registry differs from the 80-cell contract: missing={missing}, extra={extra}")
    return result


def _format_defense_value(value: float) -> str:
    return f"{value:g}"


def _learned_checkpoint(dataset: str, interface: str, seed: int, value: float) -> Path:
    return (
        LEGACY_ROOT
        / "checkpoints"
        / "primary"
        / dataset
        / interface
        / f"seed_{seed}"
        / f"learned_l2_{_format_defense_value(value)}.pt"
    )


def stable_condition_id(identity: Mapping[str, Any]) -> str:
    required = ("dataset", "interface", "method", "model_seed", "batch_size")
    missing = [key for key in required if key not in identity]
    if missing:
        raise ComputeContractError(f"Condition identity is missing fields: {missing}")
    digest = sha256_json(dict(identity))[:12]
    return (
        f"{PROTOCOL_ID}__{identity['dataset']}__{identity['interface']}__"
        f"{identity['method']}__s{identity['model_seed']}__b{identity['batch_size']}__{digest}"
    )


def enumerate_conditions() -> list[dict[str, Any]]:
    records = _primary_records()
    study_contract = LEGACY_ROOT / "study_contract.json"
    frozen = LEGACY_ROOT / "selection" / "frozen_operating_points.json"
    frozen_lock = LEGACY_ROOT / "selection" / "frozen_operating_points.lock.json"
    dependencies = {
        "legacy_study_contract": {
            "path": project_relative(study_contract),
            "sha256": sha256_file(study_contract),
        },
        "frozen_operating_points": {
            "path": project_relative(frozen),
            "sha256": sha256_file(frozen),
        },
        "frozen_operating_points_lock": {
            "path": project_relative(frozen_lock),
            "sha256": sha256_file(frozen_lock),
        },
    }
    conditions: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for interface in INTERFACES:
            for method in METHODS:
                for seed in MODEL_SEEDS:
                    manifest_path, manifest = records[(dataset, interface, method, seed)]
                    result = manifest["result"]
                    encoder_path = PROJECT_ROOT / str(result["encoder_checkpoint"])
                    if not encoder_path.is_file():
                        raise ComputeContractError(f"Missing encoder checkpoint: {encoder_path}")
                    defense = dict(result["defense"])
                    learned: dict[str, Any] | None = None
                    if method == "learned":
                        learned_path = _learned_checkpoint(
                            dataset, interface, seed, float(result["selected_defense_value"])
                        )
                        if not learned_path.is_file():
                            raise ComputeContractError(f"Missing learned perturbation: {learned_path}")
                        learned = {
                            "path": project_relative(learned_path),
                            "sha256": sha256_file(learned_path),
                        }
                    for batch_size in BATCH_SIZES:
                        identity = {
                            "protocol_id": PROTOCOL_ID,
                            "dataset": dataset,
                            "interface": interface,
                            "method": method,
                            "model_seed": seed,
                            "batch_size": batch_size,
                        }
                        condition = {
                            **identity,
                            "condition_id": stable_condition_id(identity),
                            "dtype": "float32",
                            "cpu_threads": 1,
                            "gpu_model": "NVIDIA GeForce RTX 2080 Ti",
                            "feature_shape": list(FEATURE_SHAPES[interface]),
                            "num_classes": CLASS_COUNTS[dataset],
                            "primary_run_id": manifest["run_id"],
                            "primary_manifest": {
                                "path": project_relative(manifest_path),
                                "sha256": sha256_file(manifest_path),
                            },
                            "encoder_checkpoint": {
                                "path": project_relative(encoder_path),
                                "sha256": sha256_file(encoder_path),
                                "size_bytes": encoder_path.stat().st_size,
                            },
                            "defense": defense,
                            "learned_perturbation": learned,
                            "legacy_dependencies": dependencies,
                        }
                        conditions.append(condition)
    if len(conditions) != 160 or len({row["condition_id"] for row in conditions}) != 160:
        raise ComputeContractError("Condition enumeration is not exactly 160 unique cells")
    return conditions


def source_bindings(paths: Iterable[Path] = RUNTIME_SOURCE_PATHS) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ComputeContractError(f"Missing bound compute source: {path}")
        rows.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    return rows


def capture_environment() -> dict[str, Any]:
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComputeContractError(f"Cannot query NVIDIA environment: {error}") from error
    if len(gpu_query) != 1:
        raise ComputeContractError(f"Expected exactly one visible GPU, found {len(gpu_query)}")
    gpu_name, gpu_uuid, memory_mib, driver = [part.strip() for part in gpu_query[0].split(",", 3)]

    import torch

    if not torch.cuda.is_available():
        raise ComputeContractError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise ComputeContractError(f"Expected exactly one CUDA device, found {torch.cuda.device_count()}")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2080 Ti":
        raise ComputeContractError(f"Unexpected GPU: {torch.cuda.get_device_name(0)}")

    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            cpu_model = match.group(1).strip()
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_threads_required": 1,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": {
            "name": gpu_name,
            "uuid": gpu_uuid,
            "memory_total_mib": int(memory_mib),
            "driver_version": driver,
        },
        "measurement_exclusions": list(MEASUREMENT_EXCLUSIONS),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "source_bindings": source_bindings(),
    }


def expected_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "title": "A Validation-Frozen Evaluation of Reconstruction--Utility--Compute Trade-Offs in Split Computing",
        "research_question": "What component-level compute latency, throughput, and memory costs do the four frozen candidates exhibit on the bounded CPU--GPU testbed?",
        "prospective": True,
        "dimensions": {
            "datasets": list(DATASETS),
            "interfaces": list(INTERFACES),
            "methods": list(METHODS),
            "model_seeds": list(MODEL_SEEDS),
            "batch_sizes": list(BATCH_SIZES),
            "condition_count": 160,
        },
        "measurement": {
            "dtype": "float32",
            "edge_cpu_threads": 1,
            "cloud_gpu": "NVIDIA GeForce RTX 2080 Ti",
            "resident_features_for_gpu_classifier": True,
            "warmup_iterations": WARMUP_ITERATIONS,
            "technical_blocks": TECHNICAL_BLOCKS,
            "timer": "torch.utils.benchmark.Timer.blocked_autorange",
            "minimum_run_time_seconds": MIN_RUN_TIME_SECONDS,
            "stability_metric": "IQR(block_medians) / median(block_medians)",
            "stability_threshold": STABILITY_THRESHOLD,
            "maximum_retries": MAX_RETRIES,
            "quarantine_blocks_analysis": True,
            "components": [
                "cpu_encoder_plus_defense",
                "defense_only_for_laplace_and_learned",
                "gpu_classifier_with_resident_features",
                "static_component_bytes",
                "isolated_cpu_rss_delta_bytes",
                "cuda_peak_allocation_bytes",
                "summed_component_time_not_end_to_end_latency",
            ],
            "exclusions": list(MEASUREMENT_EXCLUSIONS),
        },
        "analysis": {
            "inferential_unit": "model_seed",
            "technical_blocks_role": "measurement_noise_only",
            "seed_summary": "median_of_accepted_technical_block_medians",
            "uncertainty": "descriptive_two_sided_student_t_95_percent_interval",
            "paired_effect": "AFD_minus_standard_within_dataset_interface_model_seed_batch_size",
            "p_values": False,
            "multiplicity_tests": False,
            "equivalence_claims": False,
            "unfavorable_and_null_results_retained": True,
        },
        "claims": {
            "E01": "Completeness and bounded component-only measurement on the declared CPU--GPU testbed.",
            "E02": "Descriptive interface effects compared with paired AFD-minus-standard component differences without equivalence or causal attribution.",
            "E03": "Explicit Laplace and learned transformation costs compared with paired AFD-minus-standard component differences; null and unfavorable overheads retained.",
        },
        "compute_macros": list(COMPUTE_MACROS),
        "legacy_evidence": {
            "historical_artifacts": 956,
            "legacy_claims": 13,
            "legacy_macros": 129,
            "benchmark_macros": 23,
            "must_remain_independently_verified": True,
        },
        "external_actions_authorized": False,
    }


def load_sealed_bundle() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    for path in (CONTRACT_PATH, CONDITIONS_PATH, ENVIRONMENT_PATH, SEAL_PATH):
        if not path.is_file():
            raise ComputeContractError(f"Missing sealed protocol file: {path}")
    contract = read_json(CONTRACT_PATH)
    condition_payload = read_json(CONDITIONS_PATH)
    environment = read_json(ENVIRONMENT_PATH)
    seal = read_json(SEAL_PATH)
    if contract != expected_contract():
        raise ComputeContractError("Sealed contract differs from the prospective contract implemented in source")
    conditions = condition_payload.get("conditions") if isinstance(condition_payload, Mapping) else None
    if not isinstance(conditions, list) or conditions != enumerate_conditions():
        raise ComputeContractError("Sealed conditions differ from the exact 160-condition enumeration")
    expected_hashes = {
        "contract.json": sha256_file(CONTRACT_PATH),
        "conditions.json": sha256_file(CONDITIONS_PATH),
        "environment.json": sha256_file(ENVIRONMENT_PATH),
    }
    if seal.get("file_hashes") != expected_hashes:
        raise ComputeContractError("Protocol seal file hashes differ")
    if seal.get("source_bindings") != source_bindings():
        raise ComputeContractError("Protocol seal source bindings differ")
    if environment.get("source_bindings") != source_bindings():
        raise ComputeContractError("Environment source bindings differ")
    if environment.get("gpu", {}).get("name") != contract["measurement"]["cloud_gpu"]:
        raise ComputeContractError("Environment GPU differs from the contract")
    return contract, conditions, environment, seal


def block_stability(block_medians_seconds: Sequence[float]) -> dict[str, float | int | bool]:
    values = np.asarray(block_medians_seconds, dtype=np.float64)
    if values.shape != (TECHNICAL_BLOCKS,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ComputeContractError(f"Expected {TECHNICAL_BLOCKS} positive finite block medians")
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75], method="linear")
    ratio = float((q75 - q25) / median)
    return {
        "block_count": int(values.size),
        "median_seconds": median,
        "q25_seconds": float(q25),
        "q75_seconds": float(q75),
        "iqr_seconds": float(q75 - q25),
        "iqr_over_median": ratio,
        "stable": ratio <= STABILITY_THRESHOLD,
    }


def retry_decision(attempt_component_stability: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not attempt_component_stability:
        raise ComputeContractError("No timing attempt was supplied")
    for index, components in enumerate(attempt_component_stability):
        unstable = sorted(name for name, row in components.items() if not bool(row.get("stable")))
        if not unstable:
            return {
                "status": "accepted",
                "selected_attempt": index,
                "attempt_count": index + 1,
                "quarantine_components": [],
            }
        if index >= MAX_RETRIES:
            return {
                "status": "quarantined",
                "selected_attempt": None,
                "attempt_count": index + 1,
                "quarantine_components": unstable,
            }
    return {
        "status": "retry_required",
        "selected_attempt": None,
        "attempt_count": len(attempt_component_stability),
        "quarantine_components": sorted(
            name for name, row in attempt_component_stability[-1].items() if not bool(row.get("stable"))
        ),
    }


def module_static_bytes(module: Any) -> int:
    tensors = list(module.parameters()) + list(module.buffers())
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def student_t_interval(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ComputeContractError("Student-t interval requires at least two finite one-dimensional values")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    from scipy.stats import t

    critical = float(t.ppf(0.975, df=array.size - 1))
    half = critical * std / math.sqrt(array.size)
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def validate_protected_dirty_files(snapshot_manifest: Path | None = None) -> dict[str, Any]:
    manifest_path = snapshot_manifest or (
        PROJECT_ROOT / "audit" / "eng_compute_migration_20260827" / "pre_change" / "snapshot_manifest.json"
    )
    manifest = read_json(manifest_path)
    errors: list[str] = []
    rows = []
    for expected in manifest.get("protected_dirty_files", []):
        path = PROJECT_ROOT / expected["path"]
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == expected["sha256"]
        if not passed:
            errors.append(f"protected dirty file changed: {expected['path']}")
        rows.append({"path": expected["path"], "expected_sha256": expected["sha256"], "actual_sha256": actual, "passed": passed})
    return {"passed": not errors, "files_verified": len(rows), "files": rows, "errors": errors}
