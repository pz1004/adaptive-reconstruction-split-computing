"""Immutable validation-selection sealing and downstream provenance checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


FROZEN_RELATIVE_PATH = Path("selection/frozen_operating_points.json")
LOCK_RELATIVE_PATH = Path("selection/frozen_operating_points.lock.json")
SEALED_MODE = 0o444


class SelectionProvenanceError(RuntimeError):
    """Raised when validation-selection evidence is absent or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite JSON number: {raw}")
    return value


def _reject_constant(raw: str) -> None:
    raise ValueError(f"Non-standard JSON number: {raw}")


def strict_json_load(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_float=_parse_float, parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SelectionProvenanceError(f"Invalid JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SelectionProvenanceError(f"Expected a JSON object at {path}")
    return payload


def _require_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(value)
    if missing:
        raise SelectionProvenanceError(f"{context} is missing {sorted(missing)}")


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SelectionProvenanceError(f"{context} must be a finite number")
    return float(value)


def _validate_candidate(
    candidate: Mapping[str, Any],
    *,
    dataset: str,
    split_point: str,
    method: str,
    value: float,
    seed: int,
    attacker_architectures: list[str],
) -> tuple[float, float, str]:
    _require_keys(
        candidate,
        {
            "dataset",
            "split_point",
            "seed",
            "method",
            "candidate_value",
            "utility",
            "attackers",
            "worst_case_validation_ssim",
            "strongest_validation_attacker",
            "evaluation_partition",
            "test_loader_constructed",
        },
        f"candidate {dataset}/{split_point}/{method}/{value:g}",
    )
    identity = (candidate["dataset"], candidate["split_point"], candidate["method"], int(candidate["seed"]))
    if identity != (dataset, split_point, method, seed):
        raise SelectionProvenanceError(f"Candidate identity mismatch for {dataset}/{split_point}/{method}/{value:g}")
    if _number(candidate["candidate_value"], "candidate_value") != value:
        raise SelectionProvenanceError(f"Candidate grid value mismatch for {dataset}/{split_point}/{method}")
    if candidate["evaluation_partition"] != "validation" or candidate["test_loader_constructed"] is not False:
        raise SelectionProvenanceError(f"Candidate has test-selection provenance for {dataset}/{split_point}/{method}")
    utility = candidate["utility"]
    if not isinstance(utility, Mapping) or "accuracy" not in utility:
        raise SelectionProvenanceError(f"Candidate utility is incomplete for {dataset}/{split_point}/{method}")
    accuracy = _number(utility["accuracy"], "candidate utility accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise SelectionProvenanceError(f"Candidate utility accuracy is outside [0, 1] for {dataset}/{split_point}/{method}")
    attackers = candidate["attackers"]
    if not isinstance(attackers, Mapping) or set(attackers) != set(attacker_architectures):
        raise SelectionProvenanceError(f"Candidate attacker grid mismatch for {dataset}/{split_point}/{method}")
    attacker_ssim: dict[str, float] = {}
    for architecture in attacker_architectures:
        record = attackers[architecture]
        try:
            summary = record["validation_metrics"]["summary"]
            attacker_ssim[architecture] = _number(summary["ssim"], f"{architecture} validation SSIM")
        except (KeyError, TypeError) as error:
            raise SelectionProvenanceError(
                f"Candidate attacker result is incomplete for {dataset}/{split_point}/{method}/{architecture}"
            ) from error
    strongest = max(attacker_architectures, key=lambda name: attacker_ssim[name])
    worst_ssim = attacker_ssim[strongest]
    if candidate["strongest_validation_attacker"] != strongest:
        raise SelectionProvenanceError(f"Strongest validation attacker mismatch for {dataset}/{split_point}/{method}")
    if _number(candidate["worst_case_validation_ssim"], "worst-case validation SSIM") != worst_ssim:
        raise SelectionProvenanceError(f"Worst-case validation SSIM mismatch for {dataset}/{split_point}/{method}")
    return accuracy, worst_ssim, strongest


def validate_and_summarize_selection(
    frozen: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every candidate and independently recompute all selected settings."""
    _require_keys(
        frozen,
        {
            "schema_version",
            "selection_seed",
            "partition",
            "test_access",
            "utility_loss_limit_percentage_points",
            "settings",
        },
        "frozen selection",
    )
    required_contract = {
        "datasets",
        "split_points",
        "selection_seed",
        "afd_alphas",
        "laplace_scales",
        "learned_l2_limits",
        "utility_loss_limit_percentage_points",
        "attackers",
        "selection_partition",
        "test_access_before_freeze",
    }
    _require_keys(contract, required_contract, "study contract")
    seed = int(contract["selection_seed"])
    limit = _number(contract["utility_loss_limit_percentage_points"], "contract utility-loss limit")
    if frozen["schema_version"] != 2:
        raise SelectionProvenanceError("Frozen selection schema_version must be 2")
    if int(frozen["selection_seed"]) != seed:
        raise SelectionProvenanceError("Frozen selection seed does not match the study contract")
    if frozen["partition"] != contract["selection_partition"] or frozen["partition"] != "validation_only":
        raise SelectionProvenanceError("Frozen selection is not validation-only")
    if frozen["test_access"] is not False or contract["test_access_before_freeze"] is not False:
        raise SelectionProvenanceError("Frozen selection permits test access before freezing")
    if _number(frozen["utility_loss_limit_percentage_points"], "frozen utility-loss limit") != limit:
        raise SelectionProvenanceError("Frozen utility-loss limit does not match the study contract")
    settings = frozen["settings"]
    datasets = list(contract["datasets"])
    split_points = list(contract["split_points"])
    if not isinstance(settings, Mapping) or set(settings) != set(datasets):
        raise SelectionProvenanceError("Frozen selection dataset set does not match the study contract")
    attacker_architectures = list(contract["attackers"].get("architectures", []))
    if not attacker_architectures:
        raise SelectionProvenanceError("Study contract has no attacker architectures")
    method_grids = {
        "afd": list(map(float, contract["afd_alphas"])),
        "laplace": list(map(float, contract["laplace_scales"])),
        "learned": list(map(float, contract["learned_l2_limits"])),
    }
    summaries: list[dict[str, Any]] = []
    selected_attackers: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in datasets:
        dataset_settings = settings[dataset]
        if not isinstance(dataset_settings, Mapping) or set(dataset_settings) != set(split_points):
            raise SelectionProvenanceError(f"Frozen split-point set does not match for {dataset}")
        selected_attackers[dataset] = {}
        for split_point in split_points:
            setting = dataset_settings[split_point]
            _require_keys(
                setting,
                {"standard_validation_accuracy", "standard_adaptive_evaluation", "methods"},
                f"selection setting {dataset}/{split_point}",
            )
            standard_accuracy = _number(
                setting["standard_validation_accuracy"], f"standard accuracy {dataset}/{split_point}"
            )
            standard = setting["standard_adaptive_evaluation"]
            standard_candidate_accuracy, _, standard_attacker = _validate_candidate(
                standard,
                dataset=dataset,
                split_point=split_point,
                method="standard",
                value=0.0,
                seed=seed,
                attacker_architectures=attacker_architectures,
            )
            if standard_accuracy != standard_candidate_accuracy:
                raise SelectionProvenanceError(f"Standard validation accuracy mismatch for {dataset}/{split_point}")
            methods = setting["methods"]
            if not isinstance(methods, Mapping) or set(methods) != set(method_grids):
                raise SelectionProvenanceError(f"Frozen method set does not match for {dataset}/{split_point}")
            selected_attackers[dataset][split_point] = {"standard": standard_attacker}
            utility_floor = standard_accuracy - limit / 100.0
            for method, expected_grid in method_grids.items():
                method_record = methods[method]
                _require_keys(
                    method_record,
                    {
                        "status",
                        "utility_compatible_wording_allowed",
                        "selected_value",
                        "selected_worst_case_validation_ssim",
                        "selected_strongest_attacker",
                        "candidates",
                    },
                    f"selection method {dataset}/{split_point}/{method}",
                )
                candidates = method_record["candidates"]
                if not isinstance(candidates, list) or len(candidates) != len(expected_grid):
                    raise SelectionProvenanceError(f"Candidate count mismatch for {dataset}/{split_point}/{method}")
                observed_grid = [_number(candidate.get("candidate_value"), "candidate grid value") for candidate in candidates]
                if observed_grid != expected_grid or len(set(observed_grid)) != len(observed_grid):
                    raise SelectionProvenanceError(f"Candidate grid mismatch for {dataset}/{split_point}/{method}")
                validated = []
                for value, candidate in zip(expected_grid, candidates, strict=True):
                    accuracy, worst_ssim, strongest = _validate_candidate(
                        candidate,
                        dataset=dataset,
                        split_point=split_point,
                        method=method,
                        value=value,
                        seed=seed,
                        attacker_architectures=attacker_architectures,
                    )
                    validated.append((candidate, accuracy, worst_ssim, strongest))
                compatible = [row for row in validated if row[1] >= utility_floor]
                if compatible:
                    selected = min(compatible, key=lambda row: row[2])
                    expected_status = "utility_compatible"
                    wording_allowed = True
                else:
                    selected = max(validated, key=lambda row: row[1])
                    expected_status = "exploratory_no_utility_compatible_candidate"
                    wording_allowed = False
                candidate, _, selected_ssim, selected_attacker = selected
                selected_value = _number(method_record["selected_value"], "selected defense value")
                if selected_value != _number(candidate["candidate_value"], "recomputed selected value"):
                    raise SelectionProvenanceError(f"Selected value does not recompute for {dataset}/{split_point}/{method}")
                if method_record["status"] != expected_status:
                    raise SelectionProvenanceError(f"Utility-compatibility status mismatch for {dataset}/{split_point}/{method}")
                if method_record["utility_compatible_wording_allowed"] is not wording_allowed:
                    raise SelectionProvenanceError(f"Utility-wording status mismatch for {dataset}/{split_point}/{method}")
                if _number(method_record["selected_worst_case_validation_ssim"], "selected validation SSIM") != selected_ssim:
                    raise SelectionProvenanceError(f"Selected validation SSIM mismatch for {dataset}/{split_point}/{method}")
                if method_record["selected_strongest_attacker"] != selected_attacker:
                    raise SelectionProvenanceError(f"Selected attacker mismatch for {dataset}/{split_point}/{method}")
                selected_attackers[dataset][split_point][method] = selected_attacker
                summaries.append(
                    {
                        "dataset": dataset,
                        "split_point": split_point,
                        "method": method,
                        "candidate_grid": expected_grid,
                        "selected_value": selected_value,
                        "utility_compatibility_status": expected_status,
                        "utility_compatible_wording_allowed": wording_allowed,
                    }
                )
    if len(summaries) != 12:
        raise SelectionProvenanceError(f"Expected 12 selected defense settings, found {len(summaries)}")
    return {
        "selection_seed": seed,
        "partition": "validation_only",
        "test_access": False,
        "utility_loss_limit_percentage_points": limit,
        "selected_defense_settings": summaries,
        "validation_selected_attackers": selected_attackers,
    }


def _lock_payload(
    *,
    frozen_path: Path,
    contract_path: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seal_type": "validation_only_selection",
        "frozen_artifact": {
            "path": FROZEN_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(frozen_path),
            "size_bytes": frozen_path.stat().st_size,
            "required_mode": "0444",
        },
        "study_contract": {
            "path": "study_contract.json",
            "sha256": sha256_file(contract_path),
            "size_bytes": contract_path.stat().st_size,
        },
        "selection_contract": dict(summary),
    }


def _selection_provenance(lock: Mapping[str, Any], lock_path: Path) -> dict[str, str]:
    return {
        "frozen_sha256": str(lock["frozen_artifact"]["sha256"]),
        "lock_sha256": sha256_file(lock_path),
        "contract_sha256": str(lock["study_contract"]["sha256"]),
    }


def load_sealed_selection(
    *,
    contract: Mapping[str, Any],
    revision_root: str | Path,
    contract_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Load a sealed selection only when files, values, hashes, and modes agree."""
    root = Path(revision_root)
    frozen_path = root / FROZEN_RELATIVE_PATH
    lock_path = root / LOCK_RELATIVE_PATH
    resolved_contract_path = Path(contract_path) if contract_path is not None else root / "study_contract.json"
    for path, label in ((frozen_path, "frozen selection"), (lock_path, "selection lock"), (resolved_contract_path, "study contract")):
        if not path.is_file():
            raise SelectionProvenanceError(f"Missing {label}: {path}")
    if (frozen_path.stat().st_mode & 0o777) != SEALED_MODE:
        raise SelectionProvenanceError(f"Frozen selection mode must be 0444: {frozen_path}")
    if (lock_path.stat().st_mode & 0o777) != SEALED_MODE:
        raise SelectionProvenanceError(f"Selection lock mode must be 0444: {lock_path}")
    contract_on_disk = strict_json_load(resolved_contract_path)
    if contract_on_disk != dict(contract):
        raise SelectionProvenanceError("In-memory study contract differs from the sealed contract file")
    frozen = strict_json_load(frozen_path)
    lock = strict_json_load(lock_path)
    summary = validate_and_summarize_selection(frozen, contract_on_disk)
    expected_lock = _lock_payload(
        frozen_path=frozen_path,
        contract_path=resolved_contract_path,
        summary=summary,
    )
    if lock != expected_lock:
        raise SelectionProvenanceError("Selection lock does not match the frozen artifact, contract, and recomputed values")
    return frozen, _selection_provenance(lock, lock_path), lock


def _write_lock_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(SEALED_MODE)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seal_selection(
    *,
    contract: Mapping[str, Any],
    revision_root: str | Path,
    contract_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Create the one-time lock without rewriting the frozen selection artifact."""
    root = Path(revision_root)
    frozen_path = root / FROZEN_RELATIVE_PATH
    lock_path = root / LOCK_RELATIVE_PATH
    resolved_contract_path = Path(contract_path) if contract_path is not None else root / "study_contract.json"
    if lock_path.exists():
        return load_sealed_selection(
            contract=contract,
            revision_root=root,
            contract_path=resolved_contract_path,
        )
    if not frozen_path.is_file():
        raise SelectionProvenanceError(f"Missing frozen selection: {frozen_path}")
    contract_on_disk = strict_json_load(resolved_contract_path)
    if contract_on_disk != dict(contract):
        raise SelectionProvenanceError("In-memory study contract differs from the contract file")
    frozen = strict_json_load(frozen_path)
    summary = validate_and_summarize_selection(frozen, contract_on_disk)
    payload = _lock_payload(
        frozen_path=frozen_path,
        contract_path=resolved_contract_path,
        summary=summary,
    )
    frozen_path.chmod(SEALED_MODE)
    try:
        _write_lock_once(lock_path, payload)
    except FileExistsError:
        pass
    return load_sealed_selection(
        contract=contract,
        revision_root=root,
        contract_path=resolved_contract_path,
    )


def require_selection_provenance(
    record: Mapping[str, Any], expected: Mapping[str, str], *, context: str | None = None
) -> None:
    if record.get("selection_provenance") != dict(expected):
        label = context or str(record.get("run_id", "record"))
        raise SelectionProvenanceError(f"Missing or mismatched selection provenance for {label}")
