from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.manifest import deterministic_run_id
from src.pipeline import (
    initialize_study,
    pipeline_status,
    planned_primary_specs,
    run_selection,
)
from src.selection_provenance import (
    SelectionProvenanceError,
    load_sealed_selection,
    require_selection_provenance,
    seal_selection,
)
from src.protocol import primary_integrity_gate, registry_gate


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads(
        (
            ROOT / "config" / "study_contract.json"
        ).read_text(encoding="utf-8")
    )


def _candidate(
    dataset: str,
    split_point: str,
    method: str,
    value: float,
    seed: int,
    index: int,
) -> dict:
    attacker_ssim = {"deconv_mse": 0.20 + index / 100.0, "residual_lpips": 0.30 + index / 100.0}
    return {
        "dataset": dataset,
        "split_point": split_point,
        "seed": seed,
        "method": method,
        "candidate_value": value,
        "utility": {"accuracy": 0.90},
        "attackers": {
            architecture: {"validation_metrics": {"summary": {"ssim": ssim}}}
            for architecture, ssim in attacker_ssim.items()
        },
        "worst_case_validation_ssim": attacker_ssim["residual_lpips"],
        "strongest_validation_attacker": "residual_lpips",
        "evaluation_partition": "validation",
        "test_loader_constructed": False,
    }


def _frozen(contract: dict) -> dict:
    method_grids = {
        "afd": contract["afd_alphas"],
        "laplace": contract["laplace_scales"],
        "learned": contract["learned_l2_limits"],
    }
    settings = {}
    for dataset in contract["datasets"]:
        settings[dataset] = {}
        for split_point in contract["split_points"]:
            standard = _candidate(dataset, split_point, "standard", 0.0, contract["selection_seed"], 0)
            methods = {}
            for method, grid in method_grids.items():
                candidates = [
                    _candidate(
                        dataset,
                        split_point,
                        method,
                        float(value),
                        contract["selection_seed"],
                        index,
                    )
                    for index, value in enumerate(grid)
                ]
                methods[method] = {
                    "status": "utility_compatible",
                    "utility_compatible_wording_allowed": True,
                    "selected_value": float(grid[0]),
                    "selected_worst_case_validation_ssim": 0.30,
                    "selected_strongest_attacker": "residual_lpips",
                    "candidates": candidates,
                }
            settings[dataset][split_point] = {
                "standard_validation_accuracy": 0.90,
                "standard_adaptive_evaluation": standard,
                "methods": methods,
            }
    return {
        "schema_version": 2,
        "selection_seed": contract["selection_seed"],
        "partition": "validation_only",
        "test_access": False,
        "utility_loss_limit_percentage_points": contract["utility_loss_limit_percentage_points"],
        "settings": settings,
    }


def _write_unsealed(root: Path) -> tuple[dict, Path, Path]:
    contract = _contract()
    contract_path = root / "study_contract.json"
    frozen_path = root / "selection" / "frozen_operating_points.json"
    frozen_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    frozen_path.write_text(json.dumps(_frozen(contract), indent=2) + "\n", encoding="utf-8")
    return contract, contract_path, frozen_path


def _seal(root: Path) -> tuple[dict, dict[str, str], Path, Path]:
    contract, contract_path, frozen_path = _write_unsealed(root)
    _, provenance, _ = seal_selection(
        contract=contract,
        revision_root=root,
        contract_path=contract_path,
    )
    return contract, provenance, frozen_path, root / "selection" / "frozen_operating_points.lock.json"


def test_seal_is_atomic_one_time_and_idempotent(tmp_path: Path) -> None:
    contract, contract_path, frozen_path = _write_unsealed(tmp_path)
    original_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    _, first_provenance, first_lock = seal_selection(
        contract=contract,
        revision_root=tmp_path,
        contract_path=contract_path,
    )
    lock_path = tmp_path / "selection" / "frozen_operating_points.lock.json"
    lock_bytes = lock_path.read_bytes()
    first_mtime = lock_path.stat().st_mtime_ns
    _, second_provenance, second_lock = seal_selection(
        contract=contract,
        revision_root=tmp_path,
        contract_path=contract_path,
    )
    assert first_lock == second_lock
    assert first_provenance == second_provenance
    assert lock_path.read_bytes() == lock_bytes
    assert lock_path.stat().st_mtime_ns == first_mtime
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == original_hash
    assert (frozen_path.stat().st_mode & 0o777) == 0o444
    assert (lock_path.stat().st_mode & 0o777) == 0o444


def test_production_selection_refuses_existing_artifact_before_device_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_path = tmp_path / "selection" / "frozen_operating_points.json"
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("src.pipeline.resolve_device", lambda _: (_ for _ in ()).throw(AssertionError("device touched")))
    with pytest.raises(RuntimeError, match="immutable and already exists"):
        run_selection(
            contract={},
            revision_root=tmp_path,
            data_dir="unused",
            device_name="auto",
            num_workers=0,
        )


def test_seal_rejects_recomputed_value_mismatch_without_creating_lock(tmp_path: Path) -> None:
    contract, contract_path, frozen_path = _write_unsealed(tmp_path)
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["settings"]["celeba"]["early"]["methods"]["afd"]["selected_value"] = 2.0
    frozen_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SelectionProvenanceError, match="Selected value does not recompute"):
        seal_selection(contract=contract, revision_root=tmp_path, contract_path=contract_path)
    assert not (tmp_path / "selection" / "frozen_operating_points.lock.json").exists()


def test_loader_rejects_frozen_hash_tampering(tmp_path: Path) -> None:
    contract, _, frozen_path, _ = _seal(tmp_path)
    frozen_path.chmod(0o644)
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["settings"]["celeba"]["early"]["standard_validation_accuracy"] = 0.91
    frozen_path.write_text(json.dumps(payload), encoding="utf-8")
    frozen_path.chmod(0o444)
    with pytest.raises(SelectionProvenanceError):
        load_sealed_selection(contract=contract, revision_root=tmp_path)


def test_loader_rejects_contract_tampering(tmp_path: Path) -> None:
    contract, _, _, _ = _seal(tmp_path)
    contract["title"] = "tampered"
    contract_path = tmp_path / "study_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(SelectionProvenanceError, match="lock does not match"):
        load_sealed_selection(contract=contract, revision_root=tmp_path)


def test_loader_rejects_lock_value_tampering(tmp_path: Path) -> None:
    contract, _, _, lock_path = _seal(tmp_path)
    lock_path.chmod(0o644)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["selection_contract"]["selected_defense_settings"][0]["selected_value"] = 99
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_path.chmod(0o444)
    with pytest.raises(SelectionProvenanceError, match="lock does not match"):
        load_sealed_selection(contract=contract, revision_root=tmp_path)


@pytest.mark.parametrize("target", ["frozen", "lock"])
def test_loader_enforces_read_only_modes(tmp_path: Path, target: str) -> None:
    contract, _, frozen_path, lock_path = _seal(tmp_path)
    (frozen_path if target == "frozen" else lock_path).chmod(0o644)
    with pytest.raises(SelectionProvenanceError, match="mode must be 0444"):
        load_sealed_selection(contract=contract, revision_root=tmp_path)


def test_selection_provenance_metadata_does_not_change_run_ids() -> None:
    contract = _contract()
    specs = planned_primary_specs(contract)
    before = [deterministic_run_id(spec) for spec in specs]
    records = [
        {"run_id": run_id, "spec": spec, "selection_provenance": {"lock_sha256": "abc"}}
        for run_id, spec in zip(before, specs, strict=True)
    ]
    after = [deterministic_run_id(record["spec"]) for record in records]
    assert len(before) == 80
    assert before == after


def test_completed_records_require_exact_selection_provenance(tmp_path: Path) -> None:
    expected = {"frozen_sha256": "f", "lock_sha256": "l", "contract_sha256": "c"}
    record = {"run_id": "run", "status": "complete"}
    with pytest.raises(SelectionProvenanceError, match="Missing or mismatched"):
        require_selection_provenance(record, expected)
    record["selection_provenance"] = dict(expected)
    require_selection_provenance(record, expected)

    directory = tmp_path / "primary"
    directory.mkdir()
    result = {
        "attackers": {
            "deconv_mse": {"test_summary": {"ssim": 0.2}},
            "residual_lpips": {"test_summary": {"ssim": 0.3}},
        },
        "strongest_attacker": "residual_lpips",
        "worst_case_test_ssim": 0.3,
        "evaluation_partition": "test_after_frozen_selection",
    }
    path = directory / "run.json"
    path.write_text(json.dumps({"run_id": "run", "status": "complete", "result": result}), encoding="utf-8")
    assert not primary_integrity_gate(directory, 1, expected)["pass"]
    path.write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "complete",
                "selection_provenance": expected,
                "result": result,
            }
        ),
        encoding="utf-8",
    )
    assert primary_integrity_gate(directory, 1, expected)["pass"]
    assert registry_gate(directory, 1, expected)["pass"]


def test_init_preserves_sealed_provenance_and_status_is_read_only(tmp_path: Path) -> None:
    contract, provenance, frozen_path, lock_path = _seal(tmp_path)
    contract_path = tmp_path / "study_contract.json"
    initialize_study(contract_path, tmp_path)
    initialize_study(contract_path, tmp_path)
    manifest = json.loads((tmp_path / "manifests" / "study_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection_provenance"] == provenance
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (frozen_path, lock_path)
    }
    status = pipeline_status(contract, tmp_path)
    after = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (frozen_path, lock_path)
    }
    assert status["selection"]["sealed"] is True
    assert status["primary"]["status_counts"] == {"planned": 80}
    assert before == after
