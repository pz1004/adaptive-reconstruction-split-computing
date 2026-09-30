from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_semantic_results import publish_idempotent
from src.dataset import DatasetMetadata
from src.experiment import MultilabelProbe, TEST_DRAW_SEEDS
from src.manifest import RunRegistry, deterministic_run_id
from src.pipeline import (
    load_contract,
    planned_semantic_specs,
    run_auxiliary_budget_sweep,
    run_semantic_probes,
)
from src.result_audit import ResultAuditError, resolve_artifact_path, sha256_file, validate_registry_records
from src.semantic_result_audit import (
    CANONICAL_TEST_EXAMPLE_IDS,
    EXPECTED_DATASET_METADATA,
    SEMANTIC_TARGET_INDICES,
    SEMANTIC_TARGET_NAMES,
    _require_equal_metric,
    load_semantic_raw_bundle,
    validate_cross_bundle_identity,
    validate_semantic_checkpoint,
    validate_semantic_history,
    validate_semantic_raw_arrays,
)
from src.semantic_analysis import build_semantic_analysis


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT / "config" / "study_contract.json")
PROVENANCE = {
    "frozen_sha256": "f" * 64,
    "lock_sha256": "l" * 64,
    "contract_sha256": "c" * 64,
}


def _raw_arrays(method: str = "standard", *, neutral: bool = False) -> dict[str, np.ndarray]:
    example_count = len(CANONICAL_TEST_EXAMPLE_IDS)
    attribute_count = len(SEMANTIC_TARGET_NAMES)
    targets = (
        np.arange(example_count, dtype=np.uint32)[:, None]
        + np.arange(attribute_count, dtype=np.uint32)[None, :]
    ) % 2
    targets = targets.astype(np.uint8)
    draw_seeds = TEST_DRAW_SEEDS if method == "laplace" else TEST_DRAW_SEEDS[:1]
    base = np.full((example_count, attribute_count), 0.5, dtype=np.float32)
    if not neutral:
        base = np.where(targets == 1, 0.75, 0.25).astype(np.float32)
    probabilities = np.stack([base for _ in draw_seeds], axis=0)
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "probabilities_by_draw": probabilities,
        "binary_targets": targets,
        "example_ids": np.asarray(CANONICAL_TEST_EXAMPLE_IDS, dtype="U10"),
        "target_names": np.asarray(SEMANTIC_TARGET_NAMES, dtype="U32"),
        "target_indices": np.asarray(SEMANTIC_TARGET_INDICES, dtype=np.int64),
        "draw_seeds": np.asarray(draw_seeds, dtype=np.int64),
    }


def _history() -> list[dict[str, float | int]]:
    validation = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return [
        {"epoch": index, "train_loss": 1.0 / index, "validation_loss": value}
        for index, value in enumerate(validation, start=1)
    ]


def _checkpoint_fixture(tmp_path: Path) -> tuple[dict, dict, dict, dict, Path]:
    dependency = tmp_path / "encoder.pt"
    dependency.write_bytes(b"fixed primary dependency")
    contract = {
        "split_points": {"early": [6, 16, 16]},
        "attackers": {"max_epochs": 50, "patience": 5},
    }
    frozen = {"settings": {"celeba": {"early": {"methods": {}}}}}
    result = {
        "dataset": "celeba",
        "dataset_metadata": EXPECTED_DATASET_METADATA,
        "split_point": "early",
        "seed": 7,
        "method": "standard",
        "selected_defense_value": None,
        "defense": {"method": "standard", "stochastic": False},
        "primary_checkpoint_hashes": {
            "encoder": {"path": str(dependency), "sha256": sha256_file(dependency)}
        },
        "raw_bundle_path": str(tmp_path / "standard.predictions.npz"),
    }
    record = {"result": result}
    history = _history()
    payload = {
        "schema_version": 3,
        "artifact_type": "semantic_probe_checkpoint",
        "model_state_dict": MultilabelProbe((6, 16, 16)).state_dict(),
        "dataset": "celeba",
        "dataset_metadata": EXPECTED_DATASET_METADATA,
        "split_point": "early",
        "seed": 7,
        "method": "standard",
        "feature_shape": [6, 16, 16],
        "selected_defense_value": None,
        "defense": result["defense"],
        "selection_provenance": PROVENANCE,
        "primary_checkpoint_hashes": result["primary_checkpoint_hashes"],
        "target_names": list(SEMANTIC_TARGET_NAMES),
        "target_indices": list(SEMANTIC_TARGET_INDICES),
        "test_example_count": 19_962,
        "semantic_draw_seeds": [1701],
        "raw_bundle_path": result["raw_bundle_path"],
        "selected_epoch": 2,
        "best_validation_loss": 0.5,
        "epochs_completed": len(history),
        "max_epochs": 50,
        "patience": 5,
        "history": history,
    }
    result.update(
        {
            "selected_epoch": payload["selected_epoch"],
            "best_validation_loss": payload["best_validation_loss"],
            "history": payload["history"],
        }
    )
    return payload, contract, record, frozen, tmp_path


def test_semantic_registry_detects_duplicate_and_missing_cells() -> None:
    specs = planned_semantic_specs(CONTRACT)
    records = [
        {"run_id": deterministic_run_id(spec), "status": "complete", "spec": spec}
        for spec in specs[:-1]
    ]
    records.append(copy.deepcopy(records[0]))
    errors = validate_registry_records(records, specs)
    assert any("Duplicate run IDs" in error for error in errors)
    assert any("Missing registered run IDs" in error for error in errors)


def test_semantic_checkpoint_rejects_provenance_and_incompatible_state(tmp_path: Path) -> None:
    payload, contract, record, frozen, revision_root = _checkpoint_fixture(tmp_path)
    validate_semantic_checkpoint(
        payload,
        contract=contract,
        record=record,
        provenance=PROVENANCE,
        frozen=frozen,
        revision_root=revision_root,
    )
    altered = copy.deepcopy(payload)
    altered["selection_provenance"]["frozen_sha256"] = "x" * 64
    with pytest.raises(ResultAuditError, match="selection_provenance"):
        validate_semantic_checkpoint(
            altered,
            contract=contract,
            record=record,
            provenance=PROVENANCE,
            frozen=frozen,
            revision_root=revision_root,
        )
    incompatible = copy.deepcopy(payload)
    incompatible["model_state_dict"] = {"unexpected": np.asarray([1])}
    with pytest.raises(ResultAuditError, match="incompatible semantic checkpoint state"):
        validate_semantic_checkpoint(
            incompatible,
            contract=contract,
            record=record,
            provenance=PROVENANCE,
            frozen=frozen,
            revision_root=revision_root,
        )


def test_semantic_history_rejects_wrong_selected_epoch_and_malformed_history() -> None:
    payload = {
        "history": _history(),
        "selected_epoch": 1,
        "best_validation_loss": 1.0,
        "epochs_completed": 7,
        "max_epochs": 50,
        "patience": 5,
    }
    contract = {"attackers": {"max_epochs": 50, "patience": 5}}
    with pytest.raises(ResultAuditError, match="minimum validation-loss"):
        validate_semantic_history(payload, contract)
    payload["selected_epoch"] = 2
    payload["best_validation_loss"] = 0.5
    payload["history"][3]["epoch"] = 9
    with pytest.raises(ResultAuditError, match="not contiguous"):
        validate_semantic_history(payload, contract)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda arrays: arrays["probabilities_by_draw"].__setitem__((0, 0, 0), np.nan), "outside \\[0,1\\]"),
        (lambda arrays: arrays["probabilities_by_draw"].__setitem__((0, 0, 0), 1.1), "outside \\[0,1\\]"),
        (lambda arrays: arrays.__setitem__("probabilities_by_draw", arrays["probabilities_by_draw"][:, :-1]), "shape/dtype"),
        (lambda arrays: arrays["binary_targets"].__setitem__((0, 0), 2), "not binary"),
        (lambda arrays: arrays.__setitem__("target_names", arrays["target_names"][::-1]), "target names or order"),
        (lambda arrays: arrays.__setitem__("draw_seeds", np.asarray([1702], dtype=np.int64)), "draw seeds/count"),
    ],
)
def test_semantic_raw_validator_rejects_malformed_evidence(mutation, match: str) -> None:
    arrays = _raw_arrays()
    mutation(arrays)
    with pytest.raises(ResultAuditError, match=match):
        validate_semantic_raw_arrays(arrays, method="standard")


def test_semantic_raw_validator_enforces_laplace_draw_count_and_accepts_adverse_outcome() -> None:
    laplace = _raw_arrays("laplace", neutral=True)
    summary, rows = validate_semantic_raw_arrays(laplace, method="laplace")
    assert summary["semantic_draw_count"] == 5
    assert len(rows) == 39
    assert summary["macro_auroc"] == pytest.approx(0.5)
    assert summary["macro_balanced_accuracy"] == pytest.approx(0.5)


def test_semantic_metric_drift_and_cross_bundle_identity_are_rejected() -> None:
    _require_equal_metric(0.5, 0.5, "macro_auroc")
    with pytest.raises(ResultAuditError, match="metric drift"):
        _require_equal_metric(0.5, 0.5001, "macro_auroc")
    arrays = _raw_arrays()
    targets = arrays["binary_targets"]
    ids = arrays["example_ids"]
    changed_targets = targets.copy()
    changed_targets[0, 0] = 1 - changed_targets[0, 0]
    with pytest.raises(ResultAuditError, match="target arrays differ"):
        validate_cross_bundle_identity(targets, ids, changed_targets, ids)
    changed_ids = ids.copy()
    changed_ids[0] = "999999.jpg"
    with pytest.raises(ResultAuditError, match="example arrays differ"):
        validate_cross_bundle_identity(targets, ids, targets, changed_ids)


def test_semantic_artifact_path_escape_and_conflicting_outputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ResultAuditError, match="escapes revision root"):
        resolve_artifact_path("../escaped.npz", tmp_path / "revision")
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    (output_dir / "semantic_result_audit.json").write_bytes(b"conflict")
    with pytest.raises(RuntimeError, match="no artifact was overwritten"):
        publish_idempotent(
            output_dir,
            {
                "semantic_result_audit.json": b"expected",
                "semantic_seed_results.csv": b"header\n",
            },
            check=False,
        )
    assert not (output_dir / "semantic_seed_results.csv").exists()


def test_semantic_npz_loader_rejects_malformed_and_duplicate_members(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.npz"
    malformed.write_bytes(b"not a zip archive")
    with pytest.raises(ResultAuditError, match="malformed semantic NPZ"):
        load_semantic_raw_bundle(malformed)
    duplicated = tmp_path / "duplicated.npz"
    with zipfile.ZipFile(duplicated, "w") as archive:
        archive.writestr("schema_version.npy", b"first")
        archive.writestr("schema_version.npy", b"second")
    with pytest.raises(ResultAuditError, match="duplicate members"):
        load_semantic_raw_bundle(duplicated)


def _register_semantic_matrix(revision_root: Path) -> list[dict]:
    registry = RunRegistry(revision_root / "manifests" / "semantic")
    return [registry.register(spec) for spec in planned_semantic_specs(CONTRACT)]


def test_unknown_semantic_only_run_id_fails_before_device_or_data(monkeypatch, tmp_path: Path) -> None:
    revision_root = tmp_path / "revision"
    _register_semantic_matrix(revision_root)

    def forbidden(*args, **kwargs):
        raise AssertionError("device/data/selection initialization must not occur")

    monkeypatch.setattr("src.pipeline.load_sealed_selection", forbidden)
    monkeypatch.setattr("src.pipeline.resolve_device", forbidden)
    monkeypatch.setattr("src.pipeline.get_dataset_loaders", forbidden)
    with pytest.raises(ValueError, match="Unknown semantic --only-run-id"):
        run_semantic_probes(
            contract=CONTRACT,
            revision_root=revision_root,
            data_dir="unused",
            device_name="cuda",
            num_workers=0,
            only_run_id="not-registered",
        )


def test_semantic_only_run_id_recovers_only_the_registered_cell(monkeypatch, tmp_path: Path) -> None:
    revision_root = tmp_path / "revision"
    records = _register_semantic_matrix(revision_root)
    standard = next(record for record in records if record["spec"]["method"] == "standard")
    encoder_path = tmp_path / "encoder.pt"
    encoder_path.write_bytes(b"encoder")
    calls: list[str] = []

    monkeypatch.setattr("src.pipeline.load_sealed_selection", lambda **kwargs: ({}, PROVENANCE, {}))
    monkeypatch.setattr("src.pipeline.resolve_device", lambda name: "cpu")
    monkeypatch.setattr(
        "src.pipeline._primary_record",
        lambda *args, **kwargs: {
            "result": {"encoder_checkpoint": str(encoder_path), "selected_defense_value": None}
        },
    )
    metadata = DatasetMetadata("celeba", 162_770, 19_867, 19_962, "canonical_boundaries", False)
    monkeypatch.setattr(
        "src.pipeline.get_dataset_loaders",
        lambda *args, **kwargs: ((object(), object(), object()), 2, metadata),
    )
    monkeypatch.setattr(
        "src.pipeline.load_split_model",
        lambda *args, **kwargs: (SimpleNamespace(config=SimpleNamespace(feature_shape=(6, 16, 16))), {}),
    )

    def fake_train(**kwargs):
        calls.append(str(kwargs["artifact_metadata"]["method"]))
        return {
            "macro_auroc": 0.5,
            "macro_balanced_accuracy": 0.5,
            "attribute_count": 39,
            "test_example_count": 19_962,
            "checkpoint_path": str(kwargs["checkpoint_path"]),
            "checkpoint_sha256": "a" * 64,
            "raw_bundle_path": str(kwargs["raw_bundle_path"]),
            "raw_bundle_sha256": "b" * 64,
            "target_names": list(SEMANTIC_TARGET_NAMES),
            "target_indices": list(SEMANTIC_TARGET_INDICES),
            "semantic_draw_seeds": [1701],
            "semantic_draw_count": 1,
            "mc_draws_are_independent_replicates": False,
            "attributes_are_independent_replicates": False,
            "images_are_independent_replicates": False,
        }

    monkeypatch.setattr("src.pipeline.train_semantic_probe", fake_train)
    run_semantic_probes(
        contract=CONTRACT,
        revision_root=revision_root,
        data_dir="unused",
        device_name="cpu",
        num_workers=0,
        only_run_id=standard["run_id"],
    )
    final = RunRegistry(revision_root / "manifests" / "semantic").records()
    assert calls == ["standard"]
    assert sum(record["status"] == "complete" for record in final) == 1
    assert sum(record["status"] == "planned" for record in final) == 39
    assert next(record for record in final if record["run_id"] == standard["run_id"])["spec"] == standard["spec"]


def test_budget_interlock_fails_before_device_or_data(monkeypatch, tmp_path: Path) -> None:
    def blocked(*args, **kwargs):
        raise RuntimeError("semantic audit blocked")

    def forbidden(*args, **kwargs):
        raise AssertionError("budget device/data initialization must not occur")

    monkeypatch.setattr("src.semantic_result_audit.require_canonical_semantic_audit", blocked)
    monkeypatch.setattr("src.pipeline.resolve_device", forbidden)
    monkeypatch.setattr("src.pipeline.get_dataset_loaders", forbidden)
    with pytest.raises(RuntimeError, match="semantic audit blocked"):
        run_auxiliary_budget_sweep(
            contract=CONTRACT,
            revision_root=tmp_path / "revision",
            data_dir="unused",
            device_name="cuda",
            num_workers=0,
        )


def test_semantic_analysis_publishes_eight_groups_and_separate_twelve_contrasts() -> None:
    rows = []
    for split_index, split_point in enumerate(("early", "current")):
        for method_index, method in enumerate(("standard", "afd", "laplace", "learned")):
            for seed_index, seed in enumerate(CONTRACT["seeds"]):
                rows.append(
                    {
                        "split_point": split_point,
                        "method": method,
                        "seed": seed,
                        "macro_auroc": 0.5 + 0.01 * split_index + 0.02 * method_index + 0.001 * seed_index,
                        "macro_balanced_accuracy": 0.5 + 0.02 * split_index - 0.01 * method_index + 0.001 * seed_index,
                    }
                )
    analysis = build_semantic_analysis(rows, list(CONTRACT["seeds"]))
    assert analysis["group_summary_count"] == 8
    assert analysis["contrast_count"] == 12
    assert len(analysis["group_summaries"]) == 8
    assert len(analysis["contrasts"]) == 12
    assert all(contrast["holm_family"].startswith("12 declared semantic") for contrast in analysis["contrasts"].values())
    assert analysis["images_are_replicates"] is False
    assert analysis["attributes_are_replicates"] is False
    assert analysis["draws_are_replicates"] is False
