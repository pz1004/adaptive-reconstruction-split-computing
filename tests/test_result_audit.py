from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.analysis import paired_analysis
from src.manifest import deterministic_run_id
from src.result_audit import (
    ResultAuditError,
    describe_test_utility,
    resolve_artifact_path,
    validate_attacker_checkpoint,
    validate_attacker_specific_reconstructions,
    validate_encoder_checkpoint,
    validate_learned_checkpoint,
    validate_matched_original,
    validate_manifest_learned_l2,
    validate_metric_payload,
    validate_registry_records,
    validate_sample_bundle,
    validate_selection_provenance,
    validate_training_history,
)


def _spec(seed: int = 7) -> dict:
    return {
        "stage": "primary_defense_evaluation",
        "dataset": "celeba",
        "split_point": "early",
        "seed": seed,
        "method": "standard",
    }


def _record(spec: dict | None = None) -> dict:
    selected = _spec() if spec is None else spec
    return {"run_id": deterministic_run_id(selected), "status": "complete", "spec": selected}


def _contract() -> dict:
    return {
        "attackers": {
            "max_epochs": 10,
            "patience": 2,
            "losses": {"deconv_mse": "mse", "residual_lpips": "l1_plus_0.1_lpips_alex"},
        }
    }


def _history() -> dict:
    return {
        "architecture": "deconv_mse",
        "seed": 7,
        "feature_shape": [6, 16, 16],
        "history": [
            {"epoch": 1, "train_loss": 0.4, "validation_loss": 0.3},
            {"epoch": 2, "train_loss": 0.3, "validation_loss": 0.2},
            {"epoch": 3, "train_loss": 0.2, "validation_loss": 0.25},
            {"epoch": 4, "train_loss": 0.2, "validation_loss": 0.24},
        ],
        "selected_epoch": 2,
        "epochs_completed": 4,
        "best_validation_loss": 0.2,
        "patience": 2,
    }


def _metric_payload() -> dict:
    values = {
        "mse": [0.1, 0.3],
        "psnr": [10.0, 20.0],
        "ssim": [0.2, 0.6],
        "lpips": [0.4, 0.8],
    }
    return {
        "example_count": 2,
        "mc_draw_seeds": [1701],
        "mc_draw_count": 1,
        "draws_are_independent_replicates": False,
        "metric_versions": {"version": "test"},
        "per_image": values,
        "summary": {name: float(np.mean(items)) for name, items in values.items()},
    }


def test_registry_rejects_missing_and_duplicate_records() -> None:
    expected = [_spec(7), _spec(42)]
    duplicate = _record(_spec(7))
    errors = validate_registry_records([duplicate, copy.deepcopy(duplicate)], expected)
    assert any("Duplicate run IDs" in error for error in errors)
    assert any("Missing registered run IDs" in error for error in errors)


def test_altered_provenance_is_rejected() -> None:
    expected = {"frozen_sha256": "f", "lock_sha256": "l", "contract_sha256": "c"}
    with pytest.raises(ResultAuditError, match="provenance"):
        validate_selection_provenance({"selection_provenance": {**expected, "lock_sha256": "changed"}}, expected)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "revision"
    root.mkdir()
    with pytest.raises(ResultAuditError, match="escapes"):
        resolve_artifact_path("../outside.pt", root.resolve())


def test_missing_encoder_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ResultAuditError, match="missing encoder"):
        validate_encoder_checkpoint(
            tmp_path / "missing.pt",
            dataset="celeba",
            split_point="early",
            seed=7,
            source_method="standard",
            feature_shape=(6, 16, 16),
            selected_epoch=1,
            hyperparameters={},
        )


def test_incompatible_attacker_checkpoint_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "attacker.pt"
    training = _history()
    torch.save(
        {
            "schema_version": 2,
            "architecture": "residual_lpips",
            "feature_shape": [6, 16, 16],
            "seed": 7,
            "max_epochs": 10,
            "patience": 2,
            "selected_epoch": 2,
            "best_validation_loss": 0.2,
            "loss": "mse",
            "model_state_dict": {},
        },
        path,
    )
    with pytest.raises(ResultAuditError, match="architecture metadata"):
        validate_attacker_checkpoint(
            path,
            architecture="deconv_mse",
            feature_shape=(6, 16, 16),
            seed=7,
            training=training,
            contract=_contract(),
        )


def test_incorrect_selected_epoch_is_rejected() -> None:
    training = _history()
    training["selected_epoch"] = 1
    with pytest.raises(ResultAuditError, match="minimum validation-loss epoch"):
        validate_training_history(training, _contract())


def test_learned_l2_allows_float32_drift_but_rejects_a_limit_violation(tmp_path: Path) -> None:
    path = tmp_path / "learned_l2_1.pt"
    base = {
        "schema_version": 2,
        "noise": torch.ones(1, 1, 1, 1),
        "max_l2": 1.0,
        "realized_l2": 1.0 - 5e-7,
        "metadata": {"stage": "primary", "dataset": "celeba", "split_point": "early", "seed": 7},
    }
    torch.save(base, path)
    validate_learned_checkpoint(
        path,
        dataset="celeba",
        split_point="early",
        seed=7,
        feature_shape=(1, 1, 1),
        selected_value=1.0,
    )
    base["noise"] = torch.full((1, 1, 1, 1), 1.01)
    base["realized_l2"] = 1.01
    torch.save(base, path)
    with pytest.raises(ResultAuditError, match="exceeds"):
        validate_learned_checkpoint(
            path,
            dataset="celeba",
            split_point="early",
            seed=7,
            feature_shape=(1, 1, 1),
            selected_value=1.0,
        )


def test_manifest_learned_l2_uses_the_same_float32_tolerance() -> None:
    validate_manifest_learned_l2(1.0, 1.0 - 5e-7)
    with pytest.raises(ResultAuditError, match="beyond"):
        validate_manifest_learned_l2(1.0, 1.0 - 2e-6)


@pytest.mark.parametrize("fault", ["nonfinite", "wrong_length", "summary_drift", "mc_seeds"])
def test_metric_integrity_faults_are_rejected(fault: str) -> None:
    payload = _metric_payload()
    if fault == "nonfinite":
        payload["per_image"]["ssim"][0] = float("nan")
    elif fault == "wrong_length":
        payload["per_image"]["ssim"].append(0.5)
    elif fault == "summary_drift":
        payload["summary"]["ssim"] += 0.01
    else:
        payload["mc_draw_seeds"] = [1702]
    with pytest.raises(ResultAuditError):
        validate_metric_payload(
            payload,
            example_count=2,
            draw_seeds=[1701],
            manifest_summary=_metric_payload()["summary"],
            expected_versions={"version": "test"},
        )


def test_sample_bundle_rejects_nonfinite_and_unmatched_originals() -> None:
    original = torch.zeros(5, 3, 32, 32)
    reconstruction = torch.ones(5, 3, 32, 32)
    validate_sample_bundle(
        {"original": original, "reconstruction": reconstruction, "draw_seed": 1701},
        draw_seed=1701,
    )
    with pytest.raises(ResultAuditError, match="do not match"):
        validate_matched_original(original, torch.ones_like(original))
    damaged = reconstruction.clone()
    damaged[0, 0, 0, 0] = float("nan")
    with pytest.raises(ResultAuditError, match="non-finite"):
        validate_sample_bundle(
            {"original": original, "reconstruction": damaged, "draw_seed": 1701},
            draw_seed=1701,
        )


def test_attacker_reconstructions_must_remain_distinct() -> None:
    reconstruction = torch.zeros(5, 3, 32, 32)
    with pytest.raises(ResultAuditError, match="identical"):
        validate_attacker_specific_reconstructions(
            {"deconv_mse": reconstruction, "residual_lpips": reconstruction.clone()}
        )


def test_utility_threshold_is_descriptive_and_outcome_neutral() -> None:
    rows = [
        {"dataset": "celeba", "split_point": "early", "seed": 7, "method": "standard", "accuracy_percentage_points": 90.0},
        {"dataset": "celeba", "split_point": "early", "seed": 7, "method": "afd", "accuracy_percentage_points": 80.0},
    ]
    report = describe_test_utility(rows, limit=1.0)
    assert report["no_reselection_from_test"] is True
    assert report["exceedance_count"] == 1
    assert report["per_seed_rows"][0]["role"] == "descriptive_test_behavior_only_no_reselection"


def test_paired_analysis_records_low_power_shapiro_diagnostic() -> None:
    result = paired_analysis([1, 2, 3, 4, 5], [2, 2, 4, 4, 7])
    diagnostic = result["paired_difference_shapiro_wilk"]
    assert diagnostic["n"] == 5
    assert diagnostic["role"] == "low_power_diagnostic_only"
    assert diagnostic["changes_declared_inference"] is False
