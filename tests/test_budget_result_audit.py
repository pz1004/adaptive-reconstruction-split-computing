from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.analyze_budget_results import build_budget_analysis, publish_outputs
from scripts.audit_budget_results import publish_idempotent
from scripts.analyze_budget_results import require_matching_consolidated
from src.budget_result_audit import (
    ResultAuditError,
    expected_auxiliary_identity,
    validate_budget_common_identity,
    validate_budget_sample_bundle,
)
from src.dataset import DatasetMetadata, get_budget_pool_and_test_loaders
from src.manifest import RunRegistry, deterministic_run_id
from src.pipeline import (
    EXPECTED_BUDGET_RUN_ID_DIGEST,
    BUDGET_ARTIFACT_SCHEMA_VERSION,
    BUDGET_PROTOCOL_VERSION,
    BUDGET_REGISTRY_DIRECTORY,
    _budget_run_id_digest,
    _budget_loaders,
    budget_artifact_paths,
    budget_subset_identity,
    load_contract,
    planned_budget_specs,
    run_auxiliary_budget_sweep,
)
from src.watcher_state import classify_watcher_state


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT / "config" / "study_contract.json")
PROVENANCE = {
    "frozen_sha256": "f" * 64,
    "lock_sha256": "l" * 64,
    "contract_sha256": "c" * 64,
}


def _register_budget_matrix(root: Path) -> list[dict]:
    registry = RunRegistry(root / "manifests" / BUDGET_REGISTRY_DIRECTORY)
    return [registry.register(spec) for spec in planned_budget_specs(CONTRACT)]


def test_budget_matrix_identity_counts_and_nested_subsets_are_frozen() -> None:
    specs = planned_budget_specs(CONTRACT)
    run_ids = [deterministic_run_id(spec) for spec in specs]
    assert len(specs) == len(set(run_ids)) == 24
    assert _budget_run_id_digest(run_ids) == EXPECTED_BUDGET_RUN_ID_DIGEST
    for dataset, size, counts, fitting_counts, validation_counts in (
        ("celeba", 162_770, [16_277, 81_385, 162_770], [14_650, 73_247, 146_493], [1_627, 8_138, 16_277]),
        ("cifar10", 45_000, [4_500, 22_500, 45_000], [4_050, 20_250, 40_500], [450, 2_250, 4_500]),
    ):
        identities = [budget_subset_identity(size, fraction) for fraction in (0.1, 0.5, 1.0)]
        assert [identity["auxiliary_examples"] for identity in identities] == counts
        assert [identity["auxiliary_fitting_examples"] for identity in identities] == fitting_counts
        assert [identity["auxiliary_validation_examples"] for identity in identities] == validation_counts
        assert identities[1]["indices"][: counts[0]] == identities[0]["indices"]
        assert identities[2]["indices"][: counts[1]] == identities[1]["indices"]
        assert identities[1]["fitting_indices"][: fitting_counts[0]] == identities[0]["fitting_indices"]
        assert identities[2]["fitting_indices"][: fitting_counts[1]] == identities[1]["fitting_indices"]
        assert identities[1]["validation_indices"][: validation_counts[0]] == identities[0]["validation_indices"]
        assert identities[2]["validation_indices"][: validation_counts[1]] == identities[1]["validation_indices"]
        assert not set(identities[0]["fitting_indices"]) & set(identities[0]["validation_indices"])
        assert identities[0]["ordinary_validation_access"] is False
        assert expected_auxiliary_identity(dataset, 0.5)["auxiliary_index_digest"] == identities[1][
            "auxiliary_index_digest"
        ]


def test_auditor_expected_identity_is_independent_of_the_producer_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.pipeline.budget_subset_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the auditor must not call the producer's subset helper")
        ),
    )
    identity = expected_auxiliary_identity("cifar10", 0.5)
    assert identity["auxiliary_examples"] == 22_500
    assert identity["auxiliary_fitting_examples"] == 20_250
    assert identity["auxiliary_validation_examples"] == 2_250
    assert identity["auxiliary_index_digest"] == (
        "a84854c93554fe947a9706247765b94d745e0e5f1017c36af07cc7a0711f7fba"
    )


def test_budget_loaders_are_disjoint_and_exactly_exhaust_the_total_cap() -> None:
    fitting_view = TensorDataset(torch.arange(100))
    validation_view = TensorDataset(torch.arange(100) + 1_000)
    fitting_pool = DataLoader(fitting_view, batch_size=8, shuffle=False)
    validation_pool = DataLoader(validation_view, batch_size=8, shuffle=False)
    fitting, validation, identity = _budget_loaders(
        fitting_pool,
        validation_pool,
        0.5,
        2718,
    )
    fitting_indices = set(fitting.dataset.indices)
    validation_indices = set(validation.dataset.indices)
    assert len(fitting_indices) == identity["auxiliary_fitting_examples"] == 45
    assert len(validation_indices) == identity["auxiliary_validation_examples"] == 5
    assert not fitting_indices & validation_indices
    assert fitting_indices | validation_indices == set(budget_subset_identity(100, 0.5)["indices"])
    assert fitting.dataset.dataset is fitting_view
    assert validation.dataset.dataset is validation_view


def test_budget_pool_uses_deterministic_preprocessing_for_internal_validation(
    tmp_path: Path,
) -> None:
    (
        fitting_pool,
        validation_pool,
        _test_loader,
    ), _num_classes, _metadata = get_budget_pool_and_test_loaders(
        "cifar10",
        str(tmp_path),
        batch_size=4,
        seed=42,
        num_workers=0,
        test_only_synthetic=True,
        max_samples=20,
    )
    fitting_transforms = [
        type(step).__name__
        for step in fitting_pool.dataset.base_dataset.transform.transforms
    ]
    validation_transforms = [
        type(step).__name__
        for step in validation_pool.dataset.base_dataset.transform.transforms
    ]
    assert fitting_transforms == ["RandomCrop", "RandomHorizontalFlip", "ToTensor"]
    assert validation_transforms == ["ToTensor"]
    assert fitting_pool.dataset.indices == validation_pool.dataset.indices


def test_unknown_budget_only_run_id_fails_before_selection_device_or_data(monkeypatch, tmp_path: Path) -> None:
    revision = tmp_path / "revision"
    _register_budget_matrix(revision)
    monkeypatch.setattr("src.semantic_result_audit.require_canonical_semantic_audit", lambda *args: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("selection/device/data setup must not occur")

    monkeypatch.setattr("src.pipeline.load_sealed_selection", forbidden)
    monkeypatch.setattr("src.pipeline.resolve_device", forbidden)
    monkeypatch.setattr("src.pipeline.get_dataset_loaders", forbidden)
    monkeypatch.setattr("src.pipeline.get_budget_pool_and_test_loaders", forbidden)
    with pytest.raises(ValueError, match="Unknown budget --only-run-id"):
        run_auxiliary_budget_sweep(
            contract=CONTRACT,
            revision_root=revision,
            data_dir="unused",
            device_name="cuda",
            num_workers=0,
            only_run_id="not-registered",
        )


def test_budget_bulk_refuses_failed_or_running_records_before_setup(monkeypatch, tmp_path: Path) -> None:
    revision = tmp_path / "revision"
    records = _register_budget_matrix(revision)
    RunRegistry(revision / "manifests" / BUDGET_REGISTRY_DIRECTORY).update(records[0]["run_id"], "failed")
    monkeypatch.setattr("src.semantic_result_audit.require_canonical_semantic_audit", lambda *args: {})
    monkeypatch.setattr(
        "src.pipeline.load_sealed_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection setup must not occur")),
    )
    with pytest.raises(RuntimeError, match="bulk execution refuses"):
        run_auxiliary_budget_sweep(
            contract=CONTRACT,
            revision_root=revision,
            data_dir="unused",
            device_name="cuda",
            num_workers=0,
        )


def test_budget_collision_blocks_isolated_recovery_before_setup(monkeypatch, tmp_path: Path) -> None:
    revision = tmp_path / "revision"
    record = _register_budget_matrix(revision)[0]
    collision = budget_artifact_paths(revision, record["spec"], "deconv_mse")["checkpoint"]
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"preserve me")
    monkeypatch.setattr("src.semantic_result_audit.require_canonical_semantic_audit", lambda *args: {})
    monkeypatch.setattr(
        "src.pipeline.load_sealed_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection setup must not occur")),
    )
    with pytest.raises(RuntimeError, match="quarantining existing partial/conflicting artifacts"):
        run_auxiliary_budget_sweep(
            contract=CONTRACT,
            revision_root=revision,
            data_dir="unused",
            device_name="cpu",
            num_workers=0,
            only_run_id=record["run_id"],
        )
    assert collision.read_bytes() == b"preserve me"


def test_budget_only_run_id_recovers_exactly_one_cell(monkeypatch, tmp_path: Path) -> None:
    revision = tmp_path / "revision"
    records = _register_budget_matrix(revision)
    record = next(item for item in records if item["spec"]["method"] == "standard")
    RunRegistry(revision / "manifests" / BUDGET_REGISTRY_DIRECTORY).update(record["run_id"], "failed")
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"fixed encoder")
    calls: list[str] = []
    split_counts: list[tuple[int, int]] = []
    dataset = TensorDataset(
        torch.zeros(100, 3, 32, 32), torch.zeros(100, 3, 32, 32), torch.zeros(100, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=2)
    metadata = DatasetMetadata("celeba", 162_770, 19_867, 19_962, "canonical_boundaries", False)

    monkeypatch.setattr("src.semantic_result_audit.require_canonical_semantic_audit", lambda *args: {})
    monkeypatch.setattr("src.pipeline.load_sealed_selection", lambda **kwargs: ({}, PROVENANCE, {}))
    monkeypatch.setattr("src.pipeline.resolve_device", lambda name: torch.device("cpu"))
    monkeypatch.setattr(
        "src.pipeline._primary_record",
        lambda *args, **kwargs: {
            "result": {"encoder_checkpoint": str(encoder), "selected_defense_value": None}
        },
    )
    monkeypatch.setattr(
        "src.pipeline.get_budget_pool_and_test_loaders",
        lambda *args, **kwargs: ((loader, loader, loader), 2, metadata),
    )
    monkeypatch.setattr(
        "src.pipeline.get_dataset_loaders",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ordinary validation must not be accessed")),
    )
    monkeypatch.setattr(
        "src.pipeline.load_split_model",
        lambda *args, **kwargs: (
            SimpleNamespace(edge_encoder=object(), config=SimpleNamespace(feature_shape=(6, 16, 16))),
            {},
        ),
    )

    def fake_train(**kwargs):
        calls.append(kwargs["architecture"])
        split_counts.append((len(kwargs["train_loader"].dataset), len(kwargs["val_loader"].dataset)))
        checkpoint = Path(kwargs["checkpoint_path"])
        log = Path(kwargs["log_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": {}}, checkpoint)
        training = {
            "architecture": kwargs["architecture"],
            "selected_epoch": 1,
            "best_validation_loss": 1.0,
            "epochs_completed": 1,
            "patience": 5,
            "checkpoint_path": str(checkpoint),
            "history": [{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}],
            **kwargs["artifact_metadata"],
        }
        log.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return training

    class FakeAttacker:
        def to(self, device):
            return self

        def load_state_dict(self, state, strict=True):
            return None

    def fake_evaluate(**kwargs):
        sample = Path(kwargs["sample_bundle_path"])
        sample.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample": True}, sample)
        values = {"mse": 0.1, "psnr": 10.0, "ssim": 0.2, "lpips": 0.3}
        return {
            "summary": values,
            "per_image": {key: [value] for key, value in values.items()},
            "example_count": 1,
            "mc_draw_seeds": [1701],
            "mc_draw_count": 1,
            "draws_are_independent_replicates": False,
            "metric_versions": {},
        }

    monkeypatch.setattr("src.pipeline.train_adaptive_attacker", fake_train)
    monkeypatch.setattr("src.pipeline.evaluate_adaptive_attacker", fake_evaluate)
    monkeypatch.setattr("src.models.build_attacker", lambda *args, **kwargs: FakeAttacker())
    run_auxiliary_budget_sweep(
        contract=CONTRACT,
        revision_root=revision,
        data_dir="unused",
        device_name="cpu",
        num_workers=0,
        only_run_id=record["run_id"],
    )
    final = RunRegistry(revision / "manifests" / BUDGET_REGISTRY_DIRECTORY).records()
    assert calls == ["deconv_mse", "residual_lpips"]
    assert split_counts == [(9, 1), (9, 1)]
    assert sum(item["status"] == "complete" for item in final) == 1
    assert sum(item["status"] == "planned" for item in final) == 23
    assert next(item for item in final if item["run_id"] == record["run_id"])["spec"] == record["spec"]


def _common_identity() -> dict:
    subset = {
        key: value
        for key, value in budget_subset_identity(45_000, 0.1).items()
        if key not in {"indices", "fitting_indices", "validation_indices"}
    }
    return {
        "budget_artifact_schema_version": BUDGET_ARTIFACT_SCHEMA_VERSION,
        "budget_protocol_version": BUDGET_PROTOCOL_VERSION,
        "run_id": "run",
        "dataset": "cifar10",
        "split_point": "early",
        "seed": 42,
        "method": "standard",
        "selected_defense_value": None,
        "defense": {"method": "standard", "stochastic": False},
        "auxiliary_fraction": 0.1,
        **subset,
        "primary_encoder": {"path": "encoder.pt", "sha256": "a" * 64},
        "selection_provenance": PROVENANCE,
        "evaluation_partition": "test_after_frozen_selection",
        "metric_versions": {},
        "model_seeds_are_independent_replicates": False,
        "attackers_are_independent_replicates": False,
        "images_are_independent_replicates": False,
        "auxiliary_examples_are_independent_replicates": False,
    }


def test_budget_audit_validators_reject_corruption_and_accept_unfavorable_values() -> None:
    expected = _common_identity()
    validate_budget_common_identity(copy.deepcopy(expected), expected)
    altered = copy.deepcopy(expected)
    altered["auxiliary_index_digest"] = "0" * 64
    with pytest.raises(ResultAuditError, match="auxiliary_index_digest"):
        validate_budget_common_identity(altered, expected)
    promoted = copy.deepcopy(expected)
    promoted["images_are_independent_replicates"] = True
    with pytest.raises(ResultAuditError, match="images_are_independent_replicates"):
        validate_budget_common_identity(promoted, expected)
    sample = {
        **expected,
        "artifact_type": "budget_sample_bundle",
        "architecture": "deconv_mse",
        "original": torch.zeros(5, 3, 32, 32),
        "reconstruction": torch.ones(5, 3, 32, 32),
        "draw_seed": 1701,
    }
    validate_budget_sample_bundle(sample, expected=expected, architecture="deconv_mse")
    sample["draw_seed"] = 999
    with pytest.raises(ResultAuditError, match="wrong fixed draw seed"):
        validate_budget_sample_bundle(sample, expected=expected, architecture="deconv_mse")


def test_budget_audit_publication_is_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    outputs = {"budget_result_audit.json": b"{}\n", "budget_cells.csv": b"header\n"}
    assert publish_idempotent(tmp_path, outputs, check=False) == {
        "budget_result_audit.json": "created",
        "budget_cells.csv": "created",
    }
    assert all(value == "verified_existing" for value in publish_idempotent(tmp_path, outputs, check=True).values())
    (tmp_path / "budget_result_audit.json").write_bytes(b"conflict")
    with pytest.raises(RuntimeError, match="no artifact was overwritten"):
        publish_idempotent(tmp_path, outputs, check=False)
    assert (tmp_path / "budget_cells.csv").read_bytes() == b"header\n"
    assert publish_idempotent(tmp_path, outputs, check=False, replace_reviewed=True) == {
        "budget_result_audit.json": "replaced_after_review",
        "budget_cells.csv": "verified_existing",
    }
    assert (tmp_path / "budget_result_audit.json").read_bytes() == b"{}\n"


def _synthetic_audit() -> dict:
    cells = []
    attackers = []
    index = 0
    for dataset in ("celeba", "cifar10"):
        for split_point in ("early", "current"):
            for method in ("standard", "afd"):
                for fraction in (0.1, 0.5, 1.0):
                    run_id = f"run-{index}"
                    index += 1
                    base = 0.4 + 0.01 * index
                    cells.append(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "split_point": split_point,
                            "seed": 42,
                            "method": method,
                            "auxiliary_fraction": fraction,
                            "auxiliary_examples": 1,
                            "auxiliary_index_digest": "a" * 64,
                            "primary_encoder_path": "encoder.pt",
                            "primary_encoder_sha256": "b" * 64,
                            "deconv_mse_ssim": base,
                            "residual_lpips_ssim": base + 0.01,
                            "strongest_attacker": "residual_lpips",
                            "worst_case_test_ssim": base + 0.01,
                            "record_passed": True,
                        }
                    )
                    for architecture, offset in (("deconv_mse", 0.0), ("residual_lpips", 0.01)):
                        attackers.append(
                            {
                                "run_id": run_id,
                                "architecture": architecture,
                                "mse": 0.1,
                                "psnr": 10.0,
                                "ssim": base + offset,
                                "lpips": 0.2,
                            }
                        )
    return {"budget_cell_results": cells, "budget_attacker_results": attackers, "raw_evidence": []}


def test_budget_analysis_stays_single_seed_descriptive_and_replaces_only_placeholders(tmp_path: Path) -> None:
    analysis = build_budget_analysis(_synthetic_audit())
    assert len(analysis["rows"]) == 24
    assert len(analysis["afd_minus_standard_differences"]) == 12
    assert analysis["seed"] == 42
    assert analysis["statistical_boundary"]["fixed_conditions_not_replicates"] is True
    assert analysis["statistical_boundary"]["attackers_are_adversary_family_not_replicates"] is True
    for field in (
        "standard_deviations",
        "confidence_intervals",
        "hypothesis_tests",
        "multiple_testing_corrections",
        "effect_sizes",
        "method_rankings",
        "privacy_guarantees",
    ):
        assert analysis["statistical_boundary"][field] is False
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "consolidated_budget.json").write_text(
        json.dumps(
            {
                "expected_runs": 24,
                "registered_runs": 24,
                "complete_runs": 0,
                "all_planned_runs_complete": False,
                "records": [],
                "rows": [],
                "unit_of_analysis": "single predeclared seed 42 sensitivity analysis; not an inferential replicate set",
            }
        ),
        encoding="utf-8",
    )
    (metrics / "consolidated_budget.csv").write_text(
        "run_id,dataset,split_point,seed,method,auxiliary_fraction,auxiliary_examples,worst_case_test_ssim,strongest_attacker\n",
        encoding="utf-8",
    )
    outputs = {
        Path("metrics/consolidated_budget.json"): b"new-json\n",
        Path("metrics/consolidated_budget.csv"): b"new-csv\n",
        Path("analysis/budget_descriptive_analysis.json"): b"analysis\n",
    }
    states = publish_outputs(tmp_path, outputs, check=False)
    assert states["metrics/consolidated_budget.json"] == "replaced_placeholder"
    assert states["metrics/consolidated_budget.csv"] == "replaced_placeholder"
    assert publish_outputs(tmp_path, outputs, check=True)["analysis/budget_descriptive_analysis.json"] == "verified_existing"


def test_budget_watcher_stops_on_failed_cell_and_premature_exit() -> None:
    records = [{"status": "complete"}] * 3 + [{"status": "failed"}] + [{"status": "planned"}] * 20
    assert classify_watcher_state(records, {"ActiveState": "active"}) == "failed_cell"
    records = [{"status": "complete"}] * 3 + [{"status": "planned"}] * 21
    assert classify_watcher_state(records, {"ActiveState": "inactive"}) == "premature_service_exit"
    assert classify_watcher_state([{"status": "complete"}] * 24, {"ActiveState": "active"}) == "ready_to_finalize"


def test_figure_3_requires_exact_audited_consolidation(tmp_path: Path) -> None:
    analysis = build_budget_analysis(_synthetic_audit())
    consolidated = tmp_path / "consolidated_budget.json"
    consolidated.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the passing audit"):
        require_matching_consolidated(consolidated, analysis)
    from scripts.analyze_budget_results import render_outputs

    consolidated.write_bytes(render_outputs(analysis)[Path("metrics/consolidated_budget.json")])
    assert require_matching_consolidated(consolidated, analysis)["complete_runs"] == 24
