from __future__ import annotations

import hashlib
from importlib import metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest
from torch import nn

from src.experiment import MultilabelProbe
from src.pipeline import DEFAULT_CONTRACT, build_parser, load_contract, planned_budget_specs, planned_primary_specs, planned_semantic_specs
from src.implementation_provenance import local_import_closure, validate_implementation_protocol
from src.metrics import (
    verify_lpips_weight_files,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "study_contract.json"


def test_public_contract_is_the_cli_default_and_matches_frozen_lock() -> None:
    assert DEFAULT_CONTRACT == CONTRACT_PATH
    assert Path(build_parser().parse_args(["status"]).contract) == CONTRACT_PATH
    lock = json.loads((ROOT / "results" / "protocol" / "frozen_selection_lock.json").read_text())
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == lock["study_contract"]["sha256"]
    assert lock["study_contract"]["path"] == "config/study_contract.json"
    assert lock["frozen_artifact"]["published"] is False


def test_public_contract_reconstructs_the_exact_fixed_matrices() -> None:
    contract = load_contract(CONTRACT_PATH)
    assert contract["selection_partition"] == "validation_only"
    assert contract["test_access_before_freeze"] is False
    assert len(planned_primary_specs(contract)) == 80
    assert len(planned_semantic_specs(contract)) == 40
    assert len(planned_budget_specs(contract)) == 24
    assert contract["seeds"] == [7, 42, 123, 2024, 2025]


def test_environment_record_is_sanitized_and_version_pinned() -> None:
    environment = json.loads((ROOT / "results" / "protocol" / "environment.json").read_text())
    assert environment["python"] == "3.11.5"
    assert environment["packages"]["torch"] == "2.9.0"
    assert environment["packages"]["scikit-learn"] == "1.6.1"
    assert environment["packages"]["Pillow"] == "11.1.0"
    assert environment["hardware_measurement_role"].startswith("execution environment only")
    assert "/home/" not in json.dumps(environment)


def test_implementation_supplement_matches_runtime_and_dependency_imports() -> None:
    supplement = json.loads((ROOT / "config" / "implementation_protocol_v2.json").read_text())
    assert supplement["study_contract"]["sha256"] == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert supplement["budget"]["ordinary_validation_access"] is False
    assert supplement["budget"]["split_seed"] == 2718
    assert supplement["semantic_probe"]["balanced_accuracy_threshold"] == 0.5
    assert supplement["lpips"]["backbone"] == "alex"
    assert metadata.version("scikit-learn") == "1.6.1"
    assert metadata.version("Pillow") == "11.1.0"


def test_implementation_supplement_disagreement_fails_closed(tmp_path: Path) -> None:
    supplement = json.loads((ROOT / "config" / "implementation_protocol_v2.json").read_text())
    supplement["metrics"]["ssim"] = "different runtime definition"
    path = tmp_path / "implementation_protocol_v2.json"
    path.write_text(json.dumps(supplement), encoding="utf-8")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="metrics"):
        validate_implementation_protocol(contract, CONTRACT_PATH, path)


def test_semantic_probe_architecture_matches_the_supplement() -> None:
    probe = MultilabelProbe((6, 8, 8))
    layers = tuple(probe.network)
    assert tuple(type(layer) for layer in layers) == (
        nn.Flatten,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    )
    assert (layers[1].in_features, layers[1].out_features) == (384, 256)
    assert layers[3].p == 0.2
    assert (layers[4].in_features, layers[4].out_features) == (256, 39)


def test_implementation_closure_and_pretrained_weight_hashes_are_complete() -> None:
    closure = local_import_closure()
    assert set(closure["entry_points"]) == {
        "experiment_execution",
        "selection",
        "metrics",
        "semantic_analysis",
        "statistical_analysis",
        "budget_audit",
        "budget_analysis",
    }
    assert {
        "src/pipeline.py",
        "src/experiment.py",
        "src/dataset.py",
        "src/metrics.py",
        "src/budget_result_audit.py",
        "src/implementation_provenance.py",
    } <= set(closure["files"])
    weights = verify_lpips_weight_files()
    assert weights["lpips_package_weight"]["sha256"] == (
        "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
    )
    assert weights["torchvision_alexnet_weight"]["sha256"] == (
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
    )


def test_public_runner_provisions_only_the_frozen_alexnet_weight() -> None:
    runner = (ROOT / "run_public_checks.sh").read_text(encoding="utf-8")
    assert "LPIPS_BACKBONE_WEIGHT_FILENAME" in runner
    assert "LPIPS_BACKBONE_WEIGHT_SHA256" in runner
    assert "lpips_backbone_weight_path" in runner
    assert "target = lpips_backbone_weight_path()" in runner
    assert "https://download.pytorch.org/models/" in runner
    assert "urllib.request.urlopen(url, timeout=120)" in runner
    assert 'response.geturl().startswith("https://")' in runner
    assert "temporary.replace(target)" in runner
    assert "temporary.unlink(missing_ok=True)" in runner
    assert "scripts/export_eng_compute_public.py --check --public-only" in runner
    assert "scripts/export_public_results.py --check --public-only" in runner


def test_public_release_is_self_contained_and_unpublished() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_eng_compute_public.py"),
            "--check",
            "--public-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["file_count"] == 19
    assert report["checksum_entries"] == 18
    assert report["csv_table_count"] == 8
    assert report["aggregate_rows"] == 556
    assert report["status"] == "staged_not_published"
    assert report["private_source_regeneration_verified"] is False
    assert report["external_action"] is False


def test_documented_cli_initializes_and_reads_a_repository_relative_workdir(tmp_path: Path) -> None:
    revision_root = tmp_path / "adaptive-reconstruction"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_revision_pipeline.py"),
    ]
    help_result = subprocess.run(
        [*command, "--help"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert help_result.returncode == 0
    assert "--revision-root" in help_result.stdout

    init_result = subprocess.run(
        [*command, "init", "--revision-root", str(revision_root)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert init_result.returncode == 0, init_result.stderr
    initialized = json.loads(init_result.stdout)
    assert initialized["planned_counts"] == {
        "adaptive_attackers_per_defense_evaluation": 2,
        "auxiliary_budget_evaluations": 24,
        "primary_defense_evaluations": 80,
        "primary_encoder_checkpoints": 40,
        "selection_adaptive_attacker_trainings": 104,
        "selection_defense_candidates": 52,
        "selection_encoder_checkpoints": 20,
        "semantic_probe_evaluations": 40,
    }
    assert (revision_root / "manifests" / "study_manifest.json").is_file()
    assert len(list((revision_root / "manifests" / "budget_v2").glob("*.json"))) == 24

    status_result = subprocess.run(
        [*command, "status", "--revision-root", str(revision_root)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert status_result.returncode == 0, status_result.stderr
    assert json.loads(status_result.stdout)["primary"]["registered"] == 80


def test_multistep_shell_wrappers_forward_the_same_arguments() -> None:
    ablation = (ROOT / "scripts" / "run_ablation.sh").read_text(encoding="utf-8")
    robustness = (ROOT / "scripts" / "run_seed_robustness.sh").read_text(encoding="utf-8")
    assert 'init "$@"' in ablation
    assert 'selection "$@"' in ablation
    assert 'primary "$@"' in robustness
    assert 'consolidate "$@"' in robustness
