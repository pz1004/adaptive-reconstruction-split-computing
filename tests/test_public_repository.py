"""Public repository, release, and disclosure-boundary regression tests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_public_repository import (
    EXPECTED_ALLOWLIST,
    EXPECTED_RELEASE_DECISION,
    _reader_for_audit,
    _verify_disclosure,
)
from scripts.export_public_results import (
    CORRECTED_PROMOTION_REPLACEABLE_OUTPUTS,
    PUBLIC_SOURCE_MODULES,
    RESULT_SPECS,
    _atomic_create,
    public_allowlist,
    verify_public_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CITATION_CFF = """cff-version: 1.2.0
message: "If you use this software, please cite it using the metadata below."
type: software
title: "Adaptive Reconstruction Split Computing"
authors:
  - family-names: "Jang"
    given-names: "Sooyoung"
version: 1.3.0
license: MIT
"""


def test_public_outputs_have_exact_counts_schemas_and_hashes() -> None:
    result = verify_public_outputs()
    assert result["passed"] is True
    assert result["result_rows_verified"] == 2308
    assert result["public_outputs_verified"] == 49
    assert result["allowlist_files"] == 114


def test_disclosure_manifest_has_exact_allowlist_and_licenses() -> None:
    manifest = json.loads(
        (ROOT / "results" / "protocol" / "disclosure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["public_allowlist"] == public_allowlist()
    assert manifest["result_row_count"] == 2308
    assert manifest["base_result_row_count"] == 1752
    assert manifest["supplemental_result_row_count"] == 556
    assert len(manifest["result_table_counts"]) == 13
    assert len(manifest["public_outputs"]) == 49
    assert manifest["semantic_attribute_boundary"] == {
        "published_indices": [index for index in range(40) if index != 31],
        "excluded_index": 31,
        "reason": "The supervised task-target column is excluded; indices are schema references, not anonymized labels.",
    }
    assert manifest["licenses"] == {
        "authored_code_schemas_and_documentation": "MIT",
        "contributor_owned_numeric_results_and_compute_analysis": "CC-BY-4.0",
        "third_party_data_and_assets": "not licensed or redistributed",
    }
    assert manifest["schema_version"] == 3
    assert manifest["repository_name"] == "adaptive-reconstruction-split-computing"
    assert "historical_protected_updates" not in manifest
    assert manifest["disclosure_date"] == "2026-09-30"
    assert manifest["release"] == {
        "version": "1.3.0",
        "path": "release/v1.3.0",
        "status": "authorized_for_repository_commit",
        "release_url": None,
        "release_date": None,
        "doi": None,
    }
    assert len(manifest["release_decision"]["scope"]) == 27
    assert manifest["release_decision"]["publication_authorized"] is True
    assert manifest["release_decision"] == EXPECTED_RELEASE_DECISION


def test_disclosure_manifest_rejects_self_authorizing_protected_updates() -> None:
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    manifest = json.loads(payloads["results/protocol/disclosure_manifest.json"])
    manifest["historical_protected_updates"] = {
        "src/pipeline.py": {
            "previous_sha256": "0" * 64,
            "current_sha256": "1" * 64,
            "reason": "self_authorized_change",
        }
    }
    payloads["results/protocol/disclosure_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(RuntimeError, match="Unexpected disclosure manifest keys"):
        _verify_disclosure(payloads.__getitem__)


def test_citation_metadata_is_exact_and_tampering_is_rejected() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert citation == EXPECTED_CITATION_CFF
    assert not re.search(
        r"(?m)^\s*(?:email|affiliation|orcid|doi|preferred-citation|repository-code|date-released):",
        citation,
        flags=re.IGNORECASE,
    )

    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    payloads["CITATION.cff"] = payloads["CITATION.cff"].replace(
        b"\nversion: 1.3.0\n",
        b"\nversion: 1.2.1\n",
    )
    with pytest.raises(RuntimeError, match="citation metadata"):
        _verify_disclosure(payloads.__getitem__)


def test_public_title_checksum_command_and_result_attribution_are_canonical() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    result_license = (ROOT / "LICENSE-RESULTS").read_text(encoding="utf-8")
    assert readme.startswith("# Adaptive Reconstruction Split Computing\n")
    assert "(cd release/v1.3.0 && sha256sum --check SHA256SUMS)" in readme
    assert (
        "Adaptive Reconstruction Split Computing aggregate results and analyses "
        "(version 1.3.0)" in result_license
    )
    assert "repository-code:" not in (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_public_checks_workflow_is_pinned_and_least_privilege() -> None:
    workflow = (ROOT / ".github" / "workflows" / "public-checks.yml").read_text(
        encoding="utf-8"
    )
    assert re.findall(r"(?m)^\s*uses:\s*(\S+)", workflow) == [
        "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
    ]
    assert "\n  pull_request:\n" in workflow
    assert "\n  push:\n    branches:\n      - main\n" in workflow
    assert "pull_request_target" not in workflow
    assert "permissions:\n  contents: read\n" in workflow
    assert not re.search(r"(?m)^\s*\w[\w-]*:\s*write\s*$", workflow)
    assert "secrets." not in workflow
    assert "persist-credentials: false" in workflow
    assert 'python-version: "3.11"' in workflow
    assert "timeout-minutes: 45" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "https://download.pytorch.org/whl/cpu" in workflow
    assert "-r requirements.txt" in workflow
    assert workflow.count("./run_public_checks.sh") == 1


def test_public_runner_uses_temporary_torch_hub_cache(tmp_path: Path) -> None:
    runner = (ROOT / "run_public_checks.sh").read_text(encoding="utf-8")
    assert "trap 'rm -rf -- \"${PUBLIC_CACHE_ROOT}\"' EXIT" in runner
    assert "target = lpips_backbone_weight_path()" in runner
    assert 'Path.home() / ".cache" / "torch"' not in runner

    temporary_torch_home = tmp_path / "temporary-torch"
    isolated_home = tmp_path / "home"
    environment = {
        **os.environ,
        "HOME": str(isolated_home),
        "TORCH_HOME": str(temporary_torch_home),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.metrics import lpips_backbone_weight_path; "
            "print(lpips_backbone_weight_path())",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = temporary_torch_home / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
    assert Path(completed.stdout.strip()) == expected
    assert not (
        isolated_home
        / ".cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "alexnet-owt-7be5be79.pth"
    ).exists()


def test_independent_auditor_rejects_mutable_ci_action_reference() -> None:
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    workflow_path = ".github/workflows/public-checks.yml"
    payloads[workflow_path] = payloads[workflow_path].replace(
        b"actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
        b"actions/checkout@v5",
    )

    with pytest.raises(RuntimeError, match="least-privilege policy"):
        _verify_disclosure(payloads.__getitem__)


def test_public_auditor_accepts_git_checkout_without_private_paths(tmp_path: Path) -> None:
    for relative in EXPECTED_ALLOWLIST:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "--all"], cwd=tmp_path, check=True)
    completed = subprocess.run(
        [sys.executable, "scripts/audit_public_repository.py"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["candidate_files"] == 114
    assert report["ignored_boundary_paths_verified"] == 12


def test_independent_auditor_rejects_overstated_release_permission() -> None:
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    manifest = json.loads(payloads["results/protocol/disclosure_manifest.json"])
    manifest["release_decision"]["third_party_permission_claimed"] = True
    payloads["results/protocol/disclosure_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(RuntimeError, match="release decision"):
        _verify_disclosure(lambda path: payloads[path])


def test_independent_auditor_rejects_license_notice_scope_drift() -> None:
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    payloads["LICENSE-RESULTS"] = payloads["LICENSE-RESULTS"].replace(
        b"results/raw/primary_seed_results.csv",
        b"results/raw/future_unreviewed_results.csv",
    )

    with pytest.raises(RuntimeError, match="Result license must name"):
        _verify_disclosure(lambda path: payloads[path])


def test_independent_auditor_rejects_release_payload_tampering() -> None:
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    relative = "release/v1.3.0/expected/compute_paired_effects.csv"
    payloads[relative] += b"\n"
    manifest = json.loads(payloads["results/protocol/disclosure_manifest.json"])
    manifest["public_outputs"][relative]["sha256"] = hashlib.sha256(
        payloads[relative]
    ).hexdigest()
    manifest["public_outputs"][relative]["size_bytes"] = len(payloads[relative])
    payloads["results/protocol/disclosure_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(RuntimeError, match="payload checksum differs"):
        _verify_disclosure(payloads.__getitem__)


def test_public_result_rows_exclude_private_and_per_example_fields() -> None:
    forbidden = {"attribute_name", "example_ids", "per_image", "path", "predictions"}
    for relative, (expected_count, fields) in RESULT_SPECS.items():
        with (ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert tuple(reader.fieldnames or ()) == fields
        assert len(rows) == expected_count
        assert forbidden.isdisjoint(fields)
        assert all(not value.startswith("/") and "/home/" not in value for row in rows for value in row.values())


def test_exporter_and_independent_auditor_have_matching_fixed_boundaries() -> None:
    assert set(public_allowlist()) == EXPECTED_ALLOWLIST
    source_files = {
        relative
        for relative in public_allowlist()
        if relative.startswith("src/") and relative.endswith(".py")
    }
    assert source_files == set(PUBLIC_SOURCE_MODULES)
    assert all((ROOT / relative).is_file() for relative in source_files)
    assert "src/benchmark_v1.py" not in source_files
    assert "tests/test_public_disclosure.py" not in EXPECTED_ALLOWLIST
    assert "tests/test_public_disclosure.py" not in public_allowlist()
    assert "tests/test_public_repository.py" in EXPECTED_ALLOWLIST
    assert "tests/test_public_repository.py" in public_allowlist()


def test_independent_auditor_rejects_duplicate_matrix_key_even_with_updated_hash() -> None:
    relative = "results/raw/primary_seed_results.csv"
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    lines = payloads[relative].decode("utf-8").splitlines()
    lines[2] = lines[1]
    tampered = ("\n".join(lines) + "\n").encode("utf-8")
    payloads[relative] = tampered
    manifest = json.loads(payloads["results/protocol/disclosure_manifest.json"])
    manifest["public_outputs"][relative]["sha256"] = hashlib.sha256(tampered).hexdigest()
    manifest["public_outputs"][relative]["size_bytes"] = len(tampered)
    payloads["results/protocol/disclosure_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(RuntimeError, match="Duplicate statistical-unit key"):
        _verify_disclosure(payloads.__getitem__)


def test_require_staged_selects_git_index_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_git(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        return subprocess.CompletedProcess(["git", *arguments], 0, stdout=b"staged", stderr=b"")

    monkeypatch.setattr("scripts.audit_public_repository._run_git", fake_git)
    reader = _reader_for_audit(require_staged=True, in_git=True)
    assert reader("README.md") == b"staged"
    assert calls == [["show", ":README.md"]]


def test_corrected_promotion_is_restricted_to_declared_derived_outputs(
    tmp_path: Path,
) -> None:
    assert CORRECTED_PROMOTION_REPLACEABLE_OUTPUTS == {
        "results/raw/budget_attacker_results.csv",
        "results/raw/budget_cell_results.csv",
        "results/protocol/environment.json",
        "results/protocol/disclosure_manifest.json",
    }
    protected = tmp_path / "primary_seed_results.csv"
    protected.write_bytes(b"protected\n")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        _atomic_create(protected, b"changed\n", allow_replace=False)
    assert protected.read_bytes() == b"protected\n"

    replaceable = tmp_path / "budget_cell_results.csv"
    replaceable.write_bytes(b"historical\n")
    assert _atomic_create(replaceable, b"corrected\n", allow_replace=True) == (
        "replaced_superseded_v1"
    )
    assert replaceable.read_bytes() == b"corrected\n"


def test_unsafe_pytorch_loading_is_bounded_and_prominently_disclosed() -> None:
    unsafe_opt_out = "weights_only" + "=False"
    offenders = []
    for relative in public_allowlist():
        path = ROOT / relative
        if path.suffix == ".py" and unsafe_opt_out in path.read_text(encoding="utf-8"):
            offenders.append(relative)
    assert offenders == ["src/checkpointing.py", "src/defenses.py"]
    assert "Never load an untrusted `.pt` file" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_consolidated_release_inventory_and_historical_schema_mapping() -> None:
    from scripts.export_eng_compute_public import (
        CSV_SPECS, EXPECTED_FILES, LEGACY_LOCATION_MAP, validate_public_release,
    )
    release = ROOT / 'release/v1.3.0'
    assert not (ROOT / 'release/v1.2.0').exists()
    assert len(EXPECTED_FILES) == 40
    assert len(LEGACY_LOCATION_MAP) == 17
    assert len(set(LEGACY_LOCATION_MAP.values())) == 17
    assert validate_public_release(release)['checksum_entries'] == 39
    for historical, current in LEGACY_LOCATION_MAP.items():
        schema = release / 'schemas' / (Path(historical).stem + '.schema.json')
        if historical.startswith('results/') and schema.exists():
            record = json.loads(schema.read_text())
            assert record['file'] == historical
            assert record['rows'] == CSV_SPECS[current][0]
            assert tuple(column['name'] for column in record['columns']) == CSV_SPECS[current][1]
    inputs = list((release / 'inputs').glob('*.csv'))
    assert len(inputs) == 7
    assert sum(len(list(csv.DictReader(path.open(newline='')))) for path in inputs) == 556
    evidence = release / 'evidence/benchmark_budget_attacker_rows.csv'
    with evidence.open(newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 240


@pytest.mark.parametrize('relative', [
    'analysis/compute_analysis.json',
    'evidence/benchmark_budget_attacker_rows.csv',
    'schemas/benchmark_budget_conditions.schema.json',
    'inputs/attacker_restart_rows.csv',
    'expected/compute_group_summary.csv',
])
@pytest.mark.parametrize('mutation', ['missing', 'altered'])
def test_consolidated_evidence_rejects_missing_or_altered_bytes(
    tmp_path: Path, relative: str, mutation: str,
) -> None:
    from scripts.export_eng_compute_public import validate_public_release
    release = tmp_path / 'release'
    shutil.copytree(ROOT / 'release/v1.3.0', release)
    path = release / relative
    if mutation == 'missing':
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b'\n')
        # Even a self-consistently rewritten inventory cannot authorize changed evidence.
        checksum = release / 'SHA256SUMS'
        lines = checksum.read_text().splitlines()
        checksum.write_text('\n'.join(
            hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + relative
            if line.endswith('  ' + relative) else line for line in lines
        ) + '\n')
    with pytest.raises(RuntimeError, match='inventory differs|payload checksum differs'):
        validate_public_release(release)


def test_consolidated_export_preserves_unrelated_files_and_preflights_conflicts(tmp_path: Path) -> None:
    from scripts.export_eng_compute_public import export_payloads
    reproduction = tmp_path / 'reproduce.py'
    reproduction.write_bytes(b'preserved reproduction implementation\n')
    evidence = tmp_path / 'evidence/table.csv'
    evidence.parent.mkdir()
    evidence.write_bytes(b'original\n')
    export_payloads(tmp_path, {'evidence/table.csv': b'original\n', 'analysis/extra.json': b'{}\n'})
    assert reproduction.read_bytes() == b'preserved reproduction implementation\n'
    assert evidence.read_bytes() == b'original\n'
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        export_payloads(tmp_path, {'first.txt': b'new\n', 'evidence/table.csv': b'different\n'})
    assert not (tmp_path / 'first.txt').exists()
    assert evidence.read_bytes() == b'original\n'


def test_consolidated_release_rejects_private_extras(tmp_path: Path) -> None:
    from scripts.export_eng_compute_public import validate_public_release
    release = tmp_path / 'release'
    shutil.copytree(ROOT / 'release/v1.3.0', release)
    (release / 'schemas/private-evidence.json').write_text('{}\n')
    with pytest.raises(RuntimeError, match='inventory differs'):
        validate_public_release(release)


def test_mapped_private_candidate_comparison(tmp_path: Path) -> None:
    from scripts.export_eng_compute_public import LEGACY_LOCATION_MAP, verify_private_candidate
    release = ROOT / 'release/v1.3.0'
    for historical, current in LEGACY_LOCATION_MAP.items():
        target = tmp_path / historical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((release / current).read_bytes())
    report = verify_private_candidate(release, tmp_path)
    assert report['mapped_scientific_files_verified'] == 17
    changed = tmp_path / 'results/benchmark_budget_attacker_rows.csv'
    changed.write_bytes(changed.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='differs from retained private candidate'):
        verify_private_candidate(release, tmp_path)


def test_release_validation_and_export_tolerate_generated_bytecode(tmp_path: Path, monkeypatch) -> None:
    import py_compile
    from scripts import export_eng_compute_public as exporter
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    shutil.copytree(ROOT / 'release/v1.3.0', source)
    py_compile.compile(str(source / 'compute_core.py'), doraise=True)
    assert list(source.rglob('*.pyc'))
    assert exporter.validate_public_release(source)['file_count'] == 40
    monkeypatch.setattr(exporter, 'DEFAULT_OUTPUT', source)
    monkeypatch.setattr(exporter, 'build_private_payloads', lambda: {
        current: (source / current).read_bytes()
        for current in exporter.LEGACY_LOCATION_MAP.values()
    })
    monkeypatch.setattr(exporter, 'verify_private_candidate', exporter.validate_public_release)
    monkeypatch.setattr(sys, 'argv', ['export', '--output', str(target)])
    assert exporter.main() == 0
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == exporter.EXPECTED_FILES
    (source / '__pycache__/private.json').write_text('{}\n')
    with pytest.raises(RuntimeError, match='inventory differs'):
        exporter.validate_public_release(source)


@pytest.mark.parametrize('relative', [
    'evidence/benchmark_budget_attacker_rows.csv',
    'schemas/benchmark_budget_conditions.schema.json',
    'expected/compute_group_summary.csv',
])
def test_export_restores_missing_private_derived_file(tmp_path: Path, monkeypatch, relative: str) -> None:
    from scripts import export_eng_compute_public as exporter
    source = tmp_path / 'release'
    shutil.copytree(ROOT / 'release/v1.3.0', source)
    expected = {current: (source / current).read_bytes()
                for current in exporter.LEGACY_LOCATION_MAP.values()}
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    (source / relative).unlink()
    monkeypatch.setattr(exporter, 'DEFAULT_OUTPUT', source)
    monkeypatch.setattr(exporter, 'build_private_payloads', lambda: expected)
    monkeypatch.setattr(exporter, 'verify_private_candidate', exporter.validate_public_release)
    monkeypatch.setattr(sys, 'argv', ['export', '--output', str(source)])
    assert exporter.main() == 0
    assert {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before


def test_full_export_rejects_changed_reproduction_input_before_writes(tmp_path: Path, monkeypatch) -> None:
    from scripts import export_eng_compute_public as release_exporter
    from scripts import export_public_results as exporter
    for relative in EXPECTED_ALLOWLIST:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    release = tmp_path / 'release/v1.3.0'
    monkeypatch.setattr(release_exporter, 'build_private_payloads', lambda: {
        current: (ROOT / 'release/v1.3.0' / current).read_bytes()
        for current in release_exporter.LEGACY_LOCATION_MAP.values()
    })
    changed = release / 'inputs/laplace_draw_metrics.csv'
    changed.write_bytes(changed.read_bytes() + b'\n')
    manifest = tmp_path / 'results/protocol/disclosure_manifest.json'
    before = manifest.read_bytes()
    monkeypatch.setattr(exporter, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(exporter, 'PROTOCOL_ROOT', manifest.parent)
    monkeypatch.setattr(exporter, 'MANIFEST_PATH', manifest)
    monkeypatch.setattr(sys, 'argv', ['export', '--promote-corrected'])
    with pytest.raises(RuntimeError, match='checksum|Scientific bytes'):
        exporter.main()
    assert manifest.read_bytes() == before


def test_full_export_preflights_all_conflicts_before_creating_files(tmp_path: Path, monkeypatch) -> None:
    from scripts import export_public_results as exporter
    existing = tmp_path / 'results/raw/primary_seed_results.csv'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'protected\n')
    monkeypatch.setattr(exporter, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(exporter, 'render_from_private_sources', lambda: {
        'first.txt': b'new\n', 'results/raw/primary_seed_results.csv': b'different\n',
    })
    monkeypatch.setattr(sys, 'argv', ['export'])
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        exporter.main()
    assert not (tmp_path / 'first.txt').exists()
    assert existing.read_bytes() == b'protected\n'


@pytest.mark.parametrize('layout', ['parent_symlink', 'root_symlink', 'broken_symlink', 'parent_file'])
def test_export_rejects_unsafe_destination_before_any_write(tmp_path: Path, layout: str) -> None:
    from scripts.export_eng_compute_public import export_payloads
    destination = tmp_path / 'destination'
    outside = tmp_path / 'outside'
    outside.mkdir()
    if layout == 'root_symlink':
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir()
        if layout == 'parent_symlink':
            (destination / 'evidence').symlink_to(outside, target_is_directory=True)
        elif layout == 'broken_symlink':
            (destination / 'evidence').symlink_to(outside / 'missing', target_is_directory=True)
        else:
            (destination / 'evidence').write_bytes(b'keep\n')
    with pytest.raises(RuntimeError, match='symlink|directory|overwrite'):
        export_payloads(destination, {'first.txt': b'new\n', 'evidence/table.csv': b'data\n'})
    assert not (destination / 'first.txt').exists()
    assert not list(outside.iterdir())


@pytest.mark.parametrize('relative', ['../escaped.csv', '/absolute.csv'])
def test_export_rejects_paths_outside_destination(tmp_path: Path, relative: str) -> None:
    from scripts.export_eng_compute_public import export_payloads
    with pytest.raises(RuntimeError, match='Invalid export path'):
        export_payloads(tmp_path / 'destination', {'first.txt': b'new\n', relative: b'data\n'})
    assert not (tmp_path / 'destination/first.txt').exists()


def test_export_conflicting_scientific_bytes_prevent_partial_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import export_eng_compute_public as exporter
    source = ROOT / 'release/v1.3.0'
    target = tmp_path / 'release'
    shutil.copytree(source, target)
    missing = target / 'analysis/compute_analysis.json'
    missing.unlink()
    changed = target / 'evidence/benchmark_budget_attacker_rows.csv'
    changed.write_bytes(changed.read_bytes() + b'\n')
    before = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    monkeypatch.setattr(exporter, 'DEFAULT_OUTPUT', source)
    monkeypatch.setattr(exporter, 'build_private_payloads', lambda: {
        current: (source / current).read_bytes() for current in exporter.LEGACY_LOCATION_MAP.values()
    })
    monkeypatch.setattr(sys, 'argv', ['export', '--output', str(target)])
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        exporter.main()
    assert {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()} == before


def test_release_validator_rejects_symlinked_cache(tmp_path: Path) -> None:
    from scripts.export_eng_compute_public import validate_public_release
    release = tmp_path / 'release'
    shutil.copytree(ROOT / 'release/v1.3.0', release, ignore=shutil.ignore_patterns('__pycache__'))
    (release / '__pycache__').symlink_to(tmp_path / 'outside', target_is_directory=True)
    with pytest.raises(RuntimeError, match='symlink'):
        validate_public_release(release)
