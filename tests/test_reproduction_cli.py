"""Public commands, atomic publication and preservation regressions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.audit_public_repository import EXPECTED_ALLOWLIST
from src.reproduction import cli, paper, supporting
from src.reproduction.integrity import SCIENTIFIC_HASHES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def public_copy(tmp_path: Path) -> Path:
    root = tmp_path / "public"
    for relative in EXPECTED_ALLOWLIST:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return root


def rewrite_supporting_provenance(root: Path, manifest: dict) -> None:
    bundle = root / "reproduction/supporting"
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    (bundle / "provenance.json").write_bytes(payload)
    sums = bundle / "SHA256SUMS"
    sums.write_text("".join(
        hashlib.sha256(payload).hexdigest() + "  provenance.json\n"
        if line.endswith("  provenance.json") else line + "\n"
        for line in sums.read_text().splitlines()
    ))


@pytest.mark.parametrize("name", ["primary_seed_results.csv", "laplace_draw_metrics.csv"])
@pytest.mark.parametrize("redirect", ["absolute", "relative", "parent"])
def test_supporting_rejects_self_consistent_input_redirection(public_copy: Path, tmp_path: Path,
                                                             name: str, redirect: str) -> None:
    manifest = json.loads((public_copy / "reproduction/supporting/provenance.json").read_text())
    item = manifest["inputs"][name]
    alternate = (public_copy if redirect == "relative" else tmp_path) / "alternate.csv"
    alternate.write_bytes((public_copy / item["path"]).read_bytes() + b"\n")
    item["path"] = str(alternate) if redirect == "absolute" else os.path.relpath(alternate, public_copy)
    item["sha256"] = hashlib.sha256(alternate.read_bytes()).hexdigest()
    rewrite_supporting_provenance(public_copy, manifest)
    completed = subprocess.run(
        [sys.executable, "-B", "scripts/reproduce.py", "--suite", "supporting", "--output", "rejected"],
        cwd=public_copy, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode != 0, completed.stdout
    assert "Input provenance mismatch" in completed.stderr
    assert not (public_copy / "rejected").exists()
    assert not list(public_copy.glob(".reproduce-*"))


@pytest.mark.parametrize("mutation", ["extra", "missing", "digest"])
def test_supporting_requires_exact_input_provenance(public_copy: Path, monkeypatch: pytest.MonkeyPatch,
                                                  mutation: str) -> None:
    manifest = json.loads((public_copy / "reproduction/supporting/provenance.json").read_text())
    if mutation == "extra":
        manifest["inputs"]["alias.csv"] = manifest["inputs"]["primary_seed_results.csv"].copy()
    elif mutation == "missing":
        del manifest["inputs"]["primary_seed_results.csv"]
    else:
        manifest["inputs"]["primary_seed_results.csv"]["sha256"] = "0" * 64
    rewrite_supporting_provenance(public_copy, manifest)
    monkeypatch.setattr(supporting, "PROJECT_ROOT", public_copy)
    monkeypatch.setattr(supporting, "ROOT", public_copy / "reproduction/supporting")
    with pytest.raises(ValueError, match="Input (membership|provenance) mismatch"):
        cli.run(suite="supporting", output=public_copy / "rejected")
    assert not (public_copy / "rejected").exists()
    assert not list(public_copy.glob(".reproduce-*"))


def test_supporting_parses_the_verified_input_bytes_once(public_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supporting, "PROJECT_ROOT", public_copy)
    monkeypatch.setattr(supporting, "ROOT", public_copy / "reproduction/supporting")
    reads = {public_copy / relative: 0 for relative in supporting.INPUT_PATHS.values()}
    original_read = Path.read_bytes

    def read_once(path: Path) -> bytes:
        if path in reads:
            reads[path] += 1
            assert reads[path] == 1, "Verified input was reopened before parsing"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    data = supporting.verify_inputs()
    assert sum(map(len, data.values())) == 556
    assert all(count == 1 for count in reads.values())


@pytest.mark.parametrize("name", ["primary_seed_results.csv", "laplace_draw_metrics.csv"])
def test_input_mutation_cannot_be_authorized_by_metadata(public_copy: Path, monkeypatch: pytest.MonkeyPatch,
                                                       name: str) -> None:
    manifest = json.loads((public_copy / "reproduction/supporting/provenance.json").read_text())
    item = manifest["inputs"][name]
    path = public_copy / item["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite_supporting_provenance(public_copy, manifest)
    sums = public_copy / "reproduction/supporting/SHA256SUMS"
    sums.write_text("".join(
        item["sha256"] + "  inputs/" + name + "\n"
        if line.endswith("  inputs/" + name) else line + "\n"
        for line in sums.read_text().splitlines()
    ))
    monkeypatch.setattr(supporting, "PROJECT_ROOT", public_copy)
    monkeypatch.setattr(supporting, "ROOT", public_copy / "reproduction/supporting")
    with pytest.raises(ValueError, match="Scientific checksum mismatch"):
        cli.run(suite="supporting", output=public_copy / "rejected")
    assert not (public_copy / "rejected").exists()
    assert not list(public_copy.glob(".reproduce-*"))


@pytest.mark.parametrize("inside", [False, True])
def test_source_archive_replay_then_strict_inventory(public_copy: Path, tmp_path: Path, inside: bool) -> None:
    output = (public_copy if inside else tmp_path) / "paper-replay-output"
    replay = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "scripts/reproduce.py", "--output", str(output)],
        cwd=public_copy, capture_output=True, text=True, timeout=60,
    )
    assert replay.returncode == 0, replay.stderr
    audit = subprocess.run([sys.executable, "-B", "scripts/audit_public_repository.py"],
                           cwd=public_copy, capture_output=True, text=True, timeout=60)
    if inside:
        assert audit.returncode != 0
        assert "paper-replay-output/verification.json" in audit.stderr
    else:
        assert audit.returncode == 0, audit.stderr
        assert json.loads(audit.stdout)["candidate_files"] == 162


def test_canonical_helpers_are_shared(public_copy: Path) -> None:
    from src import eng_compute
    from src.reproduction import compute_core, primary_figure
    from scripts import plot_results
    assert compute_core.student_t_interval is eng_compute.student_t_interval
    assert compute_core.ComputeContractError is eng_compute.ComputeContractError
    assert plot_results._figure_bytes is primary_figure._figure_bytes
    result = subprocess.run([sys.executable, "-I", "-B", "scripts/plot_results.py", "--help"],
                            cwd=public_copy, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("suite", ["paper", "supporting", "all"])
@pytest.mark.parametrize("outside", [False, True])
def test_documented_commands_publish_verified_suites(public_copy: Path, tmp_path: Path, suite: str, outside: bool) -> None:
    output = (tmp_path if outside else public_copy) / "outputs" / suite
    arguments = [sys.executable, "-B"]
    if suite == "paper":
        arguments += ["-I", "-S"]
    arguments += ["scripts/reproduce.py", "--output", str(output)]
    if suite != "paper":  # Exercise the default as documented.
        arguments += ["--suite", suite]
    result = subprocess.run(arguments, cwd=public_copy, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    if suite == "all":
        assert set(report["outcomes"]) == {"paper", "supporting"}
        assert (output / "paper/verification.json").is_file()
        assert (output / "supporting/reproduction-report.json").is_file()
    if suite in {"supporting", "all"}:
        supporting_report = report["outcomes"]["supporting"] if suite == "all" else report
        assert len(supporting_report["checks"]) == 8
        assert supporting_report["retained_signature_keys"] == ["2", "3", "5"]
        folder = output / "supporting" if suite == "all" else output
        assert {p.name for p in folder.glob("*.pdf")} == {
            "four-condition-accuracy-vs-ssim.pdf", "attacker-budget-sensitivity.pdf",
            "component-compute-tradeoffs.pdf",
        }
    assert not list(output.parent.glob(".reproduce-*"))


@pytest.mark.parametrize("tree", cli.PROTECTED_TREES)
def test_protected_destinations_are_rejected_without_writes(tree: str) -> None:
    path = ROOT / tree / "never-create-reproduction-output"
    with pytest.raises(ValueError, match="protected"):
        cli.run(suite="paper", output=path)
    assert not path.exists()


@pytest.mark.parametrize("suite", ["paper", "supporting", "all"])
@pytest.mark.parametrize("empty", [False, True])
def test_existing_destination_is_untouched(tmp_path: Path, suite: str, empty: bool) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    if not empty:
        (output / "retained.txt").write_bytes(b"original bytes\n")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        cli.run(suite=suite, output=output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("suite,relative", [
    ("paper", "reproduction/aggregate-data/followup-training-history.csv"),
    ("supporting", "reproduction/supporting/inputs/laplace_draw_metrics.csv"),
    ("all", "results/raw/semantic_seed_results.csv"),
])
@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_bad_inputs_leave_no_published_output(public_copy: Path, suite: str, relative: str, mutation: str) -> None:
    path = public_copy / relative
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    result = subprocess.run([sys.executable, "-B", "scripts/reproduce.py", "--suite", suite,
                             "--output", "failed-output"], cwd=public_copy, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not (public_copy / "failed-output").exists()
    assert not list(public_copy.glob(".reproduce-*"))


def test_paper_write_failure_cleans_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: Path, rows: list[dict]) -> None:
        path.write_text("partial output")
        raise OSError("injected write failure")
    monkeypatch.setattr(paper, "write", fail)
    with pytest.raises(OSError, match="injected"):
        cli.run(suite="paper", output=tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


def test_all_supporting_failure_does_not_publish_completed_paper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> dict:
        raise ValueError("injected figure verification failure")
    monkeypatch.setattr(supporting, "figures", fail)
    with pytest.raises(ValueError, match="injected"):
        cli.run(suite="all", output=tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("populated", [False, True])
def test_atomic_publication_never_replaces_concurrent_destination(tmp_path: Path, populated: bool) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "result").write_text("new")
    output = tmp_path / "destination"
    output.mkdir()
    if populated:
        (output / "old").write_text("original")
    with pytest.raises(OSError):
        cli.publish(staging, output)
    assert {p.name for p in output.iterdir()} == ({"old"} if populated else set())
    assert (staging / "result").read_text() == "new"


def test_symlink_and_custom_input_overlap_are_rejected(tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(ROOT / "reproduction", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cli.run(suite="paper", output=link / "new")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(ValueError, match="protected"):
        cli.run(suite="paper", data=inputs, output=inputs / "new")


def test_preserved_scientific_hashes_and_single_source_layout() -> None:
    for relative, expected in SCIENTIFIC_HASHES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert not (ROOT / "reproduction/scientific").exists()
    assert not (ROOT / "release/v1.3.0").exists()
    for name, digest in {
        "benchmark_v1.py": "4fb9d158863a8b885296436ac6119f6db523ea6b695016cdabb2b215002d83e5",
        "eng_compute.py": "cfe630d050055766b997c6f73e4bbf45f696ee85672d06957dce7bda405eace0",
        "analysis.py": "48a62474afc81d7872d6a078bd3778501a9b39c9c77db44b78cfa9ced40ba617",
    }.items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == digest
    assert "from src.analysis import" in (ROOT / "src/reproduction/supporting.py").read_text()
    from src.focused import components, training
    assert training.DiscoModel is components.DiscoModel
