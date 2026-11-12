"""Exercise the paper replay in isolation and protect publication provenance."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.paper_repository_policy import PAPER_HASHES, PAPER_METADATA, verify_paper_reproduction

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_replay(tmp_path: Path) -> Path:
    from scripts.audit_public_repository import EXPECTED_ALLOWLIST
    package = tmp_path / "paper"
    for relative in EXPECTED_ALLOWLIST:
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return package



def run_replay(package: Path) -> subprocess.CompletedProcess[str]:
    # -I ignores user paths; -S disables site packages to prove stdlib portability.
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", "reproduction/replay_aggregates.py", "--data", "reproduction/aggregate-data",
         "--output", "replay-output"], cwd=package, text=True, capture_output=True, timeout=30,
    )


def test_paper_replay_runs_without_installed_packages_or_research_workspace(isolated_replay: Path) -> None:
    completed = run_replay(isolated_replay)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS_CPU_AGGREGATE_REPLAY"
    assert report["descriptive_rows"] == 1140
    assert report["numerical_comparisons"] == 8227
    assert report["max_absolute_error"] <= 1e-12
    assert report["gpu_work"] == report["raw_image_inference"] == 0
    assert report["cohort_sizes"] == {
        "original_five": 5, "added_five": 5, "combined_ten": 10,
        "combined_nine_excluding42": 9, "original_four_excluding42": 4,
    }


def test_paper_replay_missing_input_does_not_publish_partial_outputs(isolated_replay: Path) -> None:
    (isolated_replay / "reproduction/aggregate-data/followup-training-history.csv").unlink()
    completed = run_replay(isolated_replay)
    assert completed.returncode != 0
    assert "Required aggregate input missing" in completed.stderr
    assert not (isolated_replay / "replay-output").exists()


def test_paper_replay_rejects_inconsistent_values_before_output(isolated_replay: Path) -> None:
    path = isolated_replay / "reproduction/aggregate-data/focused-all-primary-individual.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]["max_decoder_ssim"] = "999"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    completed = run_replay(isolated_replay)
    assert completed.returncode != 0
    assert "Scientific checksum mismatch" in completed.stderr
    assert not (isolated_replay / "replay-output").exists()


def test_paper_replay_preserves_existing_output(isolated_replay: Path) -> None:
    output = isolated_replay / "replay-output"
    output.mkdir()
    marker = output / "retained.txt"
    marker.write_text("retain this output\n")
    completed = run_replay(isolated_replay)
    assert completed.returncode != 0
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "retain this output\n"


def test_paper_hashes_cannot_be_self_authorized_by_checksum_changes() -> None:
    from scripts.audit_public_repository import EXPECTED_ALLOWLIST
    payloads = {path: (ROOT / path).read_bytes() for path in EXPECTED_ALLOWLIST}
    name = "reproduction/aggregate-data/focused-all-primary-individual.csv"
    original = hashlib.sha256(payloads[name]).hexdigest()
    payloads[name] += b"\n"
    updated = hashlib.sha256(payloads[name]).hexdigest()
    payloads["reproduction/SHA256SUMS"] = payloads["reproduction/SHA256SUMS"].replace(
        original.encode(), updated.encode(),
    )
    with pytest.raises(RuntimeError, match="Paper reproduction hash mismatch"):
        verify_paper_reproduction(payloads.__getitem__)


def test_all_paper_tables_have_exact_license_scope() -> None:
    assert verify_paper_reproduction(lambda name: (ROOT / name).read_bytes())["tables_verified"] == 16
    missing = "reproduction/aggregate-data/followup-disco-candidates.csv"

    def read_without_license(name: str) -> bytes:
        payload = (ROOT / name).read_bytes()
        return payload.replace((missing + "\n").encode(), b"") if name == "LICENSE-RESULTS" else payload

    with pytest.raises(RuntimeError, match="exact license scope"):
        verify_paper_reproduction(read_without_license)


def test_paper_identity_and_status_match_across_reader_documents() -> None:
    for name in ["README.md", "reproduction/README.md", "CITATION.cff"]:
        text = (ROOT / name).read_text()
        assert PAPER_METADATA["title"] in text
        assert "under review" in text
        assert not re.search(r"\b(revised|resubmission|revision package|reviewer response)\b", text, re.I)
    citation = (ROOT / "CITATION.cff").read_text()
    preferred = citation.split("preferred-citation:\n", 1)[1]
    assert re.findall(r'family-names: "([^"]+)"', preferred) == ["Jang", "Woo", "Lee"]
    assert "type: unpublished" in preferred
    assert not re.search(r"(?m)^\s*(doi|date-released|journal):", citation)


def test_study_entry_point_exposes_cli_without_running_a_study() -> None:
    completed = subprocess.run([sys.executable, "-B", "scripts/run_study.py", "--help"],
                               cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "--contract" in completed.stdout
    assert "--revision-root" in completed.stdout
