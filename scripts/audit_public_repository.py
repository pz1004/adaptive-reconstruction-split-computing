#!/usr/bin/env python3
"""Independently audit public disclosure bytes, schemas, and the Git boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ALLOWLIST = frozenset(
    {
        ".github/workflows/public-checks.yml",
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "LICENSE-RESULTS",
        "README.md",
        "THIRD_PARTY_DATA.md",
        "config/study_contract.json",
        "config/implementation_protocol_v2.json",
        "pytest.ini",
        "requirements-figures.txt",
        "requirements.txt",
        "results/protocol/disclosure_manifest.json",
        "results/protocol/environment.json",
        "results/protocol/frozen_selection_lock.json",
        "results/raw/budget_attacker_results.csv",
        "results/raw/budget_cell_results.csv",
        "results/raw/primary_seed_results.csv",
        "results/raw/semantic_attribute_results.csv",
        "results/raw/semantic_seed_results.csv",
        "release/v1.2.0/RELEASE_NOTES.md",
        "release/v1.2.0/SHA256SUMS",
        "release/v1.2.0/analysis/compute_analysis.json",
        "release/v1.2.0/analysis/compute_claim_audit.json",
        "release/v1.2.0/results/attacker_restart_rows.csv",
        "release/v1.2.0/results/attacker_restart_summaries.csv",
        "release/v1.2.0/results/benchmark_budget_attacker_rows.csv",
        "release/v1.2.0/results/benchmark_budget_conditions.csv",
        "release/v1.2.0/results/benchmark_budget_paired_effects.csv",
        "release/v1.2.0/results/compute_group_summary.csv",
        "release/v1.2.0/results/compute_interface_effects.csv",
        "release/v1.2.0/results/compute_paired_effects.csv",
        "release/v1.2.0/schemas/attacker_restart_rows.schema.json",
        "release/v1.2.0/schemas/attacker_restart_summaries.schema.json",
        "release/v1.2.0/schemas/benchmark_budget_attacker_rows.schema.json",
        "release/v1.2.0/schemas/benchmark_budget_conditions.schema.json",
        "release/v1.2.0/schemas/benchmark_budget_paired_effects.schema.json",
        "release/v1.2.0/schemas/benchmark_v1_tables.schema.json",
        "release/v1.2.0/schemas/eng_compute_v1.schema.json",
        "run_public_checks.sh",
        "scripts/analyze_budget_results.py",
        "scripts/analyze_revision.py",
        "scripts/analyze_semantic_results.py",
        "scripts/audit_budget_results.py",
        "scripts/audit_primary_results.py",
        "scripts/audit_public_repository.py",
        "scripts/audit_semantic_results.py",
        "scripts/export_eng_compute_public.py",
        "scripts/export_public_results.py",
        "scripts/generate_images.py",
        "scripts/generate_protocol_figures.py",
        "scripts/generate_result_figures.py",
        "scripts/plot_results.py",
        "scripts/run_ablation.sh",
        "scripts/run_eval.sh",
        "scripts/run_revision_pipeline.py",
        "scripts/run_seed_robustness.sh",
        "scripts/run_train.sh",
        "scripts/summarize_seed_robustness.py",
        "scripts/train_shredder.py",
        "scripts/verify_budget_figure.py",
        "src/analysis.py",
        "src/budget_result_audit.py",
        "src/checkpointing.py",
        "src/dataset.py",
        "src/defenses.py",
        "src/eval.py",
        "src/experiment.py",
        "src/implementation_provenance.py",
        "src/manifest.py",
        "src/metrics.py",
        "src/models.py",
        "src/pipeline.py",
        "src/posthoc_attacker_eval.py",
        "src/protocol.py",
        "src/result_audit.py",
        "src/selection_provenance.py",
        "src/semantic_analysis.py",
        "src/semantic_result_audit.py",
        "src/train.py",
        "src/watcher_state.py",
        "tests/__init__.py",
        "tests/test_budget_result_audit.py",
        "tests/test_code_scripts.py",
        "tests/test_dataset_remedy.py",
        "tests/test_public_repository.py",
        "tests/test_public_protocol.py",
        "tests/test_result_audit.py",
        "tests/test_revision_protocol.py",
        "tests/test_selection_provenance.py",
        "tests/test_semantic_result_audit.py",
    }
)

V13_CHECKSUMS = {'release/v1.3.0/DATA_DICTIONARY.md': 'c6ab93a52c632fcfbc8d6183f9ce87c9c63fc344d2ca9960d79d3adef8b25cc3',
 'release/v1.3.0/README.md': '00fe8309d7d7ecbc71206404c95a032a4be0ee79193c557d57b6f0621420d824',
 'release/v1.3.0/SHA256SUMS': '7b80ff733985ff12eb2a5e014ec19a307ea78628926cd6a443d7bbd7f6f364f9',
 'release/v1.3.0/analysis_contract.json': '45124d8e98488df044c8eb672dd9e19a42e04a7320106b4319de9ec47ad1d277',
 'release/v1.3.0/budget_figure.py': '49bb7845009a0828fb3ef8667dee1f5584fa6d7d130156ef007ed68fd76f09ae',
 'release/v1.3.0/compute_core.py': 'e1a753a324ccfb00335b4cd9b5a1c323f159a2e7468100695a3802a9eb88de3d',
 'release/v1.3.0/diagnostics/known-offset-diagnostic.csv': '70862589f7e788891ad8d810cbef8d54757108b6ac9169eb0aee447f997a937b',
 'release/v1.3.0/diagnostics/known-offset-diagnostic.json': '3679eb943b4fb948eb64f0941b1125c2609e4ffd2db9f70958ec1a95b3096089',
 'release/v1.3.0/expected/attacker_restart_summaries.csv': '89cd56dbb25d57afac00e44012e1c87ecfd32e3c065313a24355c0379e890618',
 'release/v1.3.0/expected/benchmark_budget_paired_effects.csv': '8312b09c91612e77b1c35fd713e47ffecf85dfc5a122461ff0aa4ac26c7ce0dc',
 'release/v1.3.0/expected/budget_method_summaries.csv': 'd2c3fbb4ea925d063e04c5131aa571d8f603298af298e06d0bfe7b3788a37d34',
 'release/v1.3.0/expected/compute_group_summary.csv': '402f17a7ff585944f39f13011e0fecb9b79853cd45af0b04208edbbcf2c90681',
 'release/v1.3.0/expected/compute_interface_effects.csv': '3f77831cb296e218030e4c46bbc02cbd7ac4249e2b834e4e43b7773ab5b79164',
 'release/v1.3.0/expected/compute_paired_effects.csv': '1ab4edfab53bfefc40beb0022b25a31249a164cc4ca2f62e2950be3240b8c525',
 'release/v1.3.0/expected/figure_signatures.json': '9a85f375f44a2490ffe1bdc2fde892d476d15c12e723e926097682a1fd700ed9',
 'release/v1.3.0/expected/figure_sources.json': '06fb5ca3aff79413898d3a3daf9540e38ef8a0dbaa250a9add682c572eadd356',
 'release/v1.3.0/expected/laplace-single-release-statistics.csv': '33cc3084a7467577e9c541eb72d3107210fefb7afa44fde628a7316b4e28349c',
 'release/v1.3.0/expected/supplementary-statistics.csv': 'a387a651be20f087faf91f2f7b6d5a89552e211604b8741559243229e18908b7',
 'release/v1.3.0/inputs/attacker_restart_rows.csv': 'b1da933c28665b186b27f081401c9a5445489acb4f6224d8c620c16b07670465',
 'release/v1.3.0/inputs/benchmark_budget_conditions.csv': '6d5961a3bae709f1d98e1630bd1e1f90736ae9c06f93199c58685c35f6a8e9c7',
 'release/v1.3.0/inputs/compute_condition_metrics.csv': 'beab01892dd8505e889b4f387ebd395efb6495345b15c171fbb6da9184c74f25',
 'release/v1.3.0/inputs/laplace_draw_metrics.csv': 'b60fc38b2cf5566c3f78975e1c81d85100ce8e054b3fbf60b3390b48e246b023',
 'release/v1.3.0/inputs/laplace_run_metrics.csv': 'df9c810a8c476f5920c6615ccdc4a92e0706b28c933db98f4be417a1f103b395',
 'release/v1.3.0/inputs/primary_seed_results.csv': '4a1cec03db1db89a9e2b010ebeeff530a6f88fe3d23e3601db19553ce96b266d',
 'release/v1.3.0/inputs/semantic_seed_results.csv': 'd2c3bd11ebbc9277ee8fac7b66ce2b5e3edd3959be3aa0b6a1fc3ef2971eca90',
 'release/v1.3.0/primary_figure.py': '6e804e59c2488edab83d9cdae7a10f8e865450a87ad13c6d439d76439408b361',
 'release/v1.3.0/provenance.json': '5488e382fef513fe093d83dc258be7c2081e2d0d29f8ca13216ff6d9d18e5e19',
 'release/v1.3.0/reproduce.py': '8dac6400dda09011bbb29da676d9970bac78f0cab455cb000d2179003bc4925c',
 'release/v1.3.0/requirements.txt': '8ced0882fae5b9298075703d6456f46d729d3ad239034badf90f6bf6bf95aaf9',
 'release/v1.3.0/statistics_core.py': '48a62474afc81d7872d6a078bd3778501a9b39c9c77db44b78cfa9ced40ba617'}
V13_FILES = frozenset(V13_CHECKSUMS)
V13_LICENSE_SCOPE = ('release/v1.3.0/diagnostics/known-offset-diagnostic.csv',
 'release/v1.3.0/diagnostics/known-offset-diagnostic.json',
 'release/v1.3.0/expected/attacker_restart_summaries.csv',
 'release/v1.3.0/expected/benchmark_budget_paired_effects.csv',
 'release/v1.3.0/expected/budget_method_summaries.csv',
 'release/v1.3.0/expected/compute_group_summary.csv',
 'release/v1.3.0/expected/compute_interface_effects.csv',
 'release/v1.3.0/expected/compute_paired_effects.csv',
 'release/v1.3.0/expected/figure_signatures.json',
 'release/v1.3.0/expected/figure_sources.json',
 'release/v1.3.0/expected/laplace-single-release-statistics.csv',
 'release/v1.3.0/expected/supplementary-statistics.csv',
 'release/v1.3.0/inputs/attacker_restart_rows.csv',
 'release/v1.3.0/inputs/benchmark_budget_conditions.csv',
 'release/v1.3.0/inputs/compute_condition_metrics.csv',
 'release/v1.3.0/inputs/laplace_draw_metrics.csv',
 'release/v1.3.0/inputs/laplace_run_metrics.csv',
 'release/v1.3.0/inputs/primary_seed_results.csv',
 'release/v1.3.0/inputs/semantic_seed_results.csv')
EXPECTED_ALLOWLIST |= V13_FILES | {"tests/test_aggregate_reproduction.py"}

RESULT_SPECS: dict[str, tuple[int, tuple[str, ...]]] = {
    "results/raw/primary_seed_results.csv": (
        80,
        (
            "run_id", "dataset", "split_point", "seed", "method",
            "selected_defense_value", "accuracy_percentage_points",
            "utility_loss_vs_standard_percentage_points",
            "test_utility_loss_exceeds_validation_limit", "worst_case_ssim",
            "strongest_attacker", "validation_selected_attacker", "record_passed",
        ),
    ),
    "results/raw/semantic_seed_results.csv": (
        40,
        (
            "run_id", "dataset", "split_point", "seed", "method",
            "selected_defense_value", "macro_auroc", "macro_balanced_accuracy",
            "attribute_auroc_sample_sd", "attribute_balanced_accuracy_sample_sd",
            "semantic_draw_count", "semantic_draw_seeds", "record_passed",
        ),
    ),
    "results/raw/semantic_attribute_results.csv": (
        1560,
        (
            "run_id", "dataset", "split_point", "seed", "method",
            "attribute_index", "auroc", "balanced_accuracy",
        ),
    ),
    "results/raw/budget_cell_results.csv": (
        24,
        (
            "run_id", "dataset", "split_point", "seed", "method",
            "auxiliary_fraction", "auxiliary_examples", "auxiliary_fitting_examples",
            "auxiliary_validation_examples", "deconv_mse_ssim",
            "residual_lpips_ssim", "strongest_attacker", "worst_case_test_ssim",
            "record_passed",
        ),
    ),
    "results/raw/budget_attacker_results.csv": (
        48,
        (
            "run_id", "dataset", "split_point", "seed", "method",
            "auxiliary_fraction", "auxiliary_examples", "auxiliary_fitting_examples",
            "auxiliary_validation_examples", "architecture",
            "selected_epoch", "epochs_completed", "best_validation_loss", "mse",
            "psnr", "ssim", "lpips", "record_passed",
        ),
    ),
}

RELEASE_RESULT_COUNTS = {
    "release/v1.2.0/results/attacker_restart_rows.csv": 96,
    "release/v1.2.0/results/attacker_restart_summaries.csv": 32,
    "release/v1.2.0/results/benchmark_budget_attacker_rows.csv": 240,
    "release/v1.2.0/results/benchmark_budget_conditions.csv": 120,
    "release/v1.2.0/results/benchmark_budget_paired_effects.csv": 12,
    "release/v1.2.0/results/compute_group_summary.csv": 32,
    "release/v1.2.0/results/compute_interface_effects.csv": 16,
    "release/v1.2.0/results/compute_paired_effects.csv": 8,
}
EXPECTED_RELEASE_CHECKSUMS = {
    "RELEASE_NOTES.md": "cd07d1c9dbf55a3e63d8fcb0dbc7855637335572251949dcd2ff42a5a17fb911",
    "analysis/compute_analysis.json": "262741aee3a21052f7c3d7e901f4c5943e0ac26b09ad41d39679bb9e9a4a67f1",
    "analysis/compute_claim_audit.json": "001a6e433d6d3a1749224bfe8533ee1313690e2e78c31f0f2a7c62f36fcd3bd6",
    "results/attacker_restart_rows.csv": "b1da933c28665b186b27f081401c9a5445489acb4f6224d8c620c16b07670465",
    "results/attacker_restart_summaries.csv": "89cd56dbb25d57afac00e44012e1c87ecfd32e3c065313a24355c0379e890618",
    "results/benchmark_budget_attacker_rows.csv": "0523fb0265b21f7850742645d728441c21c63f847c941f63cfb41a469b79e49a",
    "results/benchmark_budget_conditions.csv": "6d5961a3bae709f1d98e1630bd1e1f90736ae9c06f93199c58685c35f6a8e9c7",
    "results/benchmark_budget_paired_effects.csv": "8312b09c91612e77b1c35fd713e47ffecf85dfc5a122461ff0aa4ac26c7ce0dc",
    "results/compute_group_summary.csv": "402f17a7ff585944f39f13011e0fecb9b79853cd45af0b04208edbbcf2c90681",
    "results/compute_interface_effects.csv": "3f77831cb296e218030e4c46bbc02cbd7ac4249e2b834e4e43b7773ab5b79164",
    "results/compute_paired_effects.csv": "1ab4edfab53bfefc40beb0022b25a31249a164cc4ca2f62e2950be3240b8c525",
    "schemas/attacker_restart_rows.schema.json": "9a36bdd36307a32610a4980dfba9213d0e75ded5338771c1dda2fa6e12ee6dce",
    "schemas/attacker_restart_summaries.schema.json": "ce6985e37b110c0b9265bc34ce815bad8f4edd71b85d61a543455248dd7d81bf",
    "schemas/benchmark_budget_attacker_rows.schema.json": "32b7203b5f3d606095c0a0f689ad971c86533aefcf835900c4a4bd06583babc6",
    "schemas/benchmark_budget_conditions.schema.json": "2ab0feb1e3d5c3a6dc02fd2f8146671ba51b5ef655d365edb52eb703e0bb1471",
    "schemas/benchmark_budget_paired_effects.schema.json": "8e6a8338fcbdd887b9935ce4ceb282c05a29dd50eeb21803e8ae2c576d74c715",
    "schemas/benchmark_v1_tables.schema.json": "900ce37be4ff8df523ffa4a97888a5accd7098a297492dbb8b09e1ad360c08f3",
    "schemas/eng_compute_v1.schema.json": "5e2603047b488be03c800de0c897ce0ac0134c44d8e546d684e42b2639f498a1",
}
RELEASE_FILES = frozenset(
    {"release/v1.2.0/SHA256SUMS"}
    | {f"release/v1.2.0/{relative}" for relative in EXPECTED_RELEASE_CHECKSUMS}
)

PUBLIC_OUTPUTS = frozenset(
    set(RESULT_SPECS)
    | {
        "config/study_contract.json",
        "config/implementation_protocol_v2.json",
        "results/protocol/environment.json",
        "results/protocol/frozen_selection_lock.json",
    }
    | set(RELEASE_FILES)
    | set(V13_FILES)
)
SOURCE_INPUT_ROLES = frozenset(
    {
        "budget_result_audit", "environment", "frozen_selection_lock", "implementation_protocol",
        "primary_seed_results", "semantic_attribute_results",
        "semantic_seed_results", "study_contract",
    }
)
EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "schema_version", "repository_name", "repository_description",
        "disclosure_date", "result_row_count", "base_result_row_count",
        "supplemental_result_row_count", "result_table_counts", "release",
        "semantic_attribute_boundary", "source_inputs", "public_outputs",
        "public_allowlist", "excluded", "licenses", "release_decision",
    }
)
EXPECTED_RELEASE_DECISION = {
    "status": "authorized_for_repository_commit",
    "staged_on": "2026-09-30",
    "scope": sorted(
        set(RESULT_SPECS)
        | set(V13_LICENSE_SCOPE)
        | set(RELEASE_RESULT_COUNTS)
        | {
            "release/v1.2.0/analysis/compute_analysis.json",
            "release/v1.2.0/analysis/compute_claim_audit.json",
        }
    ),
    "license": "CC-BY-4.0 for contributor-owned rights only",
    "publication_authorized": True,
    "third_party_permission_claimed": False,
    "third_party_terms_superseded": False,
}
EXPECTED_RELEASE_STATE = {
    "version": "1.3.0",
    "path": "release/v1.3.0",
    "status": "authorized_for_repository_commit",
    "release_url": None,
    "release_date": None,
    "doi": None,
}
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

BANNED_SUFFIXES = {
    ".7z", ".aux", ".bib", ".bz2", ".eps", ".gif", ".gz", ".jpeg", ".jpg",
    ".log", ".npz", ".pdf", ".png", ".pt", ".tar", ".tex", ".tgz", ".tif",
    ".tiff", ".webp", ".xz", ".zip",
}
BANNED_BASENAMES = {".env", "credentials.json", "id_ed25519", "id_rsa"}
IGNORED_BOUNDARY = (
    "main.tex", "main.pdf", "references.bib", "figures", "submission", "audit",
    "analysis-output", "revisions", "data", "task_plan.md", "notes.md", "deprecated",
)
CONTENT_PATTERNS = {
    "local_home_path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "email_address": re.compile(r"(?<![\w.-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"),
    "openai_token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|password|secret|access[_-]?token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}
MAX_PUBLIC_FILE_BYTES = 1 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=False, capture_output=True
    )


def _inside_git() -> bool:
    result = _run_git(["rev-parse", "--is-inside-work-tree"])
    return result.returncode == 0 and result.stdout.strip() == b"true"


def _git_candidates() -> set[str]:
    result = _run_git(["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def _staged_index() -> dict[str, str]:
    result = _run_git(["ls-files", "--stage", "-z"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    entries: dict[str, str] = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, _object_id, stage = metadata.split()
        if stage != b"0":
            raise RuntimeError(f"Unmerged staged entry: {raw_path.decode('utf-8', errors='replace')}")
        entries[raw_path.decode("utf-8")] = mode.decode("ascii")
    return entries


def _staged_bytes(relative: str) -> bytes:
    result = _run_git(["show", f":{relative}"])
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot read staged blob {relative}: {result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def _worktree_bytes(relative: str) -> bytes:
    return (PROJECT_ROOT / relative).read_bytes()


def _reader_for_audit(*, require_staged: bool, in_git: bool) -> Callable[[str], bytes]:
    if require_staged:
        if not in_git:
            raise RuntimeError("--require-staged requires an initialized Git repository")
        return _staged_bytes
    return _worktree_bytes


def _filesystem_candidates() -> set[str]:
    excluded_parts = {".git", ".pytest_cache", "__pycache__"}
    return {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and not (excluded_parts & set(path.relative_to(PROJECT_ROOT).parts))
    }


def _csv_rows(
    payload: bytes,
    *,
    relative: str,
    expected_count: int,
    expected_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise RuntimeError(f"Unexpected staged/public CSV schema for {relative}: {reader.fieldnames}")
    rows = list(reader)
    if len(rows) != expected_count:
        raise RuntimeError(f"Unexpected staged/public row count for {relative}: {len(rows)}")
    if any(set(row) != set(expected_fields) or None in row.values() for row in rows):
        raise RuntimeError(f"Malformed staged/public CSV row in {relative}")
    return rows


def _expected_matrix(relative: str) -> tuple[tuple[str, ...], set[tuple[str, ...]]]:
    datasets = ("celeba", "cifar10")
    splits = ("early", "current")
    seeds = ("7", "42", "123", "2024", "2025")
    methods = ("standard", "afd", "laplace", "learned")
    if relative == "results/raw/primary_seed_results.csv":
        return ("dataset", "split_point", "method", "seed"), set(
            product(datasets, splits, methods, seeds)
        )
    if relative == "results/raw/semantic_seed_results.csv":
        return ("dataset", "split_point", "method", "seed"), set(
            product(("celeba",), splits, methods, seeds)
        )
    if relative == "results/raw/semantic_attribute_results.csv":
        indices = tuple(str(index) for index in range(40) if index != 31)
        return ("dataset", "split_point", "method", "seed", "attribute_index"), set(
            product(("celeba",), splits, methods, seeds, indices)
        )
    budget_methods = ("standard", "afd")
    fractions = ("0.1", "0.5", "1.0")
    if relative == "results/raw/budget_cell_results.csv":
        return ("dataset", "split_point", "method", "seed", "auxiliary_fraction"), set(
            product(datasets, splits, budget_methods, ("42",), fractions)
        )
    if relative == "results/raw/budget_attacker_results.csv":
        return (
            "dataset", "split_point", "method", "seed", "auxiliary_fraction", "architecture",
        ), set(
            product(
                datasets, splits, budget_methods, ("42",), fractions,
                ("deconv_mse", "residual_lpips"),
            )
        )
    raise RuntimeError(f"No independent matrix policy for {relative}")


def _validate_rows(relative: str, rows: list[dict[str, str]]) -> None:
    key_fields, expected = _expected_matrix(relative)
    actual = [tuple(row[field] for field in key_fields) for row in rows]
    if len(actual) != len(set(actual)):
        raise RuntimeError(f"Duplicate statistical-unit key in {relative}")
    if set(actual) != expected:
        raise RuntimeError(f"Incomplete statistical-unit matrix in {relative}")

    numeric_fields = {
        "results/raw/primary_seed_results.csv": (
            "seed", "accuracy_percentage_points", "utility_loss_vs_standard_percentage_points",
            "worst_case_ssim",
        ),
        "results/raw/semantic_seed_results.csv": (
            "seed", "macro_auroc", "macro_balanced_accuracy", "attribute_auroc_sample_sd",
            "attribute_balanced_accuracy_sample_sd", "semantic_draw_count",
        ),
        "results/raw/semantic_attribute_results.csv": (
            "seed", "attribute_index", "auroc", "balanced_accuracy",
        ),
        "results/raw/budget_cell_results.csv": (
            "seed", "auxiliary_fraction", "auxiliary_examples", "auxiliary_fitting_examples",
            "auxiliary_validation_examples", "deconv_mse_ssim",
            "residual_lpips_ssim", "worst_case_test_ssim",
        ),
        "results/raw/budget_attacker_results.csv": (
            "seed", "auxiliary_fraction", "auxiliary_examples", "auxiliary_fitting_examples",
            "auxiliary_validation_examples", "selected_epoch",
            "epochs_completed", "best_validation_loss", "mse", "psnr", "ssim", "lpips",
        ),
    }[relative]
    for row in rows:
        for field in numeric_fields:
            try:
                value = float(row[field])
            except ValueError as error:
                raise RuntimeError(f"Non-numeric {relative}:{field}") from error
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {relative}:{field}")
        if "record_passed" in row and row["record_passed"] != "True":
            raise RuntimeError(f"Non-passing result row in {relative}")
        if "auxiliary_examples" in row and int(row["auxiliary_examples"]) != (
            int(row["auxiliary_fitting_examples"]) + int(row["auxiliary_validation_examples"])
        ):
            raise RuntimeError(f"Auxiliary split counts do not sum to the total cap in {relative}")

    for row in rows:
        for field in ("worst_case_ssim", "deconv_mse_ssim", "residual_lpips_ssim", "worst_case_test_ssim", "ssim"):
            if field in row and not -1.0 <= float(row[field]) <= 1.0:
                raise RuntimeError(f"Out-of-range {relative}:{field}")
        for field in ("macro_auroc", "macro_balanced_accuracy", "auroc", "balanced_accuracy", "auxiliary_fraction"):
            if field in row and not 0.0 <= float(row[field]) <= 1.0:
                raise RuntimeError(f"Out-of-range {relative}:{field}")
        if "accuracy_percentage_points" in row and not 0.0 <= float(row["accuracy_percentage_points"]) <= 100.0:
            raise RuntimeError(f"Out-of-range {relative}:accuracy_percentage_points")


def _parse_requirements(payload: bytes) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" not in stripped:
            raise RuntimeError(f"Unpinned requirement: {stripped}")
        name, version = stripped.split("==", 1)
        requirements[name] = version
    return requirements


def _verify_release(
    read_bytes: Callable[[str], bytes],
) -> tuple[int, dict[str, tuple[int, tuple[str, ...]]]]:
    checksum_payload = read_bytes("release/v1.2.0/SHA256SUMS").decode("utf-8")
    checksum_rows: dict[str, str] = {}
    for line in checksum_payload.splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError(f"Malformed v1.2.0 checksum row: {line!r}") from error
        if relative in checksum_rows or not SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"Invalid or duplicate v1.2.0 checksum row: {line!r}")
        checksum_rows[relative] = digest
    if checksum_rows != EXPECTED_RELEASE_CHECKSUMS:
        raise RuntimeError("v1.2.0 checksums differ from the independent fixed policy")
    for relative, digest in EXPECTED_RELEASE_CHECKSUMS.items():
        if _sha256(read_bytes(f"release/v1.2.0/{relative}")) != digest:
            raise RuntimeError(f"v1.2.0 payload checksum differs: {relative}")

    result_specs: dict[str, tuple[int, tuple[str, ...]]] = {}
    result_rows = 0
    for relative, expected_count in RELEASE_RESULT_COUNTS.items():
        payload = read_bytes(relative).decode("utf-8")
        reader = csv.DictReader(io.StringIO(payload, newline=""))
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
        if not fields or len(rows) != expected_count:
            raise RuntimeError(f"v1.2.0 CSV schema/count differs: {relative}")
        if any(set(row) != set(fields) or None in row.values() for row in rows):
            raise RuntimeError(f"Malformed v1.2.0 CSV row: {relative}")
        result_specs[relative] = (expected_count, fields)
        result_rows += len(rows)

    schema_pairs = {
        "release/v1.2.0/results/attacker_restart_rows.csv": "release/v1.2.0/schemas/attacker_restart_rows.schema.json",
        "release/v1.2.0/results/attacker_restart_summaries.csv": "release/v1.2.0/schemas/attacker_restart_summaries.schema.json",
        "release/v1.2.0/results/benchmark_budget_attacker_rows.csv": "release/v1.2.0/schemas/benchmark_budget_attacker_rows.schema.json",
        "release/v1.2.0/results/benchmark_budget_conditions.csv": "release/v1.2.0/schemas/benchmark_budget_conditions.schema.json",
        "release/v1.2.0/results/benchmark_budget_paired_effects.csv": "release/v1.2.0/schemas/benchmark_budget_paired_effects.schema.json",
    }
    for csv_relative, schema_relative in schema_pairs.items():
        schema = json.loads(read_bytes(schema_relative))
        schema_fields = tuple(column["name"] for column in schema.get("columns", []))
        expected_count, expected_fields = result_specs[csv_relative]
        if (
            schema.get("schema_version") != 2
            or schema.get("file") != csv_relative.removeprefix("release/v1.2.0/")
            or schema.get("rows") != expected_count
            or schema_fields != expected_fields
        ):
            raise RuntimeError(f"v1.2.0 schema metadata differs: {schema_relative}")
    benchmark_schema = json.loads(
        read_bytes("release/v1.2.0/schemas/benchmark_v1_tables.schema.json")
    )
    if benchmark_schema.get("schema_version") != 2 or set(
        benchmark_schema.get("tables", {})
    ) != {Path(relative).name for relative in schema_pairs}:
        raise RuntimeError("v1.2.0 benchmark schema differs from the fixed table set")
    compute_schema = json.loads(
        read_bytes("release/v1.2.0/schemas/eng_compute_v1.schema.json")
    )
    if (
        compute_schema.get("schema_version") != 1
        or compute_schema.get("protocol_id") != "eng_compute_v1"
        or compute_schema.get("inferential_unit") != "model_seed"
        or compute_schema.get("technical_blocks_are_inferential_units") is not False
        or compute_schema.get("p_values") is not False
        or compute_schema.get("multiplicity_tests") is not False
        or compute_schema.get("equivalence_claims") is not False
    ):
        raise RuntimeError("v1.2.0 compute schema boundary differs")
    analysis = json.loads(read_bytes("release/v1.2.0/analysis/compute_analysis.json"))
    claims = json.loads(read_bytes("release/v1.2.0/analysis/compute_claim_audit.json"))
    if (
        analysis.get("protocol_id") != "eng_compute_v1"
        or analysis.get("condition_count") != 160
        or analysis.get("quarantine_count") != 0
        or analysis.get("inferential_unit") != "model_seed"
        or analysis.get("technical_blocks_are_inferential_units") is not False
    ):
        raise RuntimeError("v1.2.0 compute analysis boundary differs")
    if [(row.get("claim_id"), row.get("status")) for row in claims.get("claims", [])] != [
        ("E01", "supported"),
        ("E02", "supported"),
        ("E03", "supported"),
    ]:
        raise RuntimeError("v1.2.0 compute claim audit differs")
    notes = read_bytes("release/v1.2.0/RELEASE_NOTES.md").decode("utf-8")
    for fragment in (
        "local staging artifact",
        "not a published GitHub release or live URL",
        "V1.2.0_RELEASE_URL_REQUIRED",
        "No tag, release, upload, DOI, or external service action has occurred.",
    ):
        if fragment not in notes:
            raise RuntimeError(f"v1.2.0 release notes missing boundary: {fragment}")
    return result_rows, result_specs


def _verify_disclosure(read_bytes: Callable[[str], bytes]) -> dict[str, Any]:
    manifest = json.loads(read_bytes("results/protocol/disclosure_manifest.json"))
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise RuntimeError(f"Unexpected disclosure manifest keys: {sorted(set(manifest) ^ EXPECTED_MANIFEST_KEYS)}")
    if manifest["schema_version"] != 3:
        raise RuntimeError("Unsupported disclosure manifest schema")
    if manifest["repository_name"] != "adaptive-reconstruction-split-computing":
        raise RuntimeError("Unexpected repository name")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", manifest["disclosure_date"]):
        raise RuntimeError("Disclosure date is not ISO YYYY-MM-DD")
    if manifest["public_allowlist"] != sorted(EXPECTED_ALLOWLIST):
        raise RuntimeError("Manifest allowlist differs from the independent fixed policy")
    if manifest["result_row_count"] != 2308:
        raise RuntimeError("Manifest must declare exactly 2,308 result rows")
    if manifest["base_result_row_count"] != 1752:
        raise RuntimeError("Manifest base result count must be exactly 1,752")
    if manifest["supplemental_result_row_count"] != 556:
        raise RuntimeError("Manifest supplemental result count must be exactly 556")
    expected_counts = {
        **{path: spec[0] for path, spec in RESULT_SPECS.items()},
        **RELEASE_RESULT_COUNTS,
    }
    expected_counts = dict(sorted(expected_counts.items()))
    if manifest["result_table_counts"] != expected_counts:
        raise RuntimeError("Manifest result-table counts differ from independent policy")
    if manifest["semantic_attribute_boundary"] != {
        "published_indices": [index for index in range(40) if index != 31],
        "excluded_index": 31,
        "reason": "The supervised task-target column is excluded; indices are schema references, not anonymized labels.",
    }:
        raise RuntimeError("Semantic attribute boundary is missing or inconsistent")
    if manifest["licenses"] != {
        "authored_code_schemas_and_documentation": "MIT",
        "contributor_owned_numeric_results_and_compute_analysis": "CC-BY-4.0",
        "third_party_data_and_assets": "not licensed or redistributed",
    }:
        raise RuntimeError("Manifest license boundary differs from policy")
    if manifest["release_decision"] != EXPECTED_RELEASE_DECISION:
        raise RuntimeError("Manifest release decision differs from independent policy")
    if manifest["release"] != EXPECTED_RELEASE_STATE:
        raise RuntimeError("Manifest release state differs from the v1.3.0 repository-commit policy")
    try:
        citation = read_bytes("CITATION.cff").decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("Root citation metadata is not UTF-8") from error
    if citation != EXPECTED_CITATION_CFF:
        raise RuntimeError("Root citation metadata differs from the v1.3.0 repository-commit policy")
    if manifest["disclosure_date"] != "2026-09-30":
        raise RuntimeError("Manifest disclosure date differs from the local staging date")
    workflow = read_bytes(".github/workflows/public-checks.yml").decode("utf-8")
    expected_actions = [
        "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
    ]
    observed_actions = re.findall(r"(?m)^\s*uses:\s*(\S+)", workflow)
    required_workflow_fragments = (
        "\n  pull_request:\n",
        "\n  push:\n    branches:\n      - main\n",
        "permissions:\n  contents: read\n",
        "persist-credentials: false",
        'python-version: "3.11"',
        "timeout-minutes: 45",
        "cancel-in-progress: true",
        "https://download.pytorch.org/whl/cpu",
        "-r requirements.txt",
    )
    if (
        observed_actions != expected_actions
        or any(fragment not in workflow for fragment in required_workflow_fragments)
        or "pull_request_target" in workflow
        or "secrets." in workflow
        or re.search(r"(?m)^\s*\w[\w-]*:\s*write\s*$", workflow)
        or workflow.count("./run_public_checks.sh") != 1
    ):
        raise RuntimeError("Public-checks workflow differs from the independent least-privilege policy")
    result_license = read_bytes("LICENSE-RESULTS").decode("utf-8")
    third_party_notice = read_bytes("THIRD_PARTY_DATA.md").decode("utf-8")
    readme = read_bytes("README.md").decode("utf-8")
    citation_readme_fragments = (
        "# Adaptive Reconstruction Split Computing",
        "exactly 123 files",
        "2,308 rows",
        "58 verified public outputs",
        "authorized_for_repository_commit",
        "CITATION.cff",
        "intentionally omits a repository URL, release date, and DOI",
        "(cd release/v1.2.0 && sha256sum --check SHA256SUMS)",
    )
    if any(fragment not in readme for fragment in citation_readme_fragments):
        raise RuntimeError("README does not match the v1.3.0 citation policy")
    if (
        "Adaptive Reconstruction Split Computing aggregate results and analyses "
        "(version 1.2.0)" not in result_license
    ):
        raise RuntimeError("Result-license attribution differs from the public identity")
    for relative in EXPECTED_RELEASE_DECISION["scope"]:
        if result_license.count(relative) != 1:
            raise RuntimeError(f"Result license must name the reviewed file exactly once: {relative}")
    required_notice_fragments = {
        "LICENSE-RESULTS": (
            "The listed results are distributed as repository files",
            "No third-party permission or waiver is claimed",
        ),
        "THIRD_PARTY_DATA.md": (
            "The exact CC BY 4.0 scope is listed in LICENSE-RESULTS",
            "No third-party permission or waiver is claimed",
        ),
        "README.md": (
            "The licensed results are distributed as repository files",
            "does not claim third-party permission",
        ),
    }
    for label, fragments in required_notice_fragments.items():
        text = {
            "LICENSE-RESULTS": result_license,
            "THIRD_PARTY_DATA.md": third_party_notice,
            "README.md": readme,
        }[label]
        normalized_text = " ".join(text.split())
        if any(fragment not in normalized_text for fragment in fragments):
            raise RuntimeError(f"{label} does not match the reviewed release decision")
    stale_release_language = (
        "approved public release",
        "first audited GitHub-only release",
        "exact 72-file public boundary",
    )
    notices = (result_license, third_party_notice, readme)
    if any(fragment in text for text in notices for fragment in stale_release_language):
        raise RuntimeError("Public notices still contain the superseded pending-release gate")
    if set(manifest["source_inputs"]) != SOURCE_INPUT_ROLES:
        raise RuntimeError("Manifest source-input roles differ from policy")
    for role, record in manifest["source_inputs"].items():
        if set(record) != {"sha256", "size_bytes"}:
            raise RuntimeError(f"Unexpected source-input record for {role}")
        if not SHA256_RE.fullmatch(str(record["sha256"])) or int(record["size_bytes"]) <= 0:
            raise RuntimeError(f"Invalid source-input commitment for {role}")

    if set(manifest["public_outputs"]) != PUBLIC_OUTPUTS:
        raise RuntimeError("Manifest public-output set differs from independent policy")
    verified_rows = 0
    for relative, (count, fields) in RESULT_SPECS.items():
        rows = _csv_rows(
            read_bytes(relative), relative=relative, expected_count=count, expected_fields=fields
        )
        _validate_rows(relative, rows)
        verified_rows += len(rows)
    for relative, digest in V13_CHECKSUMS.items():
        if _sha256(read_bytes(relative)) != digest:
            raise RuntimeError(f"v1.3.0 payload checksum differs: {relative}")
    release_rows, release_result_specs = _verify_release(read_bytes)
    verified_rows += release_rows
    for relative in sorted(PUBLIC_OUTPUTS):
        payload = read_bytes(relative)
        record = manifest["public_outputs"][relative]
        expected_record_keys = {"sha256", "size_bytes"}
        if relative in RESULT_SPECS:
            expected_record_keys |= {"row_count", "fields"}
            if record["row_count"] != RESULT_SPECS[relative][0] or record["fields"] != list(RESULT_SPECS[relative][1]):
                raise RuntimeError(f"Manifest schema metadata mismatch: {relative}")
        elif relative in release_result_specs:
            expected_record_keys |= {"row_count", "fields"}
            expected_count, expected_fields = release_result_specs[relative]
            if record["row_count"] != expected_count or record["fields"] != list(expected_fields):
                raise RuntimeError(f"Manifest release schema metadata mismatch: {relative}")
        if set(record) != expected_record_keys:
            raise RuntimeError(f"Unexpected public-output record keys: {relative}")
        if record["sha256"] != _sha256(payload) or record["size_bytes"] != len(payload):
            raise RuntimeError(f"Public-output hash/size mismatch: {relative}")

    contract = read_bytes("config/study_contract.json")
    implementation_protocol = json.loads(read_bytes("config/implementation_protocol_v2.json"))
    if (
        implementation_protocol.get("schema_version") != 2
        or implementation_protocol.get("study_contract", {}).get("sha256") != _sha256(contract)
        or implementation_protocol.get("budget", {}).get("ordinary_validation_access") is not False
    ):
        raise RuntimeError("Implementation supplement differs from the public runtime contract")
    lock = json.loads(read_bytes("results/protocol/frozen_selection_lock.json"))
    if lock["study_contract"] != {
        "path": "config/study_contract.json",
        "sha256": _sha256(contract),
        "size_bytes": len(contract),
    }:
        raise RuntimeError("Public contract does not match the frozen-selection lock")
    frozen = lock["frozen_artifact"]
    if frozen.get("published") is not False or not SHA256_RE.fullmatch(frozen.get("sha256", "")):
        raise RuntimeError("Frozen-selection exclusion record is invalid")

    environment = json.loads(read_bytes("results/protocol/environment.json"))
    expected_environment_keys = {
        "captured_on", "python", "packages", "torch_cuda_build",
        "cuda_available_at_capture", "gpu_inventory", "hardware_measurement_role",
    }
    if set(environment) != expected_environment_keys or environment["python"] != "3.11.5":
        raise RuntimeError("Sanitized environment record differs from policy")
    requirements = {
        **_parse_requirements(read_bytes("requirements.txt")),
        **_parse_requirements(read_bytes("requirements-figures.txt")),
    }
    if environment["packages"] != requirements:
        raise RuntimeError("Requirement pins differ from the sanitized environment record")

    return {
        "passed": True,
        "result_rows_verified": verified_rows,
        "public_outputs_verified": len(PUBLIC_OUTPUTS),
        "allowlist_files": len(EXPECTED_ALLOWLIST),
        "release_files_verified": len(RELEASE_FILES),
        "release_checksum_entries": len(EXPECTED_RELEASE_CHECKSUMS),
        "manifest_sha256": _sha256(read_bytes("results/protocol/disclosure_manifest.json")),
    }


def audit(*, require_staged: bool) -> dict[str, Any]:
    errors: list[str] = []
    in_git = _inside_git()
    try:
        read_bytes = _reader_for_audit(require_staged=require_staged, in_git=in_git)
    except RuntimeError as error:
        read_bytes = _worktree_bytes
        errors.append(str(error))

    staged_index = _staged_index() if in_git else {}
    candidates = _git_candidates() if in_git else _filesystem_candidates()
    if candidates != EXPECTED_ALLOWLIST:
        errors.append(
            "Git/filesystem candidate set differs from allowlist: "
            f"unexpected={sorted(candidates - EXPECTED_ALLOWLIST)}, "
            f"missing={sorted(EXPECTED_ALLOWLIST - candidates)}"
        )
    if require_staged and set(staged_index) != EXPECTED_ALLOWLIST:
        errors.append(
            "staged set differs from allowlist: "
            f"unexpected={sorted(set(staged_index) - EXPECTED_ALLOWLIST)}, "
            f"missing={sorted(EXPECTED_ALLOWLIST - set(staged_index))}"
        )

    try:
        disclosure = _verify_disclosure(read_bytes)
    except (KeyError, OSError, RuntimeError, TypeError, UnicodeDecodeError, ValueError) as error:
        disclosure = {"passed": False, "error": str(error)}
        errors.append(f"independent public-result verification failed: {error}")

    sizes: list[tuple[int, str]] = []
    trusted_artifact_loaders: list[str] = []
    for relative in sorted(EXPECTED_ALLOWLIST):
        try:
            payload = read_bytes(relative)
        except (OSError, RuntimeError) as error:
            errors.append(f"cannot read audited public bytes for {relative}: {error}")
            continue
        sizes.append((len(payload), relative))
        if Path(relative).suffix.lower() in BANNED_SUFFIXES or Path(relative).name in BANNED_BASENAMES:
            errors.append(f"banned file type/name in public set: {relative}")
        if len(payload) >= MAX_PUBLIC_FILE_BYTES:
            errors.append(f"public file is at least 1 MiB: {relative}")
        if require_staged and staged_index.get(relative) not in {"100644", "100755"}:
            errors.append(f"staged public path is not a regular file: {relative}")
        if not require_staged:
            path = PROJECT_ROOT / relative
            if not path.is_file() or path.is_symlink():
                errors.append(f"allowlisted path is missing, non-file, or symlinked: {relative}")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"public file is not UTF-8 text: {relative}")
            continue
        for name, pattern in CONTENT_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{name} detected in public file: {relative}")
        unsafe_load_opt_out = "weights_only" + "=False"
        if Path(relative).suffix == ".py" and unsafe_load_opt_out in text:
            trusted_artifact_loaders.append(relative)
        if require_staged:
            path = PROJECT_ROOT / relative
            if not path.is_file() or path.is_symlink():
                errors.append(f"worktree public path is missing, non-file, or symlinked: {relative}")
            elif path.read_bytes() != payload:
                errors.append(f"worktree bytes differ from staged blob: {relative}")

    ignored_verified = 0
    if in_git:
        for relative in IGNORED_BOUNDARY:
            result = _run_git(["check-ignore", "--no-index", "-q", "--", relative])
            if result.returncode != 0:
                errors.append(f"private boundary path is not ignored: {relative}")
            else:
                ignored_verified += 1

    largest = max(sizes, default=(0, ""))
    expected_trusted_loaders = {"src/checkpointing.py", "src/defenses.py"}
    if set(trusted_artifact_loaders) != expected_trusted_loaders:
        errors.append(
            "trusted-artifact loader set differs from the reviewed compatibility boundary: "
            f"found={sorted(trusted_artifact_loaders)}"
        )
    try:
        readme = read_bytes("README.md").decode("utf-8")
    except (OSError, RuntimeError, UnicodeDecodeError):
        readme = ""
    if "Never load an untrusted `.pt` file" not in readme:
        errors.append("README is missing the mandatory trusted-checkpoint warning")
    report = {
        "passed": not errors,
        "repository_name": "adaptive-reconstruction-split-computing",
        "allowlist_files": len(EXPECTED_ALLOWLIST),
        "candidate_files": len(candidates),
        "staged_files": len(staged_index),
        "audited_byte_source": "git_index" if require_staged and in_git else "worktree",
        "ignored_boundary_paths_verified": ignored_verified,
        "largest_public_file": {"path": largest[1], "size_bytes": largest[0]},
        "trusted_artifact_loaders": sorted(trusted_artifact_loaders),
        "public_results": disclosure,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-staged", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(require_staged=args.require_staged), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
