"""Executable implementation supplement and transitive local-source binding."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPLEMENTATION_PROTOCOL = PROJECT_ROOT / "config" / "implementation_protocol_v2.json"
IMPLEMENTATION_ENTRY_POINTS: dict[str, str] = {
    "experiment_execution": "scripts/run_revision_pipeline.py",
    "selection": "src/selection_provenance.py",
    "metrics": "src/metrics.py",
    "semantic_analysis": "scripts/analyze_semantic_results.py",
    "statistical_analysis": "scripts/analyze_revision.py",
    "budget_audit": "scripts/audit_budget_results.py",
    "budget_analysis": "scripts/analyze_budget_results.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _module_name(path: Path, project_root: Path) -> str:
    relative = path.resolve().relative_to(project_root.resolve()).with_suffix("")
    return ".".join(relative.parts)


def _module_path(module: str, project_root: Path) -> Path | None:
    if not module or module.split(".", 1)[0] not in {"src", "scripts"}:
        return None
    candidate = project_root.joinpath(*module.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate.resolve()
    package = project_root.joinpath(*module.split("."), "__init__.py")
    return package.resolve() if package.is_file() else None


def _imported_local_paths(path: Path, project_root: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"Cannot parse implementation source {path}: {error}") from error
    current_module = _module_name(path, project_root)
    current_package = current_module.rpartition(".")[0]
    dependencies: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = current_package.split(".") if current_package else []
                trim = node.level - 1
                if trim > len(package_parts):
                    raise RuntimeError(f"Invalid relative import in {path}:{node.lineno}")
                base_parts = package_parts[: len(package_parts) - trim]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            modules.append(base)
            modules.extend(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
        for module in modules:
            dependency = _module_path(module, project_root)
            if dependency is not None and dependency != path.resolve():
                dependencies.add(dependency)
    return dependencies


def local_import_closure(
    entry_points: Mapping[str, str] | Iterable[str] = IMPLEMENTATION_ENTRY_POINTS,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    declared = dict(entry_points) if isinstance(entry_points, Mapping) else {
        path: path for path in entry_points
    }
    roots: dict[str, Path] = {}
    for name, relative in declared.items():
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as error:
            raise RuntimeError(f"Implementation entry point escapes project root: {relative}") from error
        if not path.is_file():
            raise RuntimeError(f"Missing implementation entry point: {relative}")
        roots[name] = path

    queue: deque[Path] = deque(sorted(set(roots.values())))
    visited: set[Path] = set()
    graph: dict[str, list[str]] = {}
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        dependencies = _imported_local_paths(path, project_root)
        relative = path.relative_to(project_root.resolve()).as_posix()
        graph[relative] = sorted(
            dependency.relative_to(project_root.resolve()).as_posix()
            for dependency in dependencies
        )
        queue.extend(sorted(dependencies - visited))

    files = {
        path.relative_to(project_root.resolve()).as_posix(): sha256_file(path)
        for path in sorted(visited)
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "entry_points": {name: path.relative_to(project_root.resolve()).as_posix() for name, path in roots.items()},
        "files": files,
        "graph": dict(sorted(graph.items())),
        "closure_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Implementation supplement mismatch for {label}: {actual!r} != {expected!r}")


def validate_implementation_protocol(
    contract: Mapping[str, Any],
    contract_path: str | Path,
    supplement_path: str | Path = DEFAULT_IMPLEMENTATION_PROTOCOL,
) -> dict[str, Any]:
    """Fail closed when declared implementation constants differ from runtime code."""
    supplement_file = Path(supplement_path).resolve()
    payload = strict_json(supplement_file)
    contract_file = Path(contract_path).resolve()
    _require_equal(payload.get("schema_version"), 2, "schema_version")
    declared_contract = payload.get("study_contract", {})
    declared_hash = declared_contract.get("sha256")
    actual_hash = sha256_file(contract_file)
    if declared_hash != actual_hash:
        declared_path = (supplement_file.parents[1] / str(declared_contract.get("path", ""))).resolve()
        if contract != strict_json(declared_path):
            _require_equal(declared_hash, actual_hash, "study_contract.sha256")
    _require_equal(list(contract.get("auxiliary_data_budgets", [])), [0.1, 0.5, 1.0], "auxiliary_data_budgets")

    from . import experiment, metrics, pipeline

    budget = payload.get("budget", {})
    _require_equal(budget.get("protocol_version"), pipeline.BUDGET_PROTOCOL_VERSION, "budget.protocol_version")
    _require_equal(budget.get("stage"), pipeline.BUDGET_STAGE, "budget.stage")
    _require_equal(budget.get("artifact_schema_version"), pipeline.BUDGET_ARTIFACT_SCHEMA_VERSION, "budget.artifact_schema_version")
    _require_equal(budget.get("split_seed"), pipeline.BUDGET_SUBSET_SEED, "budget.split_seed")
    _require_equal(budget.get("split_policy"), pipeline.BUDGET_SPLIT_POLICY, "budget.split_policy")
    _require_equal(budget.get("ordinary_validation_access"), False, "budget.ordinary_validation_access")

    expected_metrics = {
        "mse": "per-image mean squared error over CxHxW",
        "psnr": "10*log10(1/per-image-mse), data_range=1",
        "ssim": "scikit-image structural_similarity, data_range=1, channel_axis=2",
        "lpips": "lpips-0.1.4-alex on images mapped from [0,1] to [-1,1]",
    }
    _require_equal(payload.get("metrics"), expected_metrics, "metrics")

    semantic = payload.get("semantic_probe", {})
    _require_equal(semantic.get("architecture"), experiment.SEMANTIC_PROBE_ARCHITECTURE, "semantic_probe.architecture")
    _require_equal(semantic.get("balanced_accuracy_threshold"), experiment.SEMANTIC_BALANCED_ACCURACY_THRESHOLD, "semantic_probe.balanced_accuracy_threshold")
    _require_equal(semantic.get("excluded_celeba_attribute_index"), experiment.SEMANTIC_EXCLUDED_ATTRIBUTE_INDEX, "semantic_probe.excluded_celeba_attribute_index")
    _require_equal(semantic.get("learning_rate"), experiment.SEMANTIC_PROBE_LEARNING_RATE, "semantic_probe.learning_rate")
    _require_equal(semantic.get("output_count"), experiment.SEMANTIC_PROBE_OUTPUT_COUNT, "semantic_probe.output_count")

    lpips = payload.get("lpips", {})
    _require_equal(lpips.get("backbone"), metrics.LPIPS_BACKBONE, "lpips.backbone")
    _require_equal(lpips.get("package_weight", {}).get("sha256"), metrics.LPIPS_PACKAGE_WEIGHT_SHA256, "lpips.package_weight.sha256")
    _require_equal(lpips.get("torchvision_backbone_weight", {}).get("sha256"), metrics.LPIPS_BACKBONE_WEIGHT_SHA256, "lpips.torchvision_backbone_weight.sha256")

    dependencies = payload.get("dependencies", {})
    for package in ("Pillow", "scikit-learn"):
        _require_equal(
            dependencies.get(package, {}).get("version"),
            metadata.version(package),
            f"dependencies.{package}.version",
        )
    return payload
