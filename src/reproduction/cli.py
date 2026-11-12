"""Public aggregate replay with verification before atomic publication."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .integrity import SCIENTIFIC_HASHES

ROOT = Path(__file__).resolve().parents[2]
PROTECTED_TREES = (
    "src", "scripts", "config", "reproduction", "results", "tests", "docs",
    ".git", ".github", ".agents", ".aws", ".codex", "audit", "data",
    "research-records", "revisions", "submission", "figures", "deprecated",
    "analysis-output", "release",
)


def validate_destination(output: Path, data: Path) -> Path:
    """Reject existing paths, symlink traversal, and source/input overlap."""
    output = output.absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError(f"Output must not traverse a symlink: {path}")
    output = output.resolve()
    protected = [ROOT / name for name in PROTECTED_TREES] + [data.resolve()]
    for path in protected:
        if output == path or path in output.parents or output in path.parents:
            raise ValueError(f"Output overlaps protected source or input tree: {path}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    return output


def publish(staging: Path, output: Path) -> None:
    """Atomically rename a complete directory without replacing any destination."""
    if sys.platform == "win32":
        os.rename(staging, output)  # Windows refuses an existing destination.
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(library, "renameat2", None)
        arguments = (-100, os.fsencode(staging), -100, os.fsencode(output), 1)
        types = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    elif sys.platform == "darwin":
        rename = getattr(library, "renamex_np", None)
        arguments = (os.fsencode(staging), os.fsencode(output), 4)  # RENAME_EXCL
        types = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    else:
        rename = None
    if rename is None:
        raise OSError(errno.ENOTSUP, "Atomic no-replace directory publication is unavailable")
    rename.argtypes = types
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


def run(*, suite: str, output: Path, data: Path | None = None) -> dict:
    if suite not in {"paper", "supporting", "all"}:
        raise ValueError(f"Unknown suite: {suite}")
    data = data if data is not None else ROOT / "reproduction/aggregate-data"
    destination = validate_destination(output, data)
    if suite in {"paper", "all"}:
        # Compatibility --data must contain the published bytes.
        for name in SCIENTIFIC_HASHES:
            if name.startswith("reproduction/aggregate-data/"):
                path = data / Path(name).name
                if not path.is_file():
                    raise FileNotFoundError(f"Required aggregate input missing: {path.name}")
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != SCIENTIFIC_HASHES[name]:
                    raise ValueError(f"Scientific checksum mismatch: {path.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reproduce-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "result"
        reports = {}
        if suite == "all":
            staging.mkdir()
        if suite in {"paper", "all"}:
            from .paper import replay
            reports["paper"] = replay(data, staging / "paper" if suite == "all" else staging)
            report = reports["paper"]
            if (report["numerical_comparisons"] != 8227 or report["descriptive_rows"] != 1140
                    or report["absolute_tolerance"] != 1e-12):
                raise ValueError("Paper replay acceptance counts or tolerance changed")
        if suite in {"supporting", "all"}:
            from .supporting import replay  # Scientific dependencies load only here.
            reports["supporting"] = replay(staging / "supporting" if suite == "all" else staging)
            if len(reports["supporting"]["checks"]) != 8 or len(reports["supporting"]["figures"]) != 3:
                raise ValueError("Supporting replay acceptance counts changed")
        if suite == "all":
            report = {"passed": True, "suite": "all", "outcomes": reports}
            (staging / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            report = reports[suite]
        validate_destination(destination, data)
        publish(staging, destination)
    return report


def main(argv: list[str] | None = None, *, compatibility: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if compatibility:
        parser.add_argument("--data", type=Path, default=ROOT / "reproduction/aggregate-data")
    else:
        parser.add_argument("--suite", choices=("paper", "supporting", "all"), default="paper")
    parser.add_argument("--output", type=Path, default=Path("replay-output"))
    args = parser.parse_args(argv)
    try:
        report = run(suite="paper" if compatibility else args.suite,
                     output=args.output, data=args.data if compatibility else None)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"Reproduction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0
