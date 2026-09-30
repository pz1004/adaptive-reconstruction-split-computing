#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_CACHE_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${PUBLIC_CACHE_ROOT}"' EXIT
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -c 'import sys; expected = (3, 11); actual = sys.version_info[:2]; raise SystemExit(0 if actual == expected else f"Python 3.11 is required; found {actual[0]}.{actual[1]}")'

export XDG_CACHE_HOME="${PUBLIC_CACHE_ROOT}/xdg"
export TORCH_HOME="${PUBLIC_CACHE_ROOT}/torch"
export MPLCONFIGDIR="${PUBLIC_CACHE_ROOT}/matplotlib"
export PYTHONPYCACHEPREFIX="${PUBLIC_CACHE_ROOT}/pycache"

"${PYTHON_BIN}" - <<'PY'
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from src.metrics import (
    LPIPS_BACKBONE_WEIGHT_FILENAME,
    LPIPS_BACKBONE_WEIGHT_SHA256,
    lpips_backbone_weight_path,
)

filename = LPIPS_BACKBONE_WEIGHT_FILENAME
expected_sha256 = LPIPS_BACKBONE_WEIGHT_SHA256
url = f"https://download.pytorch.org/models/{filename}"
target = lpips_backbone_weight_path()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not target.is_file() or sha256_file(target) != expected_sha256:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            if not response.geturl().startswith("https://"):
                raise RuntimeError("Pretrained-weight download redirected away from HTTPS")
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise RuntimeError(
                f"AlexNet weight hash mismatch: {observed} != {expected_sha256}"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

observed = sha256_file(target)
if observed != expected_sha256:
    raise RuntimeError(f"AlexNet weight hash mismatch: {observed} != {expected_sha256}")
print(f"verified_pretrained_weight={target} sha256={observed}")
PY

cd "${REPOSITORY_ROOT}"
"${PYTHON_BIN}" scripts/export_eng_compute_public.py --check --public-only
"${PYTHON_BIN}" scripts/export_public_results.py --check --public-only
"${PYTHON_BIN}" scripts/audit_public_repository.py
"${PYTHON_BIN}" -m compileall -q src scripts tests
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
  tests/test_aggregate_reproduction.py \
  tests/test_budget_result_audit.py \
  tests/test_code_scripts.py \
  tests/test_dataset_remedy.py \
  tests/test_public_repository.py \
  tests/test_public_protocol.py \
  tests/test_result_audit.py \
  tests/test_revision_protocol.py \
  tests/test_selection_provenance.py \
  tests/test_semantic_result_audit.py
