#!/usr/bin/env bash
set -euo pipefail

# Validation-only operating-point selection. This is intentionally long-running.
python scripts/run_revision_pipeline.py init "$@"
python scripts/run_revision_pipeline.py selection "$@"
