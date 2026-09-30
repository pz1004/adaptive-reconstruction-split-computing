#!/usr/bin/env bash
set -euo pipefail

# The five-seed primary matrix requires a frozen validation-only selection file.
python scripts/run_revision_pipeline.py primary "$@"
python scripts/run_revision_pipeline.py consolidate "$@"
