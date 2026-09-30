#!/usr/bin/env python3
"""Compatibility wrapper for the machine-readable strict analysis pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("analyze_revision.py")), run_name="__main__")
