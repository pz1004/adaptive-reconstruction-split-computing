"""Portable watcher-state classification for the fixed 24-cell budget stage."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def classify_watcher_state(
    records: Sequence[Mapping[str, Any]], service: Mapping[str, str]
) -> str:
    counts = Counter(str(record.get("status", "missing")) for record in records)
    if counts.get("failed", 0):
        return "failed_cell"
    if counts.get("complete", 0) == 24:
        return "ready_to_finalize"
    if service.get("ActiveState") in {"inactive", "failed"}:
        return "premature_service_exit"
    return "monitoring"
