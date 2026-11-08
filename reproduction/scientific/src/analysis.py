"""Estimation-first paired analysis for the adaptive reconstruction study."""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Iterable

import numpy as np
from scipy import stats


def estimate(values: Iterable[float], confidence: float = 0.95) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Estimates require a nonempty finite one-dimensional sample")
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1)) if array.size > 1 else None
    if array.size > 1:
        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=array.size - 1))
        half_width = critical * float(array.std(ddof=1)) / math.sqrt(array.size)
        interval: list[float] | None = [mean - half_width, mean + half_width]
    else:
        interval = None
    return {
        "n_seeds": int(array.size),
        "values": [float(value) for value in array],
        "mean": mean,
        "sample_sd": sample_sd,
        "confidence_level": confidence,
        "mean_confidence_interval": interval,
        "unit_of_analysis": "seed",
    }


def exact_paired_randomization_pvalue(differences: Iterable[float]) -> float:
    array = np.asarray(list(differences), dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Randomization test requires finite paired differences")
    observed = abs(float(array.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=array.size):
        statistics.append(abs(float((array * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def paired_effect(differences: Iterable[float]) -> dict[str, float | None]:
    """Return the small-sample corrected paired standardized effect (Hedges g_z)."""
    array = np.asarray(list(differences), dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("Paired effect requires at least two finite differences")
    sample_sd = float(array.std(ddof=1))
    if sample_sd == 0.0:
        return {"cohens_dz": None, "hedges_gz": None, "correction": None}
    dz = float(array.mean()) / sample_sd
    degrees_of_freedom = array.size - 1
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
    return {"cohens_dz": dz, "hedges_gz": dz * correction, "correction": correction}


def shapiro_wilk_diagnostic(values: Iterable[float]) -> dict[str, Any]:
    """Return Shapiro-Wilk as a low-power diagnostic, never as a test-selection gate."""
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 3 or not np.isfinite(array).all():
        raise ValueError("Shapiro-Wilk requires at least three finite values")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        statistic, pvalue = stats.shapiro(array)
    return {
        "test": "Shapiro-Wilk",
        "n": int(array.size),
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "role": "low_power_diagnostic_only",
        "changes_declared_inference": False,
    }


def paired_analysis(reference: Iterable[float], comparison: Iterable[float]) -> dict[str, Any]:
    reference_array = np.asarray(list(reference), dtype=float)
    comparison_array = np.asarray(list(comparison), dtype=float)
    if reference_array.shape != comparison_array.shape:
        raise ValueError("Paired samples must have identical shapes")
    differences = comparison_array - reference_array
    sign_flip = exact_paired_randomization_pvalue(differences)
    return {
        "direction": "comparison_minus_standard",
        "standard": estimate(reference_array),
        "comparison": estimate(comparison_array),
        "paired_difference": estimate(differences),
        "corrected_paired_effect": paired_effect(differences),
        "paired_difference_shapiro_wilk": shapiro_wilk_diagnostic(differences),
        "exact_sign_flip_pvalue": sign_flip,
        "exact_paired_randomization_pvalue_supplementary": sign_flip,
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    if any(not 0.0 <= value <= 1.0 for value in pvalues.values()):
        raise ValueError("P-values must lie in [0, 1]")
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted
