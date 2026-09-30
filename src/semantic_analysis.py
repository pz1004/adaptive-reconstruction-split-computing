"""Five-seed semantic estimates and the separate 12-contrast comparison family."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .analysis import estimate, holm_adjust, paired_analysis


SEMANTIC_METHODS = ("standard", "afd", "laplace", "learned")
SEMANTIC_COMPARISON_METHODS = ("afd", "laplace", "learned")
SEMANTIC_METRICS = ("macro_auroc", "macro_balanced_accuracy")


def build_semantic_analysis(
    seed_rows: list[Mapping[str, Any]], required_seeds: list[int]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in seed_rows:
        key = (str(row["split_point"]), str(row["method"]))
        seed = int(row["seed"])
        if seed in grouped[key]:
            raise ValueError(f"Duplicate semantic seed row for {key}: {seed}")
        grouped[key][seed] = row

    group_summaries: dict[str, Any] = {}
    for split_point in ("early", "current"):
        for method in SEMANTIC_METHODS:
            by_seed = grouped.get((split_point, method), {})
            if sorted(by_seed) != sorted(required_seeds):
                raise RuntimeError(f"Incomplete semantic five-seed group: celeba/{split_point}/{method}")
            group_summaries[f"celeba/{split_point}/{method}"] = {
                "required_seeds": required_seeds,
                "available_seeds": sorted(by_seed),
                "complete": True,
                "metrics": {
                    metric: estimate(float(by_seed[seed][metric]) for seed in required_seeds)
                    for metric in SEMANTIC_METRICS
                },
            }

    contrasts: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for split_point in ("early", "current"):
        standard = grouped[(split_point, "standard")]
        for method in SEMANTIC_COMPARISON_METHODS:
            comparison = grouped[(split_point, method)]
            for metric in SEMANTIC_METRICS:
                contrast_id = f"celeba/{split_point}/{method}_vs_standard/{metric}"
                result = paired_analysis(
                    [float(standard[seed][metric]) for seed in required_seeds],
                    [float(comparison[seed][metric]) for seed in required_seeds],
                )
                result.update(
                    {
                        "status": "complete",
                        "seeds": required_seeds,
                        "metric": metric,
                        "metric_direction": "higher_is_more_semantic_leakage",
                        "comparison_method": method,
                        "reference_method": "standard",
                    }
                )
                contrasts[contrast_id] = result
                raw_pvalues[contrast_id] = float(result["exact_sign_flip_pvalue"])
    if len(group_summaries) != 8 or len(contrasts) != 12:
        raise RuntimeError(
            f"Semantic analysis cardinality mismatch: {len(group_summaries)} groups and {len(contrasts)} contrasts"
        )
    adjusted = holm_adjust(raw_pvalues)
    for contrast_id, value in adjusted.items():
        contrasts[contrast_id]["holm_adjusted_sign_flip_pvalue"] = value
        contrasts[contrast_id]["holm_family"] = (
            "12 declared semantic method-by-interface-by-outcome contrasts; separate from the unchanged 24-contrast primary family"
        )
    return {
        "schema_version": 1,
        "complete": True,
        "unit_of_analysis": "seed",
        "required_seeds": required_seeds,
        "images_are_replicates": False,
        "attributes_are_replicates": False,
        "draws_are_replicates": False,
        "group_summary_count": len(group_summaries),
        "contrast_count": len(contrasts),
        "group_summaries": group_summaries,
        "contrasts": contrasts,
        "holm_family": "12 semantic contrasts only",
        "shapiro_wilk_role": "low_power_diagnostic_only_at_n_5_no_test_switching",
        "outcome_neutrality": "All contrasts are retained regardless of direction, magnitude, or p-value.",
    }
