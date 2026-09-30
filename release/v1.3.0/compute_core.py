from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterable, Iterator
from collections import defaultdict
from contextlib import contextmanager
import csv, io, json, math
import numpy as np
import matplotlib.pyplot as plt
class ComputeContractError(RuntimeError): pass
DATASETS=('celeba','cifar10')
INTERFACES=('early','current')
METHODS=('standard','afd','laplace','learned')
MODEL_SEEDS=(7,42,123,2024,2025)
BATCH_SIZES=(1,32)
def student_t_interval(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ComputeContractError("Student-t interval requires at least two finite one-dimensional values")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    from scipy.stats import t

    critical = float(t.ppf(0.975, df=array.size - 1))
    half = critical * std / math.sqrt(array.size)
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }
METRIC_FIELDS = (
    "cpu_encoder_plus_defense_latency_seconds",
    "cpu_encoder_plus_defense_throughput_per_second",
    "defense_only_latency_seconds",
    "defense_only_throughput_per_second",
    "gpu_classifier_latency_seconds",
    "gpu_classifier_throughput_per_second",
    "summed_component_time_seconds",
    "isolated_cpu_rss_delta_bytes",
    "cuda_peak_allocation_bytes",
)

def _group(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)

def _group_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    summaries = []
    lookup = {}
    keys = ("dataset", "interface", "method", "batch_size")
    for group_key, group_rows in sorted(_group(rows, keys).items()):
        if sorted(int(row["model_seed"]) for row in group_rows) != list(MODEL_SEEDS):
            raise ComputeContractError(f"Group does not contain the five model seeds: {group_key}")
        record: dict[str, Any] = dict(zip(keys, group_key))
        metric_summaries = {}
        for field in METRIC_FIELDS + (
            "static_total_bytes",
            "static_defense_bytes",
        ):
            values = [row[field] for row in group_rows if row[field] is not None]
            if values:
                metric_summaries[field] = student_t_interval([float(value) for value in values])
        record["metrics"] = metric_summaries
        lookup[group_key] = record
        flat = {key: record[key] for key in keys}
        for field, summary in metric_summaries.items():
            for stat_name, value in summary.items():
                flat[f"{field}__{stat_name}"] = value
        summaries.append(flat)
    if len(summaries) != 32:
        raise ComputeContractError(f"Expected 32 compute groups, found {len(summaries)}")
    return summaries, lookup

def _paired_effects(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    by_key = {
        (row["dataset"], row["interface"], row["method"], row["model_seed"], row["batch_size"]): row
        for row in rows
    }
    output = []
    all_cpu = []
    all_gpu = []
    for dataset in DATASETS:
        for interface in INTERFACES:
            for batch_size in BATCH_SIZES:
                cpu = []
                gpu = []
                summed = []
                for seed in MODEL_SEEDS:
                    afd = by_key[(dataset, interface, "afd", seed, batch_size)]
                    standard = by_key[(dataset, interface, "standard", seed, batch_size)]
                    cpu.append(afd["cpu_encoder_plus_defense_latency_seconds"] - standard["cpu_encoder_plus_defense_latency_seconds"])
                    gpu.append(afd["gpu_classifier_latency_seconds"] - standard["gpu_classifier_latency_seconds"])
                    summed.append(afd["summed_component_time_seconds"] - standard["summed_component_time_seconds"])
                all_cpu.extend(cpu)
                all_gpu.extend(gpu)
                record: dict[str, Any] = {"dataset": dataset, "interface": interface, "batch_size": batch_size}
                for name, values in (("cpu", cpu), ("gpu", gpu), ("summed", summed)):
                    summary = student_t_interval(values)
                    record.update({f"{name}_{key}": value for key, value in summary.items()})
                    record[f"{name}_seed_differences_seconds"] = json.dumps(values, separators=(",", ":"))
                output.append(record)
    return output, all_cpu, all_gpu

def _interface_effects(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    by_key = {
        (row["dataset"], row["interface"], row["method"], row["model_seed"], row["batch_size"]): row
        for row in rows
    }
    output = []
    all_cpu = []
    all_gpu = []
    for dataset in DATASETS:
        for method in METHODS:
            for batch_size in BATCH_SIZES:
                cpu = []
                gpu = []
                for seed in MODEL_SEEDS:
                    current = by_key[(dataset, "current", method, seed, batch_size)]
                    early = by_key[(dataset, "early", method, seed, batch_size)]
                    cpu.append(current["cpu_encoder_plus_defense_latency_seconds"] - early["cpu_encoder_plus_defense_latency_seconds"])
                    gpu.append(current["gpu_classifier_latency_seconds"] - early["gpu_classifier_latency_seconds"])
                all_cpu.extend(cpu)
                all_gpu.extend(gpu)
                record: dict[str, Any] = {"dataset": dataset, "method": method, "batch_size": batch_size}
                for name, values in (("cpu", cpu), ("gpu", gpu)):
                    summary = student_t_interval(values)
                    record.update({f"{name}_{key}": value for key, value in summary.items()})
                    record[f"{name}_seed_differences_seconds"] = json.dumps(values, separators=(",", ":"))
                output.append(record)
    return output, all_cpu, all_gpu

def _figure(path_pdf: Path, path_svg: Path, group_lookup: Mapping[tuple[Any, ...], dict[str, Any]]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.hashsalt": "eng_compute_v1",
        }
    )
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"standard": "#0072B2", "afd": "#D55E00", "laplace": "#009E73", "learned": "#CC79A7"}
    markers = {"early": "o", "current": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    x = np.arange(len(METHODS), dtype=float)
    for column, dataset in enumerate(DATASETS):
        ax = axes[0, column]
        for batch_index, batch_size in enumerate(BATCH_SIZES):
            for interface_index, interface_name in enumerate(INTERFACES):
                offset = (batch_index * 2 + interface_index - 1.5) * 0.08
                means = []
                lows = []
                highs = []
                for method in METHODS:
                    summary = group_lookup[(dataset, interface_name, method, batch_size)]["metrics"]["cpu_encoder_plus_defense_latency_seconds"]
                    scale = 1000.0 / batch_size
                    means.append(float(summary["mean"]) * scale)
                    lows.append((float(summary["mean"]) - float(summary["ci95_low"])) * scale)
                    highs.append((float(summary["ci95_high"]) - float(summary["mean"])) * scale)
                ax.errorbar(
                    x + offset,
                    means,
                    yerr=np.asarray([lows, highs]),
                    color="#222222" if batch_size == 1 else "#777777",
                    marker=markers[interface_name],
                    linestyle="-" if batch_size == 1 else "--",
                    linewidth=0.9,
                    capsize=2,
                    label=f"{interface_name}, batch {batch_size}",
                )
        ax.set_yscale("log")
        ax.set_xticks(x, [method.capitalize() for method in METHODS], rotation=20)
        ax.set_ylabel("CPU latency per item (ms, log scale)")
        ax.set_title("CelebA" if dataset == "celeba" else "CIFAR-10")
        ax.grid(axis="y", alpha=0.25)
        if column == 1:
            ax.legend(frameon=False, ncol=2, loc="upper center")

    ax = axes[1, 0]
    positions = np.arange(8)
    labels = []
    laplace_means = []
    learned_means = []
    laplace_errors = [[], []]
    learned_errors = [[], []]
    index = 0
    for dataset in DATASETS:
        for interface_name in INTERFACES:
            for batch_size in BATCH_SIZES:
                labels.append(f"{dataset[:3]}-{interface_name[0]}-b{batch_size}")
                for method, means, errors in (
                    ("laplace", laplace_means, laplace_errors),
                    ("learned", learned_means, learned_errors),
                ):
                    summary = group_lookup[(dataset, interface_name, method, batch_size)]["metrics"]["defense_only_latency_seconds"]
                    scale = 1e6 / batch_size
                    means.append(float(summary["mean"]) * scale)
                    errors[0].append((float(summary["mean"]) - float(summary["ci95_low"])) * scale)
                    errors[1].append((float(summary["ci95_high"]) - float(summary["mean"])) * scale)
                index += 1
    ax.errorbar(positions - 0.08, laplace_means, yerr=np.asarray(laplace_errors), marker="o", color=colors["laplace"], linestyle="none", capsize=2, label="Laplace")
    ax.errorbar(positions + 0.08, learned_means, yerr=np.asarray(learned_errors), marker="s", color=colors["learned"], linestyle="none", capsize=2, label="Learned")
    ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Defense-only latency per item (µs, log scale)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    labels = []
    standard_means = []
    afd_means = []
    for dataset in DATASETS:
        for interface_name in INTERFACES:
            for batch_size in BATCH_SIZES:
                labels.append(f"{dataset[:3]}-{interface_name[0]}-b{batch_size}")
                for method, target in (("standard", standard_means), ("afd", afd_means)):
                    summary = group_lookup[(dataset, interface_name, method, batch_size)]["metrics"]["gpu_classifier_latency_seconds"]
                    target.append(float(summary["mean"]) * 1000 / batch_size)
    width = 0.36
    ax.bar(positions - width / 2, standard_means, width, color=colors["standard"], label="Standard")
    ax.bar(positions + width / 2, afd_means, width, color=colors["afd"], label="AFD")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("GPU classifier latency per item (ms)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="top")
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, metadata={"Creator": "eng_compute_v1", "CreationDate": None, "ModDate": None})
    fig.savefig(path_svg, metadata={"Creator": "eng_compute_v1", "Date": None})
    plt.close(fig)
FIGURE_SIZE_INCHES = (7.2, 5.65)

LEGEND_ENTRIES = {
    "ab": [
        "early, batch 1",
        "current, batch 1",
        "early, batch 32",
        "current, batch 32",
    ],
    "c": ["Laplace", "Learned"],
    "d": ["Standard", "AFD"],
}

def _route_legend(source: Any, target: Any, *, role: str, ncol: int) -> None:
    target.set_axis_off()
    target._eng_figure_role = f"legend_{role}"
    target._eng_legend_source = source

    def legend(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise ComputeContractError(
                "Sealed Figure 6 changed to positional legend arguments"
            )
        if kwargs.get("frameon") is not False:
            raise ComputeContractError("Sealed Figure 6 legend is no longer frame-free")
        handles, labels = source.get_legend_handles_labels()
        if labels != LEGEND_ENTRIES[role]:
            raise ComputeContractError(
                f"Sealed Figure 6 legend entries differ for {role}: {labels}"
            )
        routed = dict(kwargs)
        routed["ncol"] = ncol
        routed["loc"] = "center"
        artist = target.legend(handles, labels, **routed)
        target._eng_routed_handles = handles
        target._eng_legend_artist = artist
        return artist

    source.legend = legend

def _legend_band_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    if args != (2, 2):
        raise ComputeContractError(
            f"Sealed Figure 6 subplot shape differs: args={args!r}"
        )
    expected_kwargs = {"figsize": (7.2, 5.0), "constrained_layout": True}
    if kwargs != expected_kwargs:
        raise ComputeContractError(
            f"Sealed Figure 6 subplot options differ: kwargs={kwargs!r}"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=(0.14, 1.0, 0.14, 1.0),
        hspace=0.10,
        wspace=0.08,
    )
    legend_ab = figure.add_subplot(grid[0, :])
    axes = np.asarray(
        [
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
            [figure.add_subplot(grid[3, 0]), figure.add_subplot(grid[3, 1])],
        ],
        dtype=object,
    )
    legend_c = figure.add_subplot(grid[2, 0])
    legend_d = figure.add_subplot(grid[2, 1])
    for role, axis in zip(("a", "b", "c", "d"), axes.flat):
        axis._eng_figure_role = f"data_{role}"
    _route_legend(axes[0, 1], legend_ab, role="ab", ncol=4)
    _route_legend(axes[1, 0], legend_c, role="c", ncol=2)
    _route_legend(axes[1, 1], legend_d, role="d", ncol=2)
    return figure, axes

@contextmanager
def _legend_band_adapter() -> Iterator[None]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    original = plt.subplots
    plt.subplots = _legend_band_subplots
    try:
        yield
    finally:
        plt.subplots = original

def _axis_signature(axis: Any) -> dict[str, Any]:
    lines = []
    for artist in axis.lines:
        lines.append(
            {
                "x": artist.get_xdata().tolist(),
                "y": artist.get_ydata().tolist(),
                "label": artist.get_label(),
                "color": artist.get_color(),
                "marker": artist.get_marker(),
                "linestyle": artist.get_linestyle(),
            }
        )
    collections = []
    for artist in axis.collections:
        segments = getattr(artist, "get_segments", lambda: [])()
        collections.append([segment.tolist() for segment in segments])
    patches = []
    for artist in axis.patches:
        if all(
            hasattr(artist, name)
            for name in ("get_x", "get_y", "get_width", "get_height")
        ):
            patches.append(
                [
                    artist.get_x(),
                    artist.get_y(),
                    artist.get_width(),
                    artist.get_height(),
                ]
            )
    return {
        "xscale": axis.get_xscale(),
        "yscale": axis.get_yscale(),
        "xlim": list(axis.get_xlim()),
        "ylim": list(axis.get_ylim()),
        "xticks": axis.get_xticks().tolist(),
        "yticks": axis.get_yticks().tolist(),
        "lines": lines,
        "collections": collections,
        "patches": patches,
    }
