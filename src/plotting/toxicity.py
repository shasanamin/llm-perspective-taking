from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.plotting.style import (
    CUSTOM_MARKERS,
    DEFAULT_FIGSIZE,
    MAIN_PLOT_SETTINGS,
    MODEL_FAMILIES,
    MODEL_DISPLAY_NAMES,
    NATURE_COLORS,
    STAT_DISPLAY_NAMES,
    display_name,
    save_figure,
    set_plot_theme,
)


def plot_grouped_bars(summary_by_target: dict[str, any], stat: str, output_path: str | Path) -> None:
    set_plot_theme()
    legend_order = ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1", "direct", "perspective"]
    if stat == "bias":
        legend_order = ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1", "perspective"]

    figure, axis = plt.subplots(
        figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1] * (1.0 if stat == "var" else 1.3)),
        dpi=100,
    )

    positions = []
    values = []
    errors = []
    colors = []
    hatches = []
    group_centers = []
    current_position = 0.0

    for target_label, summary in summary_by_target.items():
        summary = summary[summary["n_annotators"] == 1]
        group_positions = []
        target_order = list(legend_order)
        if "perspective_out" in summary["system"].unique() and "perspective_out" not in target_order:
            target_order.append("perspective_out")

        for system in target_order:
            if system not in summary["system"].unique():
                continue
            row = summary[summary["system"] == system].iloc[0]
            positions.append(current_position)
            values.append(row[f"{stat}_mean"])
            errors.append(row[f"{stat}_se"])
            colors.append(MAIN_PLOT_SETTINGS[system]["color"])
            hatches.append(MAIN_PLOT_SETTINGS[system]["hatch"])
            group_positions.append(current_position)
            current_position += 0.7

        group_centers.append(float(np.mean(group_positions)))
        current_position += 0.8

    axis.bar(
        positions,
        values,
        width=0.7,
        yerr=errors,
        color=colors,
        hatch=hatches,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.9,
    )
    axis.set_xticks(group_centers)
    axis.set_xticklabels(list(summary_by_target.keys()), fontsize=14)
    axis.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=20, fontweight="bold")
    axis.tick_params(axis="y", labelsize=14)
    sns.despine()
    save_figure(output_path)


def plot_llm_vs_humans(summary: any, output_path: str | Path) -> None:
    set_plot_theme()
    order = ["direct", "perspective", "gpt-oss:20b", "gpt-oss:120b", "gpt-5.1"]
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=100)

    for system in order:
        if system not in summary["system"].unique():
            continue
        df = summary[summary["system"] == system].set_index("n_annotators").sort_index()
        axis.errorbar(
            df.index,
            df["mse_mean"],
            yerr=df["mse_se"],
            label=display_name(system),
            color=MAIN_PLOT_SETTINGS[system]["color"],
            marker=MAIN_PLOT_SETTINGS[system]["marker"],
            linestyle=MAIN_PLOT_SETTINGS[system]["linestyle"],
            linewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=8,
            markeredgewidth=1,
            markeredgecolor="white",
            alpha=0.95,
        )

    axis.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xticks([1, 2, 3, 5, 10])
    axis.set_xlabel("Number of Annotations Per Comment", fontsize=16, fontweight="bold")
    axis.set_ylabel("Error", fontsize=16, fontweight="bold")
    axis.legend(fontsize=14, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", fancybox=True, ncols=2)
    sns.despine()
    save_figure(output_path)


def plot_two_target_comparison(summary_by_target: dict[str, any], output_path: str | Path, model_name: str = "gpt-oss:120b") -> None:
    set_plot_theme()
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=100)

    order = [model_name, "direct", "perspective", "perspective_out"]
    linestyles = {"Male": "-", "Female": "--"}
    for target_label, summary in summary_by_target.items():
        for system in order:
            if system not in summary["system"].unique():
                continue
            df = summary[summary["system"] == system].set_index("n_annotators").sort_index()
            axis.errorbar(
                df.index,
                df["mse_mean"],
                yerr=df["mse_se"],
                label=display_name(system) if target_label == "Male" else None,
                color=MAIN_PLOT_SETTINGS[system]["color"],
                marker=MAIN_PLOT_SETTINGS[system]["marker"],
                linestyle=linestyles[target_label],
                linewidth=2,
                capsize=4,
                capthick=1.5,
                markersize=8,
                markeredgewidth=1,
                markeredgecolor="white",
                alpha=0.9,
            )

    axis.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xticks([1, 2, 3, 5, 10])
    axis.set_xlabel("Number of Annotations Per Comment", fontsize=16, fontweight="bold")
    axis.set_ylabel("Error", fontsize=16, fontweight="bold")
    axis.legend(fontsize=14, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", fancybox=True, ncols=2)
    sns.despine()
    save_figure(output_path)


def plot_model_family(summary: any, stat: str, output_path: str | Path) -> None:
    set_plot_theme()
    filtered = summary[summary["n_annotators"] == 1].copy()
    figure, axis = plt.subplots(figsize=(DEFAULT_FIGSIZE[0] * 1.25, DEFAULT_FIGSIZE[1]), dpi=100)

    palettes = [
        ["#F39B7F", "#D17B60", "#A25641"],
        ["#4DBBD5", "#318BA0", "#1A4D5C"],
        ["#E64B35", "#C63623", "#9B2115"],
        ["#8491B4", "#5E6A8C", "#404963"],
        ["#91D1C2", "#5BAD98", "#3A7D6D"],
        ["#F39B7F", "#D17B60", "#A25641"],
    ]

    family_palettes = {family: palettes[index] for index, family in enumerate(sorted(MODEL_FAMILIES))}
    positions = []
    labels = []
    colors = []
    systems = []
    families = []
    current_position = 0.0

    for family, family_models in MODEL_FAMILIES.items():
        for index, system in enumerate(family_models):
            if system not in filtered["system"].unique():
                continue
            positions.append(current_position)
            systems.append(system)
            families.append(family)
            label = system.split(":")[-1].upper() if ":" in system else system.replace("gpt-5.1_reasoning=", "R=").upper()
            labels.append(label)
            colors.append(family_palettes[family][min(index, 2)])
            current_position += 1
        current_position += 0.8

    values = []
    errors = []
    for system in systems:
        row = filtered[filtered["system"] == system].iloc[0]
        values.append(row[f"{stat}_mean"])
        errors.append(row[f"{stat}_se"])

    axis.bar(
        positions,
        values,
        width=0.7,
        yerr=errors,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.9,
    )

    for baseline_index, system in enumerate(["direct", "perspective"]):
        if system not in filtered["system"].unique():
            continue
        row = filtered[filtered["system"] == system].iloc[0]
        value = row[f"{stat}_mean"]
        error = row[f"{stat}_se"]
        axis.axhline(
            y=value,
            color=MAIN_PLOT_SETTINGS[system]["color"],
            linestyle=["--", "-."][baseline_index],
            linewidth=2.5,
            alpha=0.8,
            label=display_name(system),
        )
        axis.axhspan(value - error, value + error, color=MAIN_PLOT_SETTINGS[system]["color"], alpha=0.15, zorder=0)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    axis.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=18, fontweight="bold")
    axis.legend(fontsize=12, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", fancybox=True)
    sns.despine()
    save_figure(output_path)


def plot_nonbinary_reasoning_bias(summary: any, output_path: str | Path) -> None:
    set_plot_theme()
    filtered = summary[summary["n_annotators"] == 1].copy()
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=100)

    model_pairs = [
        ("gpt-5.1", "gpt-5.1_reasoning=high"),
        ("qwen3:1.7b", "qwen3-r:1.7b"),
        ("qwen3:8b", "qwen3-r:8b"),
        ("qwen3:32b", "qwen3-r:32b"),
    ]
    colors = ["#E64B35", "#8491B4", "#5E6A8C", "#404963"]
    positions = []
    values = []
    errors = []
    bar_colors = []
    hatches = []
    labels = []
    current_position = 0.0

    for group_index, pair in enumerate(model_pairs):
        for model_index, system in enumerate(pair):
            if system not in filtered["system"].unique():
                continue
            row = filtered[filtered["system"] == system].iloc[0]
            positions.append(current_position)
            values.append(row["bias_mean"])
            errors.append(row["bias_se"])
            bar_colors.append(colors[group_index])
            hatches.append(None if model_index == 0 else "//")
            labels.append(display_name(system).replace("GPT-5.1-R=H", "GPT-5.1-R"))
            current_position += 0.8
        current_position += 0.8

    axis.bar(
        positions,
        values,
        width=0.8,
        yerr=errors,
        color=bar_colors,
        hatch=hatches,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.9,
    )

    for baseline_index, system in enumerate(["direct", "perspective"]):
        if system not in filtered["system"].unique():
            continue
        row = filtered[filtered["system"] == system].iloc[0]
        value = row["bias_mean"]
        error = row["bias_se"]
        axis.axhspan(value - error, value + error, color=MAIN_PLOT_SETTINGS[system]["color"], alpha=0.15, zorder=0)
        axis.axhline(
            y=value,
            color=MAIN_PLOT_SETTINGS[system]["color"],
            linestyle=["--", "-."][baseline_index],
            linewidth=2.5,
            alpha=0.8,
            label=display_name(system),
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    axis.set_ylabel(STAT_DISPLAY_NAMES["bias"], fontsize=20, fontweight="bold")
    axis.legend(fontsize=14, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", fancybox=True)
    sns.despine()
    save_figure(output_path)


def plot_stat_line(summary_map: dict, stat: str, output_path: str | Path, legend_order: list | None = None, display_names: dict | None = None) -> None:
    set_plot_theme()
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=100)
    order = legend_order or list(summary_map)
    linestyles = ["-", "--", "-.", ":"]

    for index, key in enumerate(order):
        summary = summary_map[key].set_index("n_annotators").sort_index()
        axis.errorbar(
            summary.index,
            summary[f"{stat}_mean"],
            yerr=summary[f"{stat}_se"],
            label=(display_names or {}).get(key, key),
            color=NATURE_COLORS[index % len(NATURE_COLORS)],
            marker=CUSTOM_MARKERS[index % len(CUSTOM_MARKERS)],
            linestyle=linestyles[index % len(linestyles)],
            linewidth=2,
            capsize=4,
            capthick=1.5,
            markersize=8,
            markeredgewidth=1,
            markeredgecolor="white",
            alpha=0.92,
        )

    axis.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xticks([1, 2, 3, 5, 10])
    axis.set_xlabel("Number of Annotations Per Comment", fontsize=16, fontweight="bold")
    axis.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=16, fontweight="bold")
    axis.legend(fontsize=14, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", ncols=1, fancybox=True, handlelength=3.0)
    sns.despine()
    save_figure(output_path)


def plot_stat_bar(summary_map: dict, output_path: str | Path, legend_order: list | None = None, display_names: dict | None = None, show_legend: bool = True) -> None:
    set_plot_theme()
    figure, axis = plt.subplots(figsize=(DEFAULT_FIGSIZE[0] * 0.55, DEFAULT_FIGSIZE[1]), dpi=100)
    order = legend_order or list(summary_map)
    position = 0.0
    hatches = [None, "//", "xx", ".."]

    for index, key in enumerate(order):
        summary = summary_map[key]
        summary = summary[summary["n_annotators"] == 1]
        axis.bar(
            position,
            summary["bias_mean"].iloc[0],
            yerr=summary["bias_se"].iloc[0],
            label=(display_names or {}).get(key, key),
            width=0.7,
            align="edge",
            color=NATURE_COLORS[index % len(NATURE_COLORS)],
            hatch=hatches[index % len(hatches)],
            edgecolor="black",
            error_kw={"ecolor": "black", "markeredgecolor": "black"},
            linewidth=0.8,
            capsize=4,
            alpha=0.9,
        )
        position += 0.7

    axis.set_xlim(-0.5, position + 0.5)
    axis.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xticks([])
    axis.set_ylabel(STAT_DISPLAY_NAMES["bias"], fontsize=16, fontweight="bold")
    if show_legend:
        axis.legend(fontsize=12, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", ncols=1, fancybox=True, handlelength=2.0)
    sns.despine()
    save_figure(output_path)


def mixture_groups(base_models: list[str]) -> list[str]:
    groups = list(base_models)
    for pair in combinations(base_models, 2):
        groups.append(" & ".join(pair))
    groups.append(" & ".join(base_models))
    return groups


def plot_mixture_error_reduction(
    summary: any,
    mixture_map: dict[str, list[str]],
    output_dir: str | Path,
    filename_prefix: str = "",
) -> None:
    set_plot_theme()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    colors = ["#4DBBD5", "#3C5488", "#E64B35", "#00A087", "#F39B7F", "#A25641", "black"]
    markers = ["<", ">", "^", ".", "o", "D", None]

    for key, models in mixture_map.items():
        figure, axis = plt.subplots(figsize=(DEFAULT_FIGSIZE[0], int(DEFAULT_FIGSIZE[1] * 1.4)), dpi=100)

        for model_index, system in enumerate(models):
            df = summary[summary["system"] == system].set_index("n_annotators").sort_index()
            if df.empty or 1 not in df.index:
                continue
            base_value = df.loc[1, "mse_mean"]
            values = [mse - base_value for mse in df["mse_mean"].tolist()]

            if system in MODEL_DISPLAY_NAMES:
                label = display_name(system)
                linestyle = "-"
            elif system.count("&") == 1:
                label = " & ".join(display_name(name.strip()) for name in system.split("&"))
                linestyle = "--"
            else:
                label = " & ".join(display_name(name.strip()) for name in system.split("&"))
                linestyle = ":"

            axis.errorbar(
                df.index,
                values,
                label=label,
                color=colors[model_index % len(colors)],
                marker=markers[model_index % len(markers)],
                linestyle=linestyle,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                markersize=8,
                markeredgewidth=1,
                markeredgecolor="white",
                alpha=0.8,
            )

        axis.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xticks([1, 2, 3, 5, 10])
        axis.set_xlabel("Number of Annotations Per Comment", fontsize=16, fontweight="bold")
        axis.set_ylabel("Error Reduction", fontsize=16, fontweight="bold")
        axis.legend(
            bbox_to_anchor=(-0.025, -0.35, 1.05, 0.1),
            mode="expand",
            loc="upper center",
            ncol=2,
            fontsize=11,
            handlelength=5,
        )
        sns.despine()
        filename = f"{filename_prefix}{key}.pdf"
        save_figure(output_root / filename)
