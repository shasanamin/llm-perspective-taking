from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.plotting.style import (
    CUSTOM_MARKERS,
    DEFAULT_FIGSIZE,
    NATURE_COLORS,
    STAT_DISPLAY_NAMES,
    save_figure,
    set_plot_theme,
)


def format_dices_label(target: str) -> str:
    label = target
    label = label.replace("LatinX, Latino, Hispanic or Spanish Origin", "Latino")
    label = label.replace("rater_gender=Woman", "Female")
    label = label.replace("rater_gender=Man", "Male")
    label = label.replace("rater_race=Black/African American", "Black")
    label = label.replace("rater_race=White", "White")
    label = label.replace("rater_race=Asian/Asian subcontinent", "Asian")
    label = label.replace("rater_age=gen z", "Gen Z")
    label = label.replace("rater_age=gen x+", "Gen X+")
    label = label.replace("rater_age=millenial", "Millenial")
    label = label.replace("rater_education=College degree or higher", "College Ed")
    label = label.replace("rater_education=High school or below", "HS below")
    label = label.replace(",", " x ")
    return label


def plot_stat_line(summary_map: dict, stat: str, output_path: str | Path, legend_order: list[str] | None = None) -> None:
    set_plot_theme()
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=100)
    order = legend_order or list(summary_map)
    linestyles = {1: "-", 2: "--", 3: ":"}

    for index, key in enumerate(order):
        if key not in summary_map or summary_map[key].empty:
            continue
        summary = summary_map[key].set_index("n_annotators").sort_index()
        axis.errorbar(
            summary.index,
            summary[f"{stat}_mean"],
            yerr=summary[f"{stat}_se"],
            label=format_dices_label(key),
            color=NATURE_COLORS[index % len(NATURE_COLORS)],
            marker=CUSTOM_MARKERS[index % len(CUSTOM_MARKERS)],
            linestyle=linestyles.get(int(summary["L"].iloc[0]), "-"),
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
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=12, frameon=True, framealpha=0.95, edgecolor="gray", loc="best", fancybox=True, handlelength=3.0)
    sns.despine()
    save_figure(output_path)


def plot_stat_at_one(summary_map: dict, stat: str, output_path: str | Path) -> None:
    set_plot_theme()
    figure, axis = plt.subplots(figsize=(DEFAULT_FIGSIZE[0] * 1.05, DEFAULT_FIGSIZE[1] * 1.3), dpi=100)

    for index, key in enumerate(summary_map):
        if summary_map[key].empty:
            continue
        summary = summary_map[key]
        summary = summary[summary["n_annotators"] == 1].sort_values("L")
        axis.errorbar(
            summary["L"],
            summary[f"{stat}_mean"],
            yerr=summary[f"{stat}_se"],
            label=str(key),
            color=NATURE_COLORS[index % len(NATURE_COLORS)],
            marker=CUSTOM_MARKERS[index % len(CUSTOM_MARKERS)],
            linewidth=3.5,
            capsize=4,
            capthick=1.5,
            markersize=10,
            markeredgewidth=2,
            markeredgecolor="white",
            alpha=0.95,
        )

    axis.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xticks(sorted({level for summary in summary_map.values() for level in summary["L"].unique()}))
    axis.set_xticklabels([])
    axis.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=16, fontweight="bold")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            ncol=1,
            fontsize=15,
            columnspacing=1.0,
            handlelength=1.0,
        )
    sns.despine()
    save_figure(output_path)
