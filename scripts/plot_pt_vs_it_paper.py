"""Generate paper-ready plots for pretrained vs post-trained comparison.

Produces a 1×2 panel figure (Gemma3 | Ministral3) for MSE, Bias, and Variance.

Color scheme:
  Gemma3   – blues (NATURE_COLORS), light→dark by size (1B→12B→27B)
  Ministral – warm browns,           light→dark by size (3B→8B→14B)
  Hatch:    solid = post-trained, '//' = pretrained (same color per size)

No axis titles; captions carry all explanatory text.
Output: results/figures/hf_pt_vs_it/
"""
from __future__ import annotations

from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from src.plotting.style import (
    MAIN_PLOT_SETTINGS, STAT_DISPLAY_NAMES, set_plot_theme, save_figure,
)
from src.utils.models import MODEL_DISPLAY_NAMES, normalize_model_name

# ── Data ────────────────────────────────────────────────────────────────────
CSV_PATH = (
    PROJECT_ROOT / "results" / "processed" / "summaries"
    / "toxicity_generated__female__hf_full.csv"
)

# ── Color scheme ─────────────────────────────────────────────────────────────
# Gemma: blues from NATURE_COLORS, light→dark (1B→12B→27B)
GEMMA_COLORS = ["#A1CAF1", "#4DBBD5", "#3C5488"]

# Ministral: warm browns, light→dark (3B→8B→14B)
MINISTRAL_COLORS = ["#D4956A", "#A05C35", "#6B3420"]

# Hatch: post-trained = solid, pretrained = '//'
HATCH_POST = None
HATCH_PRE  = "//"

# Bar geometry
BAR_W     = 0.65
INTRA_GAP = 0.10   # gap between the two bars within a size group
INTER_GAP = 0.85   # gap between size groups


def _dn(system: str) -> str:
    return MODEL_DISPLAY_NAMES.get(normalize_model_name(system), system)


def _draw_baseline(ax, summary, stat):
    """Draw horizontal reference lines for direct and perspective baselines."""
    handles = []
    for bi, system in enumerate(["direct", "perspective"]):
        row = summary[summary["system"] == system]
        if row.empty:
            continue
        val = float(row.iloc[0][f"{stat}_mean"])
        se  = float(row.iloc[0][f"{stat}_se"])
        color = MAIN_PLOT_SETTINGS[system]["color"]
        ls    = "--" if bi == 0 else "-."
        ax.axhline(val, color=color, linestyle=ls, linewidth=2.0, alpha=0.9, zorder=3)
        ax.axhspan(val - se, val + se, color=color, alpha=0.12, zorder=0)
        handles.append(
            mlines.Line2D([], [], color=color, linestyle=ls,
                          linewidth=2.0, label=_dn(system))
        )
    return handles


def _variant_legend_handles():
    """Solid vs hatched patches for post-trained / pretrained legend."""
    return [
        mpatches.Patch(facecolor="lightgrey", edgecolor="black",
                       hatch=HATCH_POST, label="Post-trained"),
        mpatches.Patch(facecolor="lightgrey", edgecolor="black",
                       hatch=HATCH_PRE,  label="Pretrained"),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Per-family standalone plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_family(ax, summary, stat, sizes, variants, colors):
    """
    Draw grouped bars for one family onto *ax*.

    variants: list of (sys_template, hatch, label)
              where sys_template uses '{s}' for size substitution.
    colors:   list of colors indexed by size.
    Returns (xtick_pos, xtick_labels).
    """
    pos, vals, errs, cols, hats = [], [], [], [], []
    xtick_pos, xtick_labels = [], []
    cur = 0.0

    for si, size in enumerate(sizes):
        gp = []
        for tmpl, hatch, _ in variants:
            sys_key = tmpl.format(s=size)
            row = summary[summary["system"] == sys_key]
            if row.empty:
                cur += BAR_W + INTRA_GAP
                continue
            pos.append(cur)
            vals.append(float(row.iloc[0][f"{stat}_mean"]))
            errs.append(float(row.iloc[0][f"{stat}_se"]))
            cols.append(colors[si])
            hats.append(hatch)
            gp.append(cur)
            cur += BAR_W + INTRA_GAP
        if gp:
            xtick_pos.append(float(np.mean(gp)))
            xtick_labels.append(size.upper())
        cur += INTER_GAP

    if not pos:
        return [], []

    ax.bar(
        pos, vals, width=BAR_W, yerr=errs,
        color=cols, hatch=hats, edgecolor="black", linewidth=0.8,
        capsize=4, alpha=0.9, zorder=2,
        error_kw={"ecolor": "black", "zorder": 4},
    )
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    return xtick_pos, xtick_labels


def plot_gemma(summary, stat, output_path):
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    variants = [
        ("gemma3-it:{s}", HATCH_POST, "Post-trained"),
        ("gemma3-pt:{s}", HATCH_PRE,  "Pretrained"),
    ]
    tp, tl = _plot_family(ax, summary, stat, ["1b", "12b", "27b"],
                          variants, GEMMA_COLORS)

    bh = _draw_baseline(ax, summary, stat)
    ax.set_xticks(tp); ax.set_xticklabels(tl, fontsize=13)
    ax.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)

    legend_h = _variant_legend_handles() + bh
    ax.legend(handles=legend_h, fontsize=11, frameon=True, framealpha=0.95,
              edgecolor="gray", loc="best")
    sns.despine()
    save_figure(output_path)


def plot_ministral(summary, stat, output_path):
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    variants = [
        ("ministral3:{s}",      HATCH_POST, "Post-trained"),
        ("ministral3-base:{s}", HATCH_PRE,  "Pretrained"),
    ]
    tp, tl = _plot_family(ax, summary, stat, ["3b", "8b", "14b"],
                          variants, MINISTRAL_COLORS)

    bh = _draw_baseline(ax, summary, stat)
    ax.set_xticks(tp); ax.set_xticklabels(tl, fontsize=13)
    ax.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)

    legend_h = _variant_legend_handles() + bh
    ax.legend(handles=legend_h, fontsize=11, frameon=True, framealpha=0.95,
              edgecolor="gray", loc="best")
    sns.despine()
    save_figure(output_path)


# ──────────────────────────────────────────────────────────────────────────────
# 1×2 panel: Gemma3 | Ministral3
# ──────────────────────────────────────────────────────────────────────────────

PANEL_CONFIGS = [
    (
        "Gemma 3", ["1b", "12b", "27b"],
        [
            ("gemma3-it:{s}", HATCH_POST, "Post-trained"),
            ("gemma3-pt:{s}", HATCH_PRE,  "Pretrained"),
        ],
        GEMMA_COLORS,
    ),
    (
        "Ministral 3", ["3b", "8b", "14b"],
        [
            ("ministral3:{s}",      HATCH_POST, "Post-trained"),
            ("ministral3-base:{s}", HATCH_PRE,  "Pretrained"),
        ],
        MINISTRAL_COLORS,
    ),
]


def plot_panel(summary, stat, output_path):
    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=100)

    for ax, (title, sizes, variants, colors) in zip(axes, PANEL_CONFIGS):
        tp, tl = _plot_family(ax, summary, stat, sizes, variants, colors)
        if not tp:
            ax.set_visible(False)
            continue

        bh = _draw_baseline(ax, summary, stat)

        ax.set_xticks(tp)
        ax.set_xticklabels(tl, fontsize=11)
        ax.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)

        legend_h = _variant_legend_handles() + bh
        ax.legend(handles=legend_h, fontsize=9, frameon=True, framealpha=0.95,
                  edgecolor="gray", loc="best")
        sns.despine(ax=ax)

    plt.tight_layout()
    save_figure(output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = PROJECT_ROOT / "results" / "figures" / "paper" / "hf_pt_it"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"Input not found: {CSV_PATH}")
        print("Run plot_hf_females.py first to generate the summary CSV.")
        return
    summary = pd.read_csv(CSV_PATH)
    summary = summary[summary["n_annotators"] == 1].copy()
    print(f"Loaded {len(summary)} rows at n=1")

    for stat in ["mse", "bias", "var"]:
        print(f"\n── {stat.upper()} ──")

        out = out_dir / f"gemma3_pt_it_{stat}.pdf"
        plot_gemma(summary, stat, out);    print(f"  {out.name}")

        out = out_dir / f"ministral3_pt_it_{stat}.pdf"
        plot_ministral(summary, stat, out); print(f"  {out.name}")

        out = out_dir / f"panel_pt_it_{stat}.pdf"
        plot_panel(summary, stat, out);    print(f"  {out.name}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
