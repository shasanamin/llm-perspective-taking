"""Generate bias/variance/MSE figures for the HF full females experiment.

Produces per-family grouped bar charts and an all-models overview, each for
MSE, Bias, and Variance at n=1.  Human "direct" and "perspective" baselines
are drawn as horizontal reference lines on every plot.

Output directory:  results/figures/hf_full_females/
"""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401  (adds src/ to sys.path)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

from src.analysis.toxicity import summarize_main_experiment
from src.paths import DATA_ROOT
from src.plotting.style import MAIN_PLOT_SETTINGS, STAT_DISPLAY_NAMES, set_plot_theme, save_figure
from src.utils.jsonl import iter_jsonl
from src.utils.models import MODEL_DISPLAY_NAMES, normalize_model_name
from src.datasets import canonical_target_label


HF_FULL_FEMALES_DIR = DATA_ROOT / "llm_annotations" / "toxicity_detection" / "generated" / "hf_full_females"


def load_hf_full_annotations() -> "pd.DataFrame":
    """Load only the hf_full_females JSONL files (avoids malformed files elsewhere)."""
    import pandas as pd
    records = []
    for path in sorted(HF_FULL_FEMALES_DIR.glob("*.jsonl")):
        for record in iter_jsonl(path):
            if record.get("dataset") != "toxicity_detection":
                continue
            record = dict(record)
            record["source_file"] = str(path)
            records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["model_name"] = df["model_name"].map(normalize_model_name)
    df["target_group_canonical"] = df["target_group_canonical"].map(canonical_target_label)
    df["value"] = pd.to_numeric(df["parsed_percentage"], errors="coerce") / 100.0
    return df


# ── Model registry ────────────────────────────────────────────────────────────

GEMMA_SIZES = ["1b", "12b", "27b"]
GEMMA_IT = [f"gemma3-it:{s}" for s in GEMMA_SIZES]
GEMMA_PT = [f"gemma3-pt:{s}" for s in GEMMA_SIZES]
GEMMA_ALL = GEMMA_IT + GEMMA_PT

MINISTRAL_SIZES = ["3b", "8b", "14b"]
MINISTRAL_BASE = [f"ministral3-base:{s}" for s in MINISTRAL_SIZES]
MINISTRAL_INSTRUCT = [f"ministral3:{s}" for s in MINISTRAL_SIZES]
MINISTRAL_R = [f"ministral3-r:{s}" for s in MINISTRAL_SIZES]
MINISTRAL_ALL = MINISTRAL_BASE + MINISTRAL_INSTRUCT + MINISTRAL_R

QWEN35_SIZES = ["0.8b", "4b", "9b"]
QWEN35 = [f"qwen3.5:{s}" for s in QWEN35_SIZES]
QWEN35_BASE = [f"qwen3.5-base:{s}" for s in QWEN35_SIZES]
QWEN35_ALL = QWEN35 + QWEN35_BASE

LLAMA_ALL = ["llama3.1:8b", "llama3.1:8b-instruct"]

ALL_HF_MODELS = GEMMA_ALL + MINISTRAL_ALL + QWEN35_ALL + LLAMA_ALL

# ── Colour palette ────────────────────────────────────────────────────────────

# Gemma -- teal gradient for IT, orange gradient for PT (light → dark by size)
GEMMA_IT_COLORS = ["#91D1C2", "#5BAD98", "#3A7D6D"]
GEMMA_PT_COLORS = ["#F5C6A0", "#D17B60", "#A25641"]

# Ministral -- one colour per variant type (consistent across sizes)
COL_BASE = "#4DBBD5"       # blue
COL_INSTRUCT = "#3C5488"   # navy
COL_REASON = "#E64B35"     # red

# Qwen3.5 -- purple shades for Instruct, gold shades for Base
QWEN_INST_COLORS = ["#B8BCD6", "#8491B4", "#404963"]
QWEN_BASE_COLORS = ["#FAE29C", "#F0B429", "#C8960C"]

# Llama -- warm brown tones
LLAMA_COLORS = ["#B09C85", "#7B6248"]

# Hatch per variant role
HATCHES: dict[str, str | None] = {
    "it": None,
    "pt": "//",
    "instruct": None,
    "base": "//",
    "r": "xx",
    "llama_base": "//",
    "llama_inst": None,
}

BAR_W = 0.65
INTRA_GAP = 0.12   # gap between bars inside a group
INTER_GAP = 0.90   # extra whitespace between size groups


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dn(system: str) -> str:
    """Return short display name for a system key."""
    return MODEL_DISPLAY_NAMES.get(normalize_model_name(system), system)


def _baseline_lines(ax, summary, stat) -> list:
    """Draw direct/perspective horizontal baselines; return legend handles."""
    handles = []
    for i, system in enumerate(["direct", "perspective"]):
        row = summary[summary["system"] == system]
        if row.empty:
            continue
        val = float(row.iloc[0][f"{stat}_mean"])
        se = float(row.iloc[0][f"{stat}_se"])
        color = MAIN_PLOT_SETTINGS[system]["color"]
        ls = "--" if i == 0 else "-."
        ax.axhline(val, color=color, linestyle=ls, linewidth=2.5, alpha=0.85, zorder=3)
        ax.axhspan(val - se, val + se, color=color, alpha=0.12, zorder=0)
        handles.append(mlines.Line2D([], [], color=color, linestyle=ls,
                                      linewidth=2.5, label=_dn(system)))
    return handles


def _draw_bars(ax, positions, values, errors, colors, hatches):
    ax.bar(
        positions, values,
        width=BAR_W,
        yerr=errors,
        color=colors,
        hatch=hatches,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.9,
        zorder=2,
        error_kw={"ecolor": "black", "zorder": 4},
    )


def _finalise(ax, xtick_pos, xtick_labels, stat, title, legend_handles, figsize):
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=13)
    ax.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.legend(handles=legend_handles, fontsize=11, frameon=True,
              framealpha=0.95, edgecolor="gray", loc="best")
    sns.despine()


# ── Per-family plots ──────────────────────────────────────────────────────────

def plot_gemma(summary: "pd.DataFrame", stat: str, output_path: Path) -> None:
    """Gemma3 IT vs PT grouped by size (1B / 12B / 27B)."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    pos, vals, errs, cols, hats = [], [], [], [], []
    xtick_pos, xtick_labels = [], []
    cur = 0.0

    for i, size in enumerate(GEMMA_SIZES):
        gp = []
        for variant, color, hk in [
            ("it", GEMMA_IT_COLORS[i], "it"),
            ("pt", GEMMA_PT_COLORS[i], "pt"),
        ]:
            sys_key = f"gemma3-{variant}:{size}"
            row = summary[summary["system"] == sys_key]
            if row.empty:
                cur += BAR_W + INTRA_GAP
                continue
            pos.append(cur); vals.append(float(row.iloc[0][f"{stat}_mean"]))
            errs.append(float(row.iloc[0][f"{stat}_se"]))
            cols.append(color); hats.append(HATCHES[hk])
            gp.append(cur); cur += BAR_W + INTRA_GAP
        if gp:
            xtick_pos.append(float(np.mean(gp)))
            xtick_labels.append(size.upper())
        cur += INTER_GAP

    _draw_bars(ax, pos, vals, errs, cols, hats)
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    baseline_handles = _baseline_lines(ax, summary, stat)

    legend_handles = [
        mpatches.Patch(facecolor=GEMMA_IT_COLORS[1], edgecolor="black", label="Instruct (IT)"),
        mpatches.Patch(facecolor=GEMMA_PT_COLORS[1], edgecolor="black", hatch="//", label="Pre-trained (PT)"),
    ] + baseline_handles

    _finalise(ax, xtick_pos, xtick_labels, stat, "Gemma 3  —  Female Target", legend_handles, (8, 4.5))
    save_figure(output_path)


def plot_ministral(summary: "pd.DataFrame", stat: str, output_path: Path) -> None:
    """Ministral3 Base / Instruct / Reasoning grouped by size."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)

    variant_cfg = [
        ("base",    COL_BASE,    "base",    "ministral3-base:{s}"),
        ("instruct",COL_INSTRUCT,"instruct","ministral3:{s}"),
        ("r",       COL_REASON,  "r",       "ministral3-r:{s}"),
    ]
    pos, vals, errs, cols, hats = [], [], [], [], []
    xtick_pos, xtick_labels = [], []
    cur = 0.0

    for size in MINISTRAL_SIZES:
        gp = []
        for vname, color, hk, tmpl in variant_cfg:
            sys_key = tmpl.format(s=size)
            row = summary[summary["system"] == sys_key]
            if row.empty:
                cur += BAR_W + INTRA_GAP
                continue
            pos.append(cur); vals.append(float(row.iloc[0][f"{stat}_mean"]))
            errs.append(float(row.iloc[0][f"{stat}_se"]))
            cols.append(color); hats.append(HATCHES[hk])
            gp.append(cur); cur += BAR_W + INTRA_GAP
        if gp:
            xtick_pos.append(float(np.mean(gp)))
            xtick_labels.append(size.upper())
        cur += INTER_GAP

    _draw_bars(ax, pos, vals, errs, cols, hats)
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    baseline_handles = _baseline_lines(ax, summary, stat)

    legend_handles = [
        mpatches.Patch(facecolor=COL_BASE,    edgecolor="black", hatch="//", label="Base"),
        mpatches.Patch(facecolor=COL_INSTRUCT, edgecolor="black",             label="Instruct"),
        mpatches.Patch(facecolor=COL_REASON,   edgecolor="black", hatch="xx", label="Reasoning"),
    ] + baseline_handles

    _finalise(ax, xtick_pos, xtick_labels, stat, "Ministral 3  —  Female Target", legend_handles, (9, 4.5))
    save_figure(output_path)


def plot_qwen35(summary: "pd.DataFrame", stat: str, output_path: Path) -> None:
    """Qwen3.5 Instruct vs Base grouped by size."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    pos, vals, errs, cols, hats = [], [], [], [], []
    xtick_pos, xtick_labels = [], []
    cur = 0.0

    for i, size in enumerate(QWEN35_SIZES):
        gp = []
        for sys_key, color, hk in [
            (f"qwen3.5:{size}",      QWEN_INST_COLORS[i], "instruct"),
            (f"qwen3.5-base:{size}", QWEN_BASE_COLORS[i], "base"),
        ]:
            row = summary[summary["system"] == sys_key]
            if row.empty:
                cur += BAR_W + INTRA_GAP
                continue
            pos.append(cur); vals.append(float(row.iloc[0][f"{stat}_mean"]))
            errs.append(float(row.iloc[0][f"{stat}_se"]))
            cols.append(color); hats.append(HATCHES[hk])
            gp.append(cur); cur += BAR_W + INTRA_GAP
        if gp:
            xtick_pos.append(float(np.mean(gp)))
            xtick_labels.append(size.upper())
        cur += INTER_GAP

    _draw_bars(ax, pos, vals, errs, cols, hats)
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    baseline_handles = _baseline_lines(ax, summary, stat)

    legend_handles = [
        mpatches.Patch(facecolor=QWEN_INST_COLORS[1], edgecolor="black",             label="Instruct"),
        mpatches.Patch(facecolor=QWEN_BASE_COLORS[1], edgecolor="black", hatch="//", label="Base"),
    ] + baseline_handles

    _finalise(ax, xtick_pos, xtick_labels, stat, "Qwen 3.5  —  Female Target", legend_handles, (8, 4.5))
    save_figure(output_path)


def plot_llama(summary: "pd.DataFrame", stat: str, output_path: Path) -> None:
    """Llama 3.1 Base vs Instruct for 8B."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=100)

    models_cfg = [
        ("llama3.1:8b",          LLAMA_COLORS[0], "llama_base", "8B Base"),
        ("llama3.1:8b-instruct", LLAMA_COLORS[1], "llama_inst", "8B Instruct"),
    ]
    pos, vals, errs, cols, hats = [], [], [], [], []
    xtick_pos, xtick_labels = [], []
    cur = 0.0

    for sys_key, color, hk, lbl in models_cfg:
        row = summary[summary["system"] == sys_key]
        if row.empty:
            continue
        pos.append(cur); vals.append(float(row.iloc[0][f"{stat}_mean"]))
        errs.append(float(row.iloc[0][f"{stat}_se"]))
        cols.append(color); hats.append(HATCHES[hk])
        xtick_pos.append(cur); xtick_labels.append(lbl)
        cur += BAR_W + INTRA_GAP + INTER_GAP

    if not pos:
        plt.close()
        return

    _draw_bars(ax, pos, vals, errs, cols, hats)
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    baseline_handles = _baseline_lines(ax, summary, stat)

    legend_handles = [
        mpatches.Patch(facecolor=LLAMA_COLORS[0], edgecolor="black", hatch="//", label="Base"),
        mpatches.Patch(facecolor=LLAMA_COLORS[1], edgecolor="black",             label="Instruct"),
    ] + baseline_handles

    _finalise(ax, xtick_pos, xtick_labels, stat, "Llama 3.1  —  Female Target", legend_handles, (5, 4.5))
    save_figure(output_path)


# ── All-models overview ───────────────────────────────────────────────────────

def plot_all_hf(summary: "pd.DataFrame", stat: str, output_path: Path) -> None:
    """All HF models in a single overview bar chart, grouped by family."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(17, 5.5), dpi=100)

    # Each entry: (system_key, bar_color, hatch_key)
    families: list[tuple[str, list[tuple[str, str, str | None]]]] = [
        ("Gemma 3", [
            (f"gemma3-it:{s}", GEMMA_IT_COLORS[i], None)    for i, s in enumerate(GEMMA_SIZES)
        ] + [
            (f"gemma3-pt:{s}", GEMMA_PT_COLORS[i], "//")    for i, s in enumerate(GEMMA_SIZES)
        ]),
        ("Ministral 3", [
            item
            for s in MINISTRAL_SIZES
            for item in [
                (f"ministral3-base:{s}", COL_BASE,    "//"),
                (f"ministral3:{s}",      COL_INSTRUCT, None),
                (f"ministral3-r:{s}",    COL_REASON,   "xx"),
            ]
        ]),
        ("Qwen 3.5", [
            item
            for i, s in enumerate(QWEN35_SIZES)
            for item in [
                (f"qwen3.5:{s}",      QWEN_INST_COLORS[i], None),
                (f"qwen3.5-base:{s}", QWEN_BASE_COLORS[i], "//"),
            ]
        ]),
        ("Llama 3.1", [
            ("llama3.1:8b",          LLAMA_COLORS[0], "//"),
            ("llama3.1:8b-instruct", LLAMA_COLORS[1], None),
        ]),
    ]

    pos, vals, errs, cols, hats = [], [], [], [], []
    family_centers, family_labels = [], []
    cur = 0.0

    for family_label, model_cfg in families:
        gp = []
        for sys_key, color, hatch in model_cfg:
            row = summary[summary["system"] == sys_key]
            if row.empty:
                cur += BAR_W + INTRA_GAP
                continue
            pos.append(cur); vals.append(float(row.iloc[0][f"{stat}_mean"]))
            errs.append(float(row.iloc[0][f"{stat}_se"]))
            cols.append(color); hats.append(hatch)
            gp.append(cur); cur += BAR_W + INTRA_GAP
        if gp:
            family_centers.append(float(np.mean(gp)))
            family_labels.append(family_label)
        cur += INTER_GAP * 1.4

    if not pos:
        plt.close()
        return

    _draw_bars(ax, pos, vals, errs, cols, hats)
    ax.set_xlim(min(pos) - 0.7, max(pos) + 0.7)
    baseline_handles = _baseline_lines(ax, summary, stat)

    # Family dividers
    divider_positions = []
    cur2 = 0.0
    for family_label, model_cfg in families[:-1]:
        count = sum(1 for s, *_ in model_cfg if not summary[summary["system"] == s].empty)
        cur2 += count * (BAR_W + INTRA_GAP) + INTER_GAP * 0.7
        divider_positions.append(cur2)
        cur2 += INTER_GAP * 0.7

    for xd in divider_positions:
        ax.axvline(xd, color="gray", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)

    legend_handles = [
        mpatches.Patch(facecolor="#888", edgecolor="black",             label="IT / Instruct"),
        mpatches.Patch(facecolor="#888", edgecolor="black", hatch="//", label="PT / Base"),
        mpatches.Patch(facecolor=COL_REASON, edgecolor="black", hatch="xx", label="Reasoning"),
    ] + baseline_handles

    ax.set_xticks(family_centers)
    ax.set_xticklabels(family_labels, fontsize=14, fontweight="bold")
    ax.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=17, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("All HF Models  —  Female Target", fontsize=14, fontweight="bold", pad=10)
    ax.legend(handles=legend_handles, fontsize=11, frameon=True, framealpha=0.95,
              edgecolor="gray", loc="best")
    sns.despine()
    save_figure(output_path)


# ── 2×2 family panel ─────────────────────────────────────────────────────────

def plot_family_panel(summary: "pd.DataFrame", stat: str, output_path: Path) -> None:
    """2×2 panel: Gemma / Ministral / Qwen3.5 / Llama for one stat."""
    set_plot_theme()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=100)
    axes = axes.flatten()

    configs = [
        # (title, sizes, variants[(sys_template, color, hatch_key, legend_label)])
        ("Gemma 3", GEMMA_SIZES, [
            ("gemma3-it:{s}", GEMMA_IT_COLORS, "it",       "Instruct (IT)"),
            ("gemma3-pt:{s}", GEMMA_PT_COLORS, "pt",       "Pre-trained (PT)"),
        ]),
        ("Ministral 3", MINISTRAL_SIZES, [
            ("ministral3-base:{s}", [COL_BASE]    * 3, "base",    "Base"),
            ("ministral3:{s}",      [COL_INSTRUCT]* 3, "instruct","Instruct"),
            ("ministral3-r:{s}",    [COL_REASON]  * 3, "r",       "Reasoning"),
        ]),
        ("Qwen 3.5", QWEN35_SIZES, [
            ("qwen3.5:{s}",      QWEN_INST_COLORS, "instruct", "Instruct"),
            ("qwen3.5-base:{s}", QWEN_BASE_COLORS, "base",     "Base"),
        ]),
        ("Llama 3.1", ["8b"], [
            ("llama3.1:{s}",          [LLAMA_COLORS[0]], "llama_base", "Base"),
            ("llama3.1:{s}-instruct", [LLAMA_COLORS[1]], "llama_inst", "Instruct"),
        ]),
    ]

    for ax, (title, sizes, variants) in zip(axes, configs):
        pos, vals, errs, cols, hats = [], [], [], [], []
        xtick_pos, xtick_labels = [], []
        cur = 0.0

        for s_idx, size in enumerate(sizes):
            gp = []
            for tmpl, colors_list, hk, _ in variants:
                sys_key = tmpl.format(s=size)
                row = summary[summary["system"] == sys_key]
                if row.empty:
                    cur += BAR_W + INTRA_GAP
                    continue
                c = colors_list[min(s_idx, len(colors_list) - 1)]
                pos.append(cur); vals.append(float(row.iloc[0][f"{stat}_mean"]))
                errs.append(float(row.iloc[0][f"{stat}_se"]))
                cols.append(c); hats.append(HATCHES[hk])
                gp.append(cur); cur += BAR_W + INTRA_GAP
            if gp:
                xtick_pos.append(float(np.mean(gp)))
                xtick_labels.append(size.upper())
            cur += INTER_GAP

        if not pos:
            ax.set_visible(False)
            continue

        ax.bar(pos, vals, width=BAR_W, yerr=errs, color=cols, hatch=hats,
               edgecolor="black", linewidth=0.8, capsize=3, alpha=0.9, zorder=2,
               error_kw={"ecolor": "black", "zorder": 4})
        ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)

        baseline_handles = []
        for bi, system in enumerate(["direct", "perspective"]):
            brow = summary[summary["system"] == system]
            if brow.empty:
                continue
            bval = float(brow.iloc[0][f"{stat}_mean"])
            bse = float(brow.iloc[0][f"{stat}_se"])
            bcolor = MAIN_PLOT_SETTINGS[system]["color"]
            bls = "--" if bi == 0 else "-."
            ax.axhline(bval, color=bcolor, linestyle=bls, linewidth=2.0, alpha=0.85, zorder=3)
            ax.axhspan(bval - bse, bval + bse, color=bcolor, alpha=0.12, zorder=0)
            baseline_handles.append(mlines.Line2D([], [], color=bcolor, linestyle=bls,
                                                   linewidth=2.0, label=_dn(system)))

        variant_handles = [
            mpatches.Patch(facecolor=colors_list[min(1, len(colors_list) - 1)], edgecolor="black",
                           hatch=HATCHES[hk], label=lbl)
            for _, colors_list, hk, lbl in variants
        ]

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=11)
        ax.set_ylabel(STAT_DISPLAY_NAMES[stat], fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(handles=variant_handles + baseline_handles, fontsize=9,
                  frameon=True, framealpha=0.95, edgecolor="gray", loc="best")
        sns.despine(ax=ax)

    fig.suptitle(f"HF Models — Female Target  ({STAT_DISPLAY_NAMES[stat]})",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(output_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = PROJECT_ROOT / "results" / "figures" / "hf_full_females"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = PROJECT_ROOT / "results" / "processed" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LLM annotations …")
    llm_ann = load_hf_full_annotations()

    print("Computing bootstrap summaries (this may take several minutes) …")
    summary = summarize_main_experiment(
        llm_annotations=llm_ann,
        target_label="female",
        models=ALL_HF_MODELS,
        experiment_group="generated",
        force=False,
    )

    csv_path = summary_dir / "toxicity_generated__female__hf_full.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Saved summary → {csv_path}")
    present = sorted(summary["system"].unique())
    print(f"  Systems found: {present}")

    for stat in ["mse", "bias", "var"]:
        print(f"\n── {stat.upper()} ──")

        plot_gemma(summary, stat, out_dir / f"gemma3_female_{stat}.pdf")
        print(f"  gemma3_female_{stat}.pdf")

        plot_ministral(summary, stat, out_dir / f"ministral3_female_{stat}.pdf")
        print(f"  ministral3_female_{stat}.pdf")

        plot_qwen35(summary, stat, out_dir / f"qwen35_female_{stat}.pdf")
        print(f"  qwen35_female_{stat}.pdf")

        plot_llama(summary, stat, out_dir / f"llama31_female_{stat}.pdf")
        print(f"  llama31_female_{stat}.pdf")

        plot_all_hf(summary, stat, out_dir / f"all_hf_female_{stat}.pdf")
        print(f"  all_hf_female_{stat}.pdf")

        plot_family_panel(summary, stat, out_dir / f"panel_families_female_{stat}.pdf")
        print(f"  panel_families_female_{stat}.pdf")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
