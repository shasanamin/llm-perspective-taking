"""
Generate paper-quality figures for differential perspective-taking results.

Usage:
    python scripts/plot_differential_pt.py
    python scripts/plot_differential_pt.py --output-dir results/figures/paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.differential import (
    OUTPUT_ROOT,
    bootstrap_pearson_ci,
    compute_metrics,
)
from src.plotting.style import (
    NATURE_COLORS,
    RCPARAMS_TICKS,
    save_figure,
)
from src.utils.models import display_model_name


# ── Style helpers ─────────────────────────────────────────────────────────

_HUMAN_COLOR = "#F2C500"
_HUMAN_OUT_COLOR = "#91D1C2"

# Consistent ordering & color for scatter panel
_PANEL_ESTIMATORS = [
    ("human_in",  "Human PT\n(In-Group)",  _HUMAN_COLOR, "o"),
    ("human_out", "Human PT\n(Out-Group)", _HUMAN_OUT_COLOR, "s"),
    ("gpt-5.1",               None,   "#E64B35", "^"),
    ("gpt-5.1_reasoning=medium", None, "#DC0000", "v"),
    ("qwen3-r:32b",           None,   "#3C5488", "D"),
    ("deepseek-r1:14b",       None,   "#4DBBD5", "P"),
    ("gemma3:27b",            None,   "#00A087", "X"),
]

PAIR_LABELS = {
    ("female", "male"):        "Female vs Male",
    ("female", "non-binary"):  "Female vs Non-binary",
    ("male", "non-binary"):    "Male vs Non-binary",
}


def _set_style() -> None:
    sns.set_theme(style="ticks", rc=RCPARAMS_TICKS)


def _estimator_label(est: str) -> str:
    if est == "human_in":
        return "Human PT (In)"
    if est == "human_out":
        return "Human PT (Out)"
    return display_model_name(est)


# ── Figure 1: Main scatter panel  (rows = pairs, cols = estimators) ───────

def plot_scatter_panel(
    summary: pd.DataFrame,
    data_dir: Path,
    output_path: Path,
) -> None:
    """3 rows (pairs) × N columns (selected estimators) scatter of Δ̂ vs Δ*."""
    _set_style()

    pairs = [("female", "male"), ("female", "non-binary"), ("male", "non-binary")]

    # Filter to estimators that exist in the summary
    available = set(summary["estimator"].unique())
    panel_ests = [
        (est, label, color, marker)
        for est, label, color, marker in _PANEL_ESTIMATORS
        if est in available
    ]

    n_rows = len(pairs)
    n_cols = len(panel_ests)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.7 * n_cols, 2.5 * n_rows),
        sharex=False, sharey=False,
        squeeze=False,
    )

    for row_idx, (g1, g2) in enumerate(pairs):
        for col_idx, (est, label_override, color, marker) in enumerate(panel_ests):
            ax = axes[row_idx, col_idx]
            slug = f"toxicity_{g1}_vs_{g2}_{est}".replace(":", "-").replace("=", "-")
            csv_path = data_dir / f"{slug}.csv"
            if not csv_path.exists():
                ax.set_visible(False)
                continue

            df = pd.read_csv(csv_path)
            ds = df["delta_star"].to_numpy()
            dh = df["delta_hat"].to_numpy()

            m = compute_metrics(ds, dh, 0.0)
            r_val = m["pearson_r"]
            da_val = m["directional_accuracy"]

            ax.scatter(ds, dh, s=18, alpha=0.55, color=color, marker=marker,
                       edgecolors="white", linewidths=0.3)

            # OLS fit line
            if np.isfinite(m["slope"]):
                x_range = np.linspace(ds.min(), ds.max(), 50)
                coeffs = np.polyfit(ds, dh, 1)
                ax.plot(x_range, np.polyval(coeffs, x_range), color="black",
                        linewidth=1.0, linestyle="--", alpha=0.6)

            # Diagonal reference
            lim_min = min(ds.min(), dh.min()) - 0.02
            lim_max = max(ds.max(), dh.max()) + 0.02
            ax.plot([lim_min, lim_max], [lim_min, lim_max],
                    color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
            ax.axhline(0, color="gray", linewidth=0.4, alpha=0.3)
            ax.axvline(0, color="gray", linewidth=0.4, alpha=0.3)

            # Annotation
            r_str = f"$\\rho$={r_val:.2f}" if np.isfinite(r_val) else "$\\rho$=n/a"
            da_str = f"DA={da_val:.0%}" if np.isfinite(da_val) else ""
            ax.text(0.04, 0.96, f"{r_str}\n{da_str}", transform=ax.transAxes,
                    fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            # Labels
            if row_idx == n_rows - 1:
                ax.set_xlabel("$\\Delta^*$", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("$\\hat{\\Delta}$", fontsize=10)
                pair_label = PAIR_LABELS.get((g1, g2), f"{g1} vs {g2}")
                ax.annotate(
                    pair_label, xy=(-0.45, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", rotation=90,
                    ha="center", va="center",
                )

            if row_idx == 0:
                label_text = label_override if label_override else _estimator_label(est)
                ax.set_title(label_text, fontsize=9, fontweight="bold")

            ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=0.3, h_pad=0.5)
    fig.subplots_adjust(left=0.10)
    save_figure(output_path)
    print(f"  → {output_path}")


# ── Figure 2: Bar chart  ρ by model, coloured by group pair ──────────────

def plot_correlation_bars(
    summary: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Grouped bar chart of Pearson ρ for key models across group pairs."""
    _set_style()

    pairs = [("female", "male"), ("male", "non-binary")]
    pair_colors = {
        ("female", "male"): NATURE_COLORS[3],       # red
        ("male", "non-binary"): NATURE_COLORS[2],    # teal
    }

    # Pick models: all humans + top LLMs by mean ρ across the two pairs
    llm_rows = summary[
        ~summary["estimator"].str.startswith("human")
        & summary.apply(lambda r: (r["group1"], r["group2"]) in pairs, axis=1)
    ]
    mean_r = llm_rows.groupby("estimator")["pearson_r"].mean().sort_values(ascending=False)
    top_models = mean_r.head(top_n).index.tolist()

    estimator_order = ["human_in", "human_out"] + top_models
    estimator_order = [e for e in estimator_order if e in summary["estimator"].values]

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)

    bar_width = 0.35
    x = np.arange(len(estimator_order))

    for pair_idx, (g1, g2) in enumerate(pairs):
        pair_data = summary[(summary["group1"] == g1) & (summary["group2"] == g2)]
        vals = []
        ci_lo, ci_hi = [], []
        for est in estimator_order:
            row = pair_data[pair_data["estimator"] == est]
            if row.empty:
                vals.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
            else:
                r = row.iloc[0]
                vals.append(r["pearson_r"])
                ci_lo.append(r.get("pearson_r_lo", np.nan))
                ci_hi.append(r.get("pearson_r_hi", np.nan))

        vals = np.array(vals, dtype=float)
        errs_lo = vals - np.array(ci_lo, dtype=float)
        errs_hi = np.array(ci_hi, dtype=float) - vals
        errs = np.array([np.where(np.isfinite(errs_lo), errs_lo, 0),
                         np.where(np.isfinite(errs_hi), errs_hi, 0)])

        offset = (pair_idx - 0.5) * bar_width
        ax.bar(
            x + offset, vals,
            width=bar_width, color=pair_colors[(g1, g2)],
            edgecolor="black", linewidth=0.5, alpha=0.85,
            label=PAIR_LABELS.get((g1, g2), f"{g1} vs {g2}"),
            yerr=errs, capsize=2.5,
        )

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="-")

    x_labels = [_estimator_label(e) for e in estimator_order]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson $\\rho$", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    sns.despine()
    save_figure(output_path)
    print(f"  → {output_path}")


# ── Figure 3: Scaling by model family ────────────────────────────────────

def plot_scaling(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Line plot: mean ρ across pairs vs model size (log-scale B params) for each family."""
    _set_style()

    # Map model keys to actual parameter counts in billions
    MODEL_PARAMS_B = {
        "deepseek-r1:1.5b": 1.5, "deepseek-r1:7b": 7, "deepseek-r1:14b": 14, "deepseek-r1:32b": 32,
        "gemma3:1b": 1, "gemma3:12b": 12, "gemma3:27b": 27,
        "qwen3:1.7b": 1.7, "qwen3:8b": 8, "qwen3:32b": 32,
        "qwen3-r:1.7b": 1.7, "qwen3-r:8b": 8, "qwen3-r:32b": 32,
    }

    families = {
        "DeepSeek-R1": ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b"],
        "Gemma3":      ["gemma3:1b", "gemma3:12b", "gemma3:27b"],
        "Qwen3":       ["qwen3:1.7b", "qwen3:8b", "qwen3:32b"],
        "Qwen3-R":     ["qwen3-r:1.7b", "qwen3-r:8b", "qwen3-r:32b"],
    }
    fam_colors = {
        "DeepSeek-R1": NATURE_COLORS[1],
        "Gemma3":      NATURE_COLORS[2],
        "Qwen3":       NATURE_COLORS[4],
        "Qwen3-R":     NATURE_COLORS[3],
    }
    fam_markers = {
        "DeepSeek-R1": "D",
        "Gemma3":      "s",
        "Qwen3":       "o",
        "Qwen3-R":     "^",
    }

    # Only use the two informative pairs
    informative_pairs = {("female", "male"), ("male", "non-binary")}
    sub = summary[summary.apply(lambda r: (r["group1"], r["group2"]) in informative_pairs, axis=1)]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)

    for fam_name, model_list in families.items():
        params = []
        mean_rs = []
        for model in model_list:
            rows = sub[sub["estimator"] == model]
            if rows.empty:
                continue
            params.append(MODEL_PARAMS_B[model])
            mean_rs.append(rows["pearson_r"].mean())
        if params:
            ax.plot(params, mean_rs, marker=fam_markers.get(fam_name, "o"),
                    linewidth=2, markersize=7,
                    color=fam_colors.get(fam_name, "gray"), label=fam_name)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 14, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Parameters (B)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Pearson $\\rho$", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    sns.despine()
    save_figure(output_path)
    print(f"  → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot differential PT figures.")
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "results" / "figures" / "paper",
    )
    parser.add_argument(
        "--data-dir",
        default=OUTPUT_ROOT / "toxicity",
        help="Dir with per-item CSVs from analyze_differential_pt.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "summary.csv"
    if not summary_path.exists():
        print(f"ERROR: summary not found at {summary_path}.")
        print("Run  python scripts/analyze_differential_pt.py  first.")
        return

    summary = pd.read_csv(summary_path)
    print(f"Loaded summary: {len(summary)} rows\n")

    print("Generating scatter panel …")
    plot_scatter_panel(summary, data_dir, output_dir / "dpt_scatter_panel.pdf")

    print("Generating correlation bar chart …")
    plot_correlation_bars(summary, output_dir / "dpt_correlation_bars.pdf")

    print("Generating scaling plot …")
    plot_scaling(summary, output_dir / "dpt_scaling.pdf")

    print("\nDone — all figures saved to", output_dir)


if __name__ == "__main__":
    main()
