"""
Differential Perspective-Taking (DPT) analysis.

Computes Δ̂ = f̂(x,g₁) - f̂(x,g₂) vs Δ* = f*(x,g₁) - f*(x,g₂)
for toxicity-detection and DICES datasets.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.analysis.toxicity import load_llm_annotations
from src.paths import DATA_ROOT, RESULTS_ROOT
from src.datasets import (
    canonical_target_label,
    load_ground_truth as load_toxicity_ground_truth_raw,
    load_perspective_annotations,
)
from src.utils.models import normalize_model_name, display_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOXICITY_GROUPS = ["female", "male", "non-binary"]

TOXICITY_LLM_MODELS = [
    "gpt-5.1",
    "gpt-5.1_reasoning=low",
    "gpt-5.1_reasoning=medium",
    "gpt-5.1_reasoning=high",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-oss:120b",
    "gpt-oss:20b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "gemma3:1b",
    "gemma3:12b",
    "gemma3:27b",
    "qwen3:1.7b",
    "qwen3:8b",
    "qwen3:32b",
    "qwen3-r:1.7b",
    "qwen3-r:8b",
    "qwen3-r:32b",
    "qwen2.5:72b",
]

OUTPUT_ROOT = RESULTS_ROOT / "processed" / "differential_pt"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_toxicity_ground_truth(group1: str, group2: str) -> pd.DataFrame:
    """
    Return per-item ground-truth disagreement for two toxicity groups.

    Columns: comment_id, comment, f_star_g1, f_star_g2, delta_star
    """
    g1 = canonical_target_label(group1)
    g2 = canonical_target_label(group2)
    df = load_toxicity_ground_truth_raw()
    c1 = f"direct_annotation_{g1}"
    c2 = f"direct_annotation_{g2}"
    out = df[["comment_id", "comment"]].copy()
    out["f_star_g1"] = df[c1].astype(float)
    out["f_star_g2"] = df[c2].astype(float)
    out["delta_star"] = out["f_star_g1"] - out["f_star_g2"]
    return out


# ---------------------------------------------------------------------------
# LLM predictions  (uses the JSONL-based loader from analysis.toxicity)
# ---------------------------------------------------------------------------

def _load_llm_means(llm_annotations: pd.DataFrame, model_name: str, target_label: str) -> pd.DataFrame:
    """
    Return mean LLM prediction per comment for *model_name* and *target_label*.

    Columns: comment_id, f_hat
    """
    canonical = canonical_target_label(target_label)
    norm = normalize_model_name(model_name)
    sub = llm_annotations[
        (llm_annotations["model_name"] == norm)
        & (llm_annotations["target_group_canonical"] == canonical)
        & (llm_annotations["experiment_group"] == "main")
    ].copy()
    if sub.empty:
        raise ValueError(f"No LLM annotations for model={model_name}, target={target_label}")
    sub["value"] = pd.to_numeric(sub["parsed_percentage"], errors="coerce") / 100.0
    return (
        sub.groupby("comment_id", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "f_hat"})
    )


# ---------------------------------------------------------------------------
# Human PT predictions
# ---------------------------------------------------------------------------

def _load_human_pt_means(target_label: str, out_group: bool = False) -> pd.DataFrame:
    """
    Return mean human perspective-taking prediction per comment.

    Columns: comment_id, f_hat
    """
    df = load_perspective_annotations(target_label, out_group=out_group)
    return (
        df.groupby("comment_id", as_index=False)["percentage"]
        .mean()
        .rename(columns={"percentage": "f_hat"})
    )


# ---------------------------------------------------------------------------
# Compute differential for one experiment
# ---------------------------------------------------------------------------

def compute_toxicity_differential(
    group1: str,
    group2: str,
    estimator: str,
    llm_annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute per-item differential PT for one (group-pair, estimator) setting.

    *estimator*:  'human_in', 'human_out', or an LLM model name.

    Returns DataFrame with columns:
        comment_id, comment, f_star_g1, f_star_g2, delta_star,
        f_hat_g1, f_hat_g2, delta_hat
    """
    gt = load_toxicity_ground_truth(group1, group2)

    if estimator.startswith("human"):
        out_group = estimator == "human_out"
        p1 = _load_human_pt_means(group1, out_group).rename(columns={"f_hat": "f_hat_g1"})
        p2 = _load_human_pt_means(group2, out_group).rename(columns={"f_hat": "f_hat_g2"})
        preds = p1.merge(p2, on="comment_id", how="inner")
    else:
        if llm_annotations is None:
            llm_annotations = load_llm_annotations()
        p1 = _load_llm_means(llm_annotations, estimator, group1).rename(columns={"f_hat": "f_hat_g1"})
        p2 = _load_llm_means(llm_annotations, estimator, group2).rename(columns={"f_hat": "f_hat_g2"})
        preds = p1.merge(p2, on="comment_id", how="inner")

    preds["delta_hat"] = preds["f_hat_g1"] - preds["f_hat_g2"]
    merged = gt.merge(preds, on="comment_id", how="inner")
    return merged[
        ["comment_id", "comment", "f_star_g1", "f_star_g2", "delta_star",
         "f_hat_g1", "f_hat_g2", "delta_hat"]
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    delta_star: np.ndarray,
    delta_hat: np.ndarray,
    epsilon: float = 0.0,
) -> dict:
    """Pearson r, directional accuracy, attenuation, slope for epsilon-filtered items."""
    ds = np.asarray(delta_star, dtype=float)
    dh = np.asarray(delta_hat, dtype=float)
    mask = np.isfinite(ds) & np.isfinite(dh)
    ds, dh = ds[mask], dh[mask]

    if epsilon > 0:
        emask = np.abs(ds) > epsilon
        ds, dh = ds[emask], dh[emask]

    n = len(ds)
    if n < 3:
        return dict(
            epsilon=epsilon, n_items=n,
            pearson_r=np.nan, pearson_p=np.nan,
            directional_accuracy=np.nan,
            mean_delta_star=np.nan, mean_delta_hat=np.nan,
            std_delta_star=np.nan, std_delta_hat=np.nan,
            attenuation_ratio=np.nan, slope=np.nan,
        )

    r, p = pearsonr(ds, dh)

    # Directional accuracy (excluding near-zero ground truth)
    clear = np.abs(ds) > 0.01
    dir_acc = float((np.sign(ds[clear]) == np.sign(dh[clear])).mean()) if clear.sum() > 0 else np.nan

    std_s = float(np.std(ds, ddof=0))
    std_h = float(np.std(dh, ddof=0))
    attn = std_h / std_s if std_s > 0 else np.nan
    slope = float(np.polyfit(ds, dh, 1)[0]) if std_s > 0 else np.nan

    return dict(
        epsilon=epsilon, n_items=n,
        pearson_r=float(r), pearson_p=float(p),
        directional_accuracy=dir_acc,
        mean_delta_star=float(np.mean(ds)),
        mean_delta_hat=float(np.mean(dh)),
        std_delta_star=std_s, std_delta_hat=std_h,
        attenuation_ratio=attn, slope=slope,
    )


def compute_epsilon_sweep(
    delta_star: np.ndarray,
    delta_hat: np.ndarray,
    epsilons: Sequence[float] | None = None,
) -> pd.DataFrame:
    if epsilons is None:
        epsilons = [round(e, 2) for e in np.arange(0.0, 1.05, 0.1)]
    return pd.DataFrame([compute_metrics(delta_star, delta_hat, eps) for eps in epsilons])


# ---------------------------------------------------------------------------
# Bootstrap CI for Pearson r
# ---------------------------------------------------------------------------

def bootstrap_pearson_ci(
    delta_star: np.ndarray,
    delta_hat: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for Pearson r."""
    rng = np.random.default_rng(random_state)
    ds = np.asarray(delta_star, dtype=float)
    dh = np.asarray(delta_hat, dtype=float)
    mask = np.isfinite(ds) & np.isfinite(dh)
    ds, dh = ds[mask], dh[mask]
    n = len(ds)
    if n < 3:
        return (np.nan, np.nan)
    boot_rs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r, _ = pearsonr(ds[idx], dh[idx])
        boot_rs[b] = r
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_rs, 100 * alpha)),
            float(np.percentile(boot_rs, 100 * (1 - alpha))))


# ---------------------------------------------------------------------------
# Fisher z-test for comparing correlations
# ---------------------------------------------------------------------------

def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """
    Two-sided Fisher z-test for comparing two independent Pearson rs.

    Returns (z_statistic, two_sided_p_value).
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    from scipy.stats import norm
    p_two = 2 * norm.sf(abs(z))
    return (float(z), float(p_two))


# ---------------------------------------------------------------------------
# Batch runner:  all toxicity pairs × all available estimators
# ---------------------------------------------------------------------------

def _detect_available_models(llm_annotations: pd.DataFrame) -> list[str]:
    """
    Return model names that have annotations for all three toxicity groups
    in the 'main' experiment group.
    """
    main = llm_annotations[llm_annotations["experiment_group"] == "main"]
    groups_per_model = (
        main.groupby("model_name")["target_group_canonical"]
        .apply(lambda s: set(s.unique()))
    )
    available = [
        m for m, gs in groups_per_model.items()
        if set(canonical_target_label(g) for g in TOXICITY_GROUPS).issubset(gs)
    ]
    return sorted(available)


def run_all_toxicity(
    llm_annotations: pd.DataFrame | None = None,
    models: list[str] | None = None,
    include_human: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Compute DPT for all (group-pair, estimator) combinations.

    Saves per-item CSVs and returns a summary DataFrame.
    """
    if llm_annotations is None:
        llm_annotations = load_llm_annotations()
    if models is None:
        models = _detect_available_models(llm_annotations)
    if output_dir is None:
        output_dir = OUTPUT_ROOT / "toxicity"
    output_dir.mkdir(parents=True, exist_ok=True)

    estimators: list[str] = []
    if include_human:
        estimators.extend(["human_in", "human_out"])
    estimators.extend(models)

    summary_rows = []
    for g1, g2 in combinations(TOXICITY_GROUPS, 2):
        for est in estimators:
            # Check human PT file existence
            if est.startswith("human"):
                out_group = est == "human_out"
                try:
                    _load_human_pt_means(g1, out_group)
                    _load_human_pt_means(g2, out_group)
                except (FileNotFoundError, KeyError):
                    continue

            try:
                df = compute_toxicity_differential(g1, g2, est, llm_annotations)
            except (ValueError, FileNotFoundError) as exc:
                print(f"  skip {g1} vs {g2} | {est}: {exc}")
                continue

            # Save per-item CSV
            slug = f"toxicity_{g1}_vs_{g2}_{est}".replace(":", "-").replace("=", "-")
            df.to_csv(output_dir / f"{slug}.csv", index=False)

            ds = df["delta_star"].to_numpy()
            dh = df["delta_hat"].to_numpy()

            m0 = compute_metrics(ds, dh, 0.0)
            m1 = compute_metrics(ds, dh, 0.1)
            r_lo, r_hi = bootstrap_pearson_ci(ds, dh)

            summary_rows.append({
                "group1": g1,
                "group2": g2,
                "estimator": est,
                "display_name": display_model_name(est) if not est.startswith("human") else {
                    "human_in": "Human PT (In-Group)",
                    "human_out": "Human PT (Out-Group)",
                }.get(est, est),
                "n_items": m0["n_items"],
                "pearson_r": m0["pearson_r"],
                "pearson_p": m0["pearson_p"],
                "pearson_r_lo": r_lo,
                "pearson_r_hi": r_hi,
                "dir_acc": m0["directional_accuracy"],
                "attenuation": m0["attenuation_ratio"],
                "slope": m0["slope"],
                "std_delta_star": m0["std_delta_star"],
                "std_delta_hat": m0["std_delta_hat"],
                "n_items_eps01": m1["n_items"],
                "pearson_r_eps01": m1["pearson_r"],
                "dir_acc_eps01": m1["directional_accuracy"],
            })
            rho_str = f"{m0['pearson_r']:+.3f}" if np.isfinite(m0['pearson_r']) else " nan"
            da_str = f"{m0['directional_accuracy']:.1%}" if np.isfinite(m0['directional_accuracy']) else "nan"
            disp = summary_rows[-1]["display_name"]
            print(f"  ✓ {g1} vs {g2} | {disp:28s} | ρ={rho_str}  DA={da_str}  n={m0['n_items']}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# Statistical comparison helpers (for paper)
# ---------------------------------------------------------------------------

def compare_human_vs_llms(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each group pair, run Fisher z-tests comparing human_in ρ to every LLM ρ.
    """
    rows = []
    for (g1, g2), grp in summary.groupby(["group1", "group2"]):
        human_row = grp[grp["estimator"] == "human_in"]
        if human_row.empty:
            continue
        h = human_row.iloc[0]
        llm_rows = grp[~grp["estimator"].str.startswith("human")]
        for _, lr in llm_rows.iterrows():
            z, p2 = fisher_z_test(h["pearson_r"], h["n_items"], lr["pearson_r"], lr["n_items"])
            p_one = p2 / 2 if z > 0 else 1 - p2 / 2  # one-sided: human > LLM
            rows.append({
                "group1": g1, "group2": g2,
                "human_estimator": "human_in",
                "llm_estimator": lr["estimator"],
                "llm_display": lr["display_name"],
                "r_human": h["pearson_r"],
                "r_llm": lr["pearson_r"],
                "n_human": h["n_items"],
                "n_llm": lr["n_items"],
                "fisher_z": z,
                "p_two_sided": p2,
                "p_one_sided_human_gt": p_one,
                "significant_005": p_one < 0.05,
            })
    return pd.DataFrame(rows)
