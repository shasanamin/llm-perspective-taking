from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from _bootstrap import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis.toxicity import load_llm_annotations, summarize_main_experiment
from src.plotting.style import MAIN_PLOT_SETTINGS, RCPARAMS_TICKS
from src.datasets import get_ground_truth_frame
from src.utils.models import display_model_name


PAIR_SPECS = [
    {
        "experiment_group": "nonbinary_qwen_gpt_panel_20260316",
        "base_model": "qwen3:1.7b",
        "reason_model": "qwen3-r:1.7b",
        "pair_label": "Qwen3 1.7B",
        "family": "qwen",
    },
    {
        "experiment_group": "nonbinary_qwen_gpt_panel_20260316",
        "base_model": "qwen3:8b",
        "reason_model": "qwen3-r:8b",
        "pair_label": "Qwen3 8B",
        "family": "qwen",
    },
    {
        "experiment_group": "nonbinary_qwen_gpt_panel_20260316",
        "base_model": "qwen3:32b",
        "reason_model": "qwen3-r:32b",
        "pair_label": "Qwen3 32B",
        "family": "qwen",
    },
    {
        "experiment_group": "nonbinary_qwen_gpt_panel_20260316",
        "base_model": "gpt-5.4-nano",
        "reason_model": "gpt-5.4-nano_reasoning=high",
        "pair_label": "GPT-5.4 nano",
        "family": "gpt",
    },
    {
        "experiment_group": "nonbinary_openai_gpt54_tiers_20260318",
        "base_model": "gpt-5.4-mini",
        "reason_model": "gpt-5.4-mini_reasoning=high",
        "pair_label": "GPT-5.4 mini",
        "family": "gpt",
    },
    {
        "experiment_group": "nonbinary_openai_gpt54_tiers_20260318",
        "base_model": "gpt-5.4",
        "reason_model": "gpt-5.4_reasoning=high",
        "pair_label": "GPT-5.4",
        "family": "gpt",
    },
]

NANO_SWEEP_MODELS = [
    "gpt-5.4-nano",
    "gpt-5.4-nano_reasoning=low",
    "gpt-5.4-nano_reasoning=medium",
    "gpt-5.4-nano_reasoning=high",
    "gpt-5.4-nano_reasoning=xhigh",
]

GT_BUCKET_BOUNDS = [-1.0, 0.05, 0.15, 0.30, 1.0]
GT_BUCKET_LABELS = ["0-5%", "5-15%", "15-30%", "30%+"]

TRACE_PATTERNS = {
    "direct_target_gate": re.compile(
        r"not directly|doesn['’]t directly|not specifically|not aimed|not target|doesn['’]t target|"
        r"no direct|no slur|not overtly|not explicitly|not directly attack",
        re.IGNORECASE,
    ),
    "identity_harm_projection": re.compile(
        r"gender essential|binary framing|exclusionary|dehumaniz|stereotyp|marginali|fetishiz|objectify|"
        r"invalidat|erasure|transphob|sexist|anti-femin|gender-related|gender issue|disrespectful|"
        r"problematic elements",
        re.IGNORECASE,
    ),
    "generic_toxicity_rubric": re.compile(
        r"personal attack|insult|profan|obscene|threat|sarcastic|rude|dismissive|critici|toxic tone|"
        r"frustration|competence",
        re.IGNORECASE,
    ),
    "uncertainty_language": re.compile(
        r"\bmaybe\b|\bmight\b|\bcould\b|\bperhaps\b|not sure|unclear|hard to tell|\bi think\b",
        re.IGNORECASE,
    ),
}

QUALITATIVE_TARGETS = [
    ("Qwen3 32B", "worse"),
    ("Qwen3 1.7B", "worse"),
    ("GPT-5.4", "worse"),
    ("GPT-5.4 mini", "worse"),
    ("GPT-5.4 mini", "better"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the reasoning paradox on non-binary toxicity runs.")
    parser.add_argument(
        "--processed-dir",
        default=PROJECT_ROOT / "results" / "processed" / "reasoning_paradox",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default=PROJECT_ROOT / "results" / "figures" / "reasoning_paradox",
        help="Directory for figure outputs.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute bootstrap summaries.")
    return parser.parse_args()


def build_trace_text(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["trace_text"] = frame["reasoning_summary"].fillna("")
    frame.loc[frame["trace_text"] == "", "trace_text"] = frame.loc[frame["trace_text"] == "", "reasoning_trace"].fillna("")
    frame["trace_len"] = frame["trace_text"].str.len()
    frame["trace_type"] = frame["reasoning_trace_format"].fillna("")
    frame.loc[(frame["trace_type"] == "") & (frame["trace_text"] != ""), "trace_type"] = "plain_text_trace"
    frame.loc[frame["trace_type"] == "", "trace_type"] = "none"
    frame["readable_trace"] = frame["trace_type"].isin(["plain_text_trace", "summary_text"])
    return frame


def build_comment_level_frame(
    llm_annotations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    spec: dict[str, str],
) -> pd.DataFrame:
    base_rows = llm_annotations[
        (llm_annotations["experiment_group"] == spec["experiment_group"])
        & (llm_annotations["model_name"] == spec["base_model"])
    ].copy()
    reason_rows = llm_annotations[
        (llm_annotations["experiment_group"] == spec["experiment_group"])
        & (llm_annotations["model_name"] == spec["reason_model"])
    ].copy()

    base_mean = base_rows.groupby("comment_id", as_index=False)["value"].mean().rename(columns={"value": "base_mean"})
    reason_mean = reason_rows.groupby("comment_id", as_index=False)["value"].mean().rename(columns={"value": "reason_mean"})

    reason_rows = build_trace_text(reason_rows)
    reason_rows = reason_rows.merge(reason_mean, on="comment_id", how="left")
    reason_rows["parsed_value"] = pd.to_numeric(reason_rows["parsed_percentage"], errors="coerce") / 100.0
    reason_rows["distance_to_mean"] = (reason_rows["parsed_value"] - reason_rows["reason_mean"]).abs()

    representative_any = (
        reason_rows[reason_rows["trace_text"] != ""]
        .sort_values(["comment_id", "distance_to_mean", "trace_len"], ascending=[True, True, False])
        .drop_duplicates("comment_id")
        .rename(
            columns={
                "generation_index": "rep_generation_index",
                "parsed_percentage": "rep_parsed_percentage",
                "trace_text": "rep_trace_text",
                "trace_type": "rep_trace_type",
            }
        )
    )

    representative_readable = (
        reason_rows[reason_rows["readable_trace"] & (reason_rows["trace_text"] != "")]
        .sort_values(["comment_id", "distance_to_mean", "trace_len"], ascending=[True, True, False])
        .drop_duplicates("comment_id")
        .rename(
            columns={
                "generation_index": "readable_generation_index",
                "parsed_percentage": "readable_parsed_percentage",
                "trace_text": "readable_trace_text",
                "trace_type": "readable_trace_type",
            }
        )
    )

    paired = (
        ground_truth.merge(base_mean, on="comment_id", how="inner")
        .merge(reason_mean, on="comment_id", how="inner")
        .merge(
            representative_any[
                ["comment_id", "rep_generation_index", "rep_parsed_percentage", "rep_trace_text", "rep_trace_type"]
            ],
            on="comment_id",
            how="left",
        )
        .merge(
            representative_readable[
                [
                    "comment_id",
                    "readable_generation_index",
                    "readable_parsed_percentage",
                    "readable_trace_text",
                    "readable_trace_type",
                ]
            ],
            on="comment_id",
            how="left",
        )
    )

    paired["pair_label"] = spec["pair_label"]
    paired["experiment_group"] = spec["experiment_group"]
    paired["base_model"] = spec["base_model"]
    paired["reason_model"] = spec["reason_model"]
    paired["base_display"] = display_model_name(spec["base_model"])
    paired["reason_display"] = display_model_name(spec["reason_model"])
    paired["delta"] = paired["reason_mean"] - paired["base_mean"]
    paired["base_abs_error"] = (paired["base_mean"] - paired["ground_truth"]).abs()
    paired["reason_abs_error"] = (paired["reason_mean"] - paired["ground_truth"]).abs()
    paired["error_delta"] = paired["reason_abs_error"] - paired["base_abs_error"]
    paired["base_sq_error"] = (paired["base_mean"] - paired["ground_truth"]) ** 2
    paired["reason_sq_error"] = (paired["reason_mean"] - paired["ground_truth"]) ** 2
    paired["gt_bucket"] = pd.cut(
        paired["ground_truth"],
        bins=GT_BUCKET_BOUNDS,
        labels=GT_BUCKET_LABELS,
    )

    readable_text = paired["readable_trace_text"].fillna("")
    for category, pattern in TRACE_PATTERNS.items():
        paired[category] = readable_text.str.contains(pattern, regex=True)
    return paired


def build_pair_metric_summary(llm_annotations: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    frames = []
    grouped_models: dict[str, list[str]] = {}
    for spec in PAIR_SPECS:
        grouped_models.setdefault(spec["experiment_group"], [])
        grouped_models[spec["experiment_group"]].extend([spec["base_model"], spec["reason_model"]])

    summaries: dict[str, pd.DataFrame] = {}
    for experiment_group, models in grouped_models.items():
        unique_models = sorted(set(models))
        summaries[experiment_group] = summarize_main_experiment(
            llm_annotations=llm_annotations,
            target_label="non-binary",
            models=unique_models,
            experiment_group=experiment_group,
            force=force,
        )

    for spec in PAIR_SPECS:
        summary = summaries[spec["experiment_group"]]
        subset = summary[summary["n_annotators"] == 1].copy()
        base_row = subset[subset["system"] == spec["base_model"]].iloc[0]
        reason_row = subset[subset["system"] == spec["reason_model"]].iloc[0]
        frames.append(
            {
                "pair_label": spec["pair_label"],
                "base_model": spec["base_model"],
                "reason_model": spec["reason_model"],
                "base_display": display_model_name(spec["base_model"]),
                "reason_display": display_model_name(spec["reason_model"]),
                "base_mse": base_row["mse_mean"],
                "reason_mse": reason_row["mse_mean"],
                "delta_mse": reason_row["mse_mean"] - base_row["mse_mean"],
                "base_bias": base_row["bias_mean"],
                "reason_bias": reason_row["bias_mean"],
                "delta_bias": reason_row["bias_mean"] - base_row["bias_mean"],
                "delta_abs_bias": abs(reason_row["bias_mean"]) - abs(base_row["bias_mean"]),
                "base_var": base_row["var_mean"],
                "reason_var": reason_row["var_mean"],
                "delta_var": reason_row["var_mean"] - base_row["var_mean"],
            }
        )
    return pd.DataFrame(frames)


def build_trace_availability_summary(llm_annotations: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for spec in PAIR_SPECS:
        rows = llm_annotations[
            (llm_annotations["experiment_group"] == spec["experiment_group"])
            & (llm_annotations["model_name"] == spec["reason_model"])
        ].copy()
        rows = build_trace_text(rows)
        comments = rows.groupby("comment_id", as_index=False).agg(
            has_plain=("trace_type", lambda values: any(value == "plain_text_trace" for value in values)),
            has_summary=("trace_type", lambda values: any(value == "summary_text" for value in values)),
            has_encrypted=("trace_type", lambda values: any(value == "encrypted_content" for value in values)),
        )
        frames.append(
            {
                "pair_label": spec["pair_label"],
                "reason_model": spec["reason_model"],
                "row_count": len(rows),
                "comment_count": comments.shape[0],
                "row_plain_rate": (rows["trace_type"] == "plain_text_trace").mean(),
                "row_summary_rate": (rows["trace_type"] == "summary_text").mean(),
                "row_encrypted_rate": (rows["trace_type"] == "encrypted_content").mean(),
                "comment_plain_rate": comments["has_plain"].mean(),
                "comment_summary_rate": comments["has_summary"].mean(),
                "comment_encrypted_rate": comments["has_encrypted"].mean(),
                "comment_readable_rate": (comments["has_plain"] | comments["has_summary"]).mean(),
            }
        )
    return pd.DataFrame(frames)


def build_bucket_summary(comment_level: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comment_level.groupby(["pair_label", "gt_bucket"], observed=False)
        .agg(
            mean_delta=("delta", "mean"),
            mean_error_delta=("error_delta", "mean"),
            worse_fraction=("error_delta", lambda values: (values > 0).mean()),
            better_fraction=("error_delta", lambda values: (values < 0).mean()),
            n_comments=("comment_id", "count"),
        )
        .reset_index()
    )
    return summary


def build_pair_comment_summary(comment_level: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comment_level.groupby(["pair_label", "base_model", "reason_model"], as_index=False)
        .agg(
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            mean_error_delta=("error_delta", "mean"),
            worse_fraction=("error_delta", lambda values: (values > 0).mean()),
            better_fraction=("error_delta", lambda values: (values < 0).mean()),
            readable_comment_fraction=("readable_trace_text", lambda values: values.notna().mean()),
        )
    )
    return summary


def build_trace_category_summary(comment_level: pd.DataFrame) -> pd.DataFrame:
    frames = []
    readable = comment_level[comment_level["readable_trace_text"].notna()].copy()
    for pair_label, group in readable.groupby("pair_label"):
        for category in TRACE_PATTERNS:
            present = group[group[category]]
            absent = group[~group[category]]
            frames.append(
                {
                    "pair_label": pair_label,
                    "category": category,
                    "readable_comments": group.shape[0],
                    "trigger_count": present.shape[0],
                    "trigger_rate": present.shape[0] / group.shape[0] if group.shape[0] else math.nan,
                    "mean_delta_when_present": present["delta"].mean() if not present.empty else math.nan,
                    "mean_error_delta_when_present": present["error_delta"].mean() if not present.empty else math.nan,
                    "mean_delta_when_absent": absent["delta"].mean() if not absent.empty else math.nan,
                    "mean_error_delta_when_absent": absent["error_delta"].mean() if not absent.empty else math.nan,
                }
            )
    return pd.DataFrame(frames)


def build_qualitative_examples(comment_level: pd.DataFrame) -> pd.DataFrame:
    frames = []
    readable = comment_level[comment_level["readable_trace_text"].notna()].copy()
    for pair_label, direction in QUALITATIVE_TARGETS:
        group = readable[readable["pair_label"] == pair_label].copy()
        if group.empty:
            continue
        if direction == "worse":
            row = group.sort_values("error_delta", ascending=False).iloc[0]
        else:
            row = group.sort_values("error_delta", ascending=True).iloc[0]
        frames.append(
            {
                "pair_label": pair_label,
                "selection": direction,
                "comment_id": row["comment_id"],
                "ground_truth": row["ground_truth"],
                "base_mean": row["base_mean"],
                "reason_mean": row["reason_mean"],
                "delta": row["delta"],
                "error_delta": row["error_delta"],
                "base_model": row["base_model"],
                "reason_model": row["reason_model"],
                "readable_generation_index": row["readable_generation_index"],
                "readable_parsed_percentage": row["readable_parsed_percentage"],
                "comment": row["comment"],
                "trace_text": row["readable_trace_text"],
            }
        )
    return pd.DataFrame(frames)


def build_nano_effort_summary(llm_annotations: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    summary = summarize_main_experiment(
        llm_annotations=llm_annotations,
        target_label="non-binary",
        models=NANO_SWEEP_MODELS,
        experiment_group="nonbinary_qwen_gpt_panel_20260316",
        force=force,
    )
    summary = summary[summary["n_annotators"] == 1].copy()
    effort_order = {
        "gpt-5.4-nano": 0,
        "gpt-5.4-nano_reasoning=low": 1,
        "gpt-5.4-nano_reasoning=medium": 2,
        "gpt-5.4-nano_reasoning=high": 3,
        "gpt-5.4-nano_reasoning=xhigh": 4,
    }
    effort_label = {
        "gpt-5.4-nano": "none",
        "gpt-5.4-nano_reasoning=low": "low",
        "gpt-5.4-nano_reasoning=medium": "medium",
        "gpt-5.4-nano_reasoning=high": "high",
        "gpt-5.4-nano_reasoning=xhigh": "xhigh",
    }
    summary["effort_order"] = summary["system"].map(effort_order)
    summary["effort_label"] = summary["system"].map(effort_label)
    summary = summary[summary["effort_order"].notna()].copy()
    summary["abs_bias_mean"] = summary["bias_mean"].abs()
    return summary.sort_values("effort_order")


def plot_metric_deltas(pair_metrics: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", rc=RCPARAMS_TICKS)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=100)
    metrics = [
        ("delta_mse", "Delta MSE", "Reasoning - Base"),
        ("delta_abs_bias", "Delta |Bias|", "Reasoning - Base"),
        ("delta_var", "Delta Variance", "Reasoning - Base"),
    ]
    colors = ["#C44E52" if value > 0 else "#55A868" for value in pair_metrics["delta_mse"]]
    for axis, (column, title, ylabel) in zip(axes, metrics):
        values = pair_metrics[column]
        axis.bar(
            pair_metrics["pair_label"],
            values,
            color=colors if column == "delta_mse" else "#4C78A8",
            edgecolor="black",
            linewidth=0.8,
        )
        axis.axhline(0, color="black", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
    sns.despine()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_bucket_heatmap(bucket_summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", rc=RCPARAMS_TICKS)
    pivot = bucket_summary.pivot(index="pair_label", columns="gt_bucket", values="mean_error_delta").reindex(columns=GT_BUCKET_LABELS)
    figure, axis = plt.subplots(figsize=(8.5, 4.6), dpi=100)
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Mean abs-error delta"},
        ax=axis,
    )
    axis.set_xlabel("Ground-truth toxicity bucket")
    axis.set_ylabel("")
    axis.set_title("Where Reasoning Helps Or Hurts")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_trace_availability(trace_availability: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", rc=RCPARAMS_TICKS)
    figure, axis = plt.subplots(figsize=(8.8, 4.3), dpi=100)
    x_positions = range(len(trace_availability))
    plain = trace_availability["comment_plain_rate"]
    summary = trace_availability["comment_summary_rate"]
    encrypted = trace_availability["comment_encrypted_rate"]
    axis.bar(x_positions, plain, label="Plain trace", color="#4DBBD5", edgecolor="black", linewidth=0.8)
    axis.bar(x_positions, summary, bottom=plain, label="Readable summary", color="#00A087", edgecolor="black", linewidth=0.8)
    axis.bar(
        x_positions,
        encrypted,
        bottom=plain + summary,
        label="Encrypted only",
        color="#C44E52",
        edgecolor="black",
        linewidth=0.8,
        hatch="//",
    )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(trace_availability["pair_label"], rotation=25, ha="right")
    axis.set_ylabel("Comment-level availability")
    axis.set_ylim(0, 1.05)
    axis.set_title("Readable Reasoning Coverage")
    axis.legend(frameon=True)
    sns.despine()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_nano_effort_sweep(nano_summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", rc=RCPARAMS_TICKS)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=100)
    x = nano_summary["effort_label"]
    axes[0].plot(x, nano_summary["mse_mean"], marker="o", color=MAIN_PLOT_SETTINGS["gpt-5.4-nano_reasoning=high"]["color"])
    axes[0].set_title("GPT-5.4 nano effort sweep")
    axes[0].set_ylabel("MSE")
    axes[0].set_xlabel("Reasoning effort")
    axes[1].plot(x, nano_summary["abs_bias_mean"], marker="o", color="#4C78A8")
    axes[1].set_title("Absolute bias")
    axes[1].set_ylabel("|Bias|")
    axes[1].set_xlabel("Reasoning effort")
    for axis in axes:
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    figures_dir = Path(args.figures_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    llm_annotations = load_llm_annotations()
    if llm_annotations.empty:
        print("No LLM annotations found — skipping reasoning paradox analysis.")
        return
    llm_annotations = llm_annotations[llm_annotations["target_group_canonical"] == "non-binary"].copy()
    if llm_annotations.empty:
        print("No non-binary LLM annotations found — skipping reasoning paradox analysis.")
        return
    llm_annotations = llm_annotations.drop_duplicates(
        subset=["experiment_group", "model_name", "comment_id", "generation_index"],
        keep="last",
    ).copy()
    ground_truth = get_ground_truth_frame("non-binary")[["comment_id", "comment", "ground_truth"]].copy()

    pair_metrics = build_pair_metric_summary(llm_annotations, force=args.force)
    comment_level_frames = [build_comment_level_frame(llm_annotations, ground_truth, spec) for spec in PAIR_SPECS]
    comment_level = pd.concat(comment_level_frames, ignore_index=True)
    trace_availability = build_trace_availability_summary(llm_annotations)
    bucket_summary = build_bucket_summary(comment_level)
    pair_comment_summary = build_pair_comment_summary(comment_level)
    trace_category_summary = build_trace_category_summary(comment_level)
    qualitative_examples = build_qualitative_examples(comment_level)
    nano_summary = build_nano_effort_summary(llm_annotations, force=args.force)

    pair_metrics.to_csv(processed_dir / "pair_metrics.csv", index=False)
    comment_level.to_csv(processed_dir / "comment_level_deltas.csv", index=False)
    trace_availability.to_csv(processed_dir / "trace_availability.csv", index=False)
    bucket_summary.to_csv(processed_dir / "bucket_summary.csv", index=False)
    pair_comment_summary.to_csv(processed_dir / "pair_comment_summary.csv", index=False)
    trace_category_summary.to_csv(processed_dir / "trace_category_summary.csv", index=False)
    qualitative_examples.to_csv(processed_dir / "qualitative_examples.csv", index=False)
    nano_summary.to_csv(processed_dir / "nano_effort_summary.csv", index=False)

    plot_metric_deltas(pair_metrics, figures_dir / "pair_metric_deltas.pdf")
    plot_bucket_heatmap(bucket_summary, figures_dir / "error_delta_by_gt_bucket.pdf")
    plot_trace_availability(trace_availability, figures_dir / "trace_availability.pdf")
    plot_nano_effort_sweep(nano_summary, figures_dir / "nano_effort_sweep.pdf")

    print(f"Saved processed analysis to {processed_dir}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
