from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.bootstrap import summarize_grouped_annotations
from src.paths import DATA_ROOT, RESULTS_ROOT
from src.datasets import build_target_filters, compute_ground_truth
from src.utils.jsonl import iter_jsonl
from src.utils.models import normalize_model_name


DICES_ANNOTATIONS_ROOT = DATA_ROOT / "llm_annotations" / "dices"
DICES_PROCESSED_ROOT = RESULTS_ROOT / "processed" / "dices"


def load_llm_annotations() -> pd.DataFrame:
    records = []
    for path in sorted(DICES_ANNOTATIONS_ROOT.rglob("*.jsonl")):
        for record in iter_jsonl(path) or []:
            if record.get("dataset") != "dices":
                continue
            if str(record.get("run_name", "")).endswith("_NA"):
                continue
            record = dict(record)
            record["source_file"] = str(path)
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["model_name"] = df["model_name"].map(normalize_model_name)
    df["value"] = pd.to_numeric(df["parsed_percentage"], errors="coerce") / 100.0
    return df


def filter_llm_annotations(
    llm_annotations: pd.DataFrame,
    target_labels: list[str] | None = None,
    models: list[str] | None = None,
    experiment_group: str | None = None,
) -> pd.DataFrame:
    if llm_annotations.empty:
        return llm_annotations
    frame = llm_annotations.copy()
    if target_labels is not None:
        frame = frame[frame["target_group"].isin(target_labels)]
    if models is not None:
        normalized_models = {normalize_model_name(model_name) for model_name in models}
        frame = frame[frame["model_name"].isin(normalized_models)]
    if experiment_group is not None:
        frame = frame[frame["experiment_group"] == experiment_group]
    return frame


def attach_ground_truth(annotations: pd.DataFrame) -> pd.DataFrame:
    merged_frames = []
    for target_label, group_frame in annotations.groupby("target_group", dropna=False):
        ground_truth = compute_ground_truth(target_label)
        merged = group_frame.merge(ground_truth, on="comment_id", how="inner")
        merged_frames.append(merged)
    if not merged_frames:
        return pd.DataFrame()
    return pd.concat(merged_frames, ignore_index=True)


def summarize_llm_annotations(
    annotations: pd.DataFrame,
    group_columns: list[str],
    cache_prefix: str,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir = DICES_PROCESSED_ROOT / "bootstrap"
    merged = attach_ground_truth(annotations)
    if merged.empty:
        metric_columns = [
            *group_columns,
            "n_annotators",
            "n_items",
            "bias_mean",
            "bias_se",
            "var_mean",
            "var_se",
            "mse_mean",
            "mse_se",
            "L",
        ]
        return pd.DataFrame(columns=metric_columns)
    summary = summarize_grouped_annotations(
        annotations=merged,
        group_columns=group_columns,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        force=force,
    )
    if "target_group" in summary.columns:
        summary["L"] = summary["target_group"].map(lambda target: len(build_target_filters(target)))
    return summary
