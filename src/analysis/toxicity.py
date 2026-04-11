from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.bootstrap import summarize_grouped_annotations
from src.paths import DATA_ROOT, RESULTS_ROOT
from src.datasets import (
    canonical_target_label,
    get_ground_truth_frame,
    load_direct_annotations,
    load_perspective_annotations,
)
from src.utils.common import slugify
from src.utils.jsonl import iter_jsonl
from src.utils.models import normalize_model_name


TOXICITY_ANNOTATIONS_ROOT = DATA_ROOT / "llm_annotations" / "toxicity_detection"
TOXICITY_PROCESSED_ROOT = RESULTS_ROOT / "processed" / "toxicity_detection"


def load_llm_annotations() -> pd.DataFrame:
    records = []
    for path in sorted(TOXICITY_ANNOTATIONS_ROOT.rglob("*.jsonl")):
        try:
            for record in iter_jsonl(path) or []:
                if record.get("dataset") != "toxicity_detection":
                    continue
                record = dict(record)
                record["source_file"] = str(path)
                records.append(record)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["model_name"] = df["model_name"].map(normalize_model_name)
    df["target_group_canonical"] = df["target_group_canonical"].map(canonical_target_label)
    df["value"] = pd.to_numeric(df["parsed_percentage"], errors="coerce") / 100.0
    return df


def load_human_annotations(target_label: str) -> pd.DataFrame:
    canonical = canonical_target_label(target_label)
    frames = []

    direct = load_direct_annotations(canonical)
    direct["system"] = "direct"
    frames.append(direct)

    perspective = load_perspective_annotations(canonical, out_group=False)
    perspective["system"] = "perspective"
    frames.append(perspective)

    if canonical != "non-binary":
        perspective_out = load_perspective_annotations(canonical, out_group=True)
        perspective_out["system"] = "perspective_out"
        frames.append(perspective_out)

    ground_truth = get_ground_truth_frame(canonical)[["comment_id", "ground_truth"]]
    merged_frames = []
    for frame in frames:
        merged = frame.merge(ground_truth, on="comment_id", how="inner")
        merged = merged.rename(columns={"percentage": "value"})
        merged["target_group_canonical"] = canonical
        merged_frames.append(merged)
    return pd.concat(merged_frames, ignore_index=True)


def filter_llm_annotations(
    llm_annotations: pd.DataFrame,
    target_label: str,
    experiment_group: str | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    canonical = canonical_target_label(target_label)
    frame = llm_annotations[llm_annotations["target_group_canonical"] == canonical].copy()
    if experiment_group is not None:
        frame = frame[frame["experiment_group"] == experiment_group]
    if models is not None:
        normalized_models = {normalize_model_name(model_name) for model_name in models}
        frame = frame[frame["model_name"].isin(normalized_models)]
    return frame


def build_main_annotations(
    llm_annotations: pd.DataFrame,
    target_label: str,
    models: list[str],
    experiment_group: str = "main",
) -> pd.DataFrame:
    canonical = canonical_target_label(target_label)
    ground_truth = get_ground_truth_frame(canonical)[["comment_id", "ground_truth"]]

    llm_frame = filter_llm_annotations(
        llm_annotations,
        target_label=canonical,
        experiment_group=experiment_group,
        models=models,
    )
    llm_frame = llm_frame.rename(columns={"model_name": "system"})
    llm_frame = llm_frame[["system", "comment_id", "value"]].merge(
        ground_truth,
        on="comment_id",
        how="inner",
    )
    llm_frame["target_group_canonical"] = canonical

    human_frame = load_human_annotations(canonical)
    return pd.concat([llm_frame, human_frame], ignore_index=True)


def summarize_main_experiment(
    llm_annotations: pd.DataFrame,
    target_label: str,
    models: list[str],
    experiment_group: str = "main",
    force: bool = False,
) -> pd.DataFrame:
    annotations = build_main_annotations(
        llm_annotations,
        target_label,
        models,
        experiment_group=experiment_group,
    )
    cache_dir = TOXICITY_PROCESSED_ROOT / "bootstrap"
    cache_prefix = f"main__{slugify(experiment_group)}__{canonical_target_label(target_label)}"
    summary = summarize_grouped_annotations(
        annotations=annotations,
        group_columns=["system"],
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        force=force,
    )
    summary["target_group_canonical"] = canonical_target_label(target_label)
    summary["experiment_group"] = experiment_group
    return summary


def summarize_generic_llm_annotations(
    annotations: pd.DataFrame,
    group_columns: list[str],
    cache_prefix: str,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir = TOXICITY_PROCESSED_ROOT / "bootstrap"
    return summarize_grouped_annotations(
        annotations=annotations,
        group_columns=group_columns,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        force=force,
    )
