"""Dataset loaders for the toxicity detection and DICES-350 experiments."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.paths import DATA_ROOT


# ═══════════════════════════════════════════════════════════════════════════════
# Toxicity detection dataset
# ═══════════════════════════════════════════════════════════════════════════════

TOXICITY_ROOT = DATA_ROOT / "toxicity_detection"
HUMAN_ANNOTATIONS_ROOT = TOXICITY_ROOT / "human_annotations"

TARGET_ALIASES = {
    "female": "female",
    "females": "female",
    "male": "male",
    "males": "male",
    "non-binary": "non-binary",
    "nonbinary": "non-binary",
    "non-binary people": "non-binary",
}

PROMPT_LABELS = {
    "female": "female",
    "male": "male",
    "non-binary": "non-binary people",
}

DISPLAY_LABELS = {
    "female": "Female",
    "male": "Male",
    "non-binary": "Non-binary",
}


def canonical_target_label(target_label: str) -> str:
    normalized = target_label.strip().lower()
    if normalized not in TARGET_ALIASES:
        raise ValueError(f"Unsupported toxicity target '{target_label}'.")
    return TARGET_ALIASES[normalized]


def prompt_target_label(target_label: str) -> str:
    return PROMPT_LABELS[canonical_target_label(target_label)]


def display_target_label(target_label: str) -> str:
    return DISPLAY_LABELS[canonical_target_label(target_label)]


def load_ground_truth() -> pd.DataFrame:
    path = HUMAN_ANNOTATIONS_ROOT / "direct_annotations.csv"
    return pd.read_csv(path)


def load_comments() -> pd.DataFrame:
    df = load_ground_truth()
    return df[["comment_id", "comment"]].copy()


def load_direct_annotations(target_label: str) -> pd.DataFrame:
    canonical = canonical_target_label(target_label)
    path = HUMAN_ANNOTATIONS_ROOT / "pilot_annotations.csv"
    df = pd.read_csv(path)
    gender_label = {
        "female": "Female",
        "male": "Male",
        "non-binary": "Non-binary",
    }[canonical]
    df = df[df["gender"] == gender_label].copy()
    df = df.rename(columns={"comment": "comment_id", "answer": "percentage"})
    df["percentage"] = (df["percentage"].astype(float) == 1.0).astype(float)
    return df[["comment_id", "percentage"]]


def load_perspective_annotations(target_label: str, out_group: bool = False) -> pd.DataFrame:
    canonical = canonical_target_label(target_label)
    suffix = "_out" if out_group else ""
    path = HUMAN_ANNOTATIONS_ROOT / ("out_group" if out_group else "in_group") / f"{canonical}{suffix}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"comment": "comment_id", "answer": "percentage"})
    df["percentage"] = df["percentage"].astype(float)
    return df[["comment_id", "percentage"]]


def get_ground_truth_frame(target_label: str) -> pd.DataFrame:
    canonical = canonical_target_label(target_label)
    gt_column = f"direct_annotation_{canonical}"
    df = load_ground_truth()[["comment_id", "comment", gt_column]].copy()
    return df.rename(columns={gt_column: "ground_truth"})


def build_example_payload(target_label: str, example_indices: list[int]) -> dict[str, str | int]:
    canonical = canonical_target_label(target_label)
    gt_column = f"direct_annotation_{canonical}"
    df = load_ground_truth()

    low_idx, medium_idx, high_idx = example_indices[:3]
    return {
        "example_low_comment": df.iloc[low_idx]["comment"],
        "example_low_percentage": round(float(df.iloc[low_idx][gt_column]) * 100),
        "example_medium_comment": df.iloc[medium_idx]["comment"],
        "example_medium_percentage": round(float(df.iloc[medium_idx][gt_column]) * 100),
        "example_high_comment": df.iloc[high_idx]["comment"],
        "example_high_percentage": round(float(df.iloc[high_idx][gt_column]) * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DICES-350 dataset
# ═══════════════════════════════════════════════════════════════════════════════

DICES_ROOT = DATA_ROOT / "dices" / "original"

GOOD_PATH_TARGETS = [
    {"rater_education": "College degree or higher"},
    {"rater_race": "Black/African American", "rater_education": "College degree or higher"},
    {"rater_race": "LatinX, Latino, Hispanic or Spanish Origin", "rater_education": "College degree or higher"},
    {"rater_gender": "Woman", "rater_race": "Black/African American", "rater_education": "College degree or higher"},
    {"rater_gender": "Woman", "rater_race": "LatinX, Latino, Hispanic or Spanish Origin", "rater_education": "College degree or higher"},
]

BAD_PATH_TARGETS = [
    {"rater_age": "gen z"},
    {"rater_gender": "Man", "rater_age": "gen z"},
    {"rater_gender": "Man", "rater_age": "gen z", "rater_education": "High school or below"},
]


def load_raw_annotations() -> pd.DataFrame:
    return pd.read_csv(DICES_ROOT / "dices350_raw.csv")


def load_items() -> pd.DataFrame:
    return pd.read_csv(DICES_ROOT / "dices_comments.csv")


def build_target_label(combo: dict[str, str]) -> str:
    target = combo["rater_gender"].replace("an", "en") if "rater_gender" in combo else "people"
    if "rater_age" in combo:
        target = f"{combo['rater_age']} {target}"
    if "rater_race" in combo:
        race = combo["rater_race"]
        if race == "Asian/Asian subcontinent":
            race = "Asian"
        elif race == "Black/African American":
            race = "Black"
        elif race == "LatinX, Latino, Hispanic or Spanish Origin":
            race = "Latino"
        target = f"{race} {target}"
    if "rater_education" in combo:
        target += f" with education {combo['rater_education']}"
    return target.lower()


def get_paper_target_labels() -> list[str]:
    return [build_target_label(target) for target in GOOD_PATH_TARGETS + BAD_PATH_TARGETS]


def build_target_filters(target_label: str) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    lowered = target_label.lower()

    if "women" in lowered:
        filters.append(("rater_gender", "Woman"))
    elif "men" in lowered:
        filters.append(("rater_gender", "Man"))

    if "asian" in lowered:
        filters.append(("rater_race", "Asian/Asian subcontinent"))
    elif "white" in lowered:
        filters.append(("rater_race", "White"))
    elif "black" in lowered:
        filters.append(("rater_race", "Black/African American"))
    elif "latino" in lowered:
        filters.append(("rater_race", "LatinX, Latino, Hispanic or Spanish Origin"))
    elif "multiracial" in lowered:
        filters.append(("rater_race", "Multiracial"))

    if "millenial" in lowered:
        filters.append(("rater_age", "millenial"))
    elif "gen x+" in lowered:
        filters.append(("rater_age", "gen x+"))
    elif "gen z" in lowered:
        filters.append(("rater_age", "gen z"))

    if "college degree or higher" in lowered:
        filters.append(("rater_education", "College degree or higher"))
    elif "high school or below" in lowered:
        filters.append(("rater_education", "High school or below"))

    return filters


def compute_ground_truth(target_label: str) -> pd.DataFrame:
    raw = load_raw_annotations()
    filters = build_target_filters(target_label)
    if not filters:
        raise ValueError(f"Could not infer DICES filters from '{target_label}'.")

    mask = pd.Series(True, index=raw.index)
    for column, value in filters:
        mask &= raw[column] == value

    filtered = raw[mask].copy()
    item_lookup = raw.drop_duplicates("item_id")[["item_id"]].reset_index(drop=True)
    item_lookup["comment_id"] = item_lookup.index
    rows = []
    for _, item_row in item_lookup.iterrows():
        item_rows = filtered[filtered["item_id"] == item_row["item_id"]]
        rows.append(
            {
                "comment_id": int(item_row["comment_id"]),
                "ground_truth": (item_rows["Q_overall"] == "Yes").mean(),
            }
        )

    return pd.DataFrame(rows)


def feature_target_labels(feature_name: str) -> list[str]:
    raw = load_raw_annotations()
    labels = []
    for value in sorted(raw[feature_name].dropna().unique()):
        labels.append(build_target_label({feature_name: value}))
    return labels


def path_target_labels(paths: Iterable[dict[str, str]]) -> list[str]:
    return [build_target_label(path) for path in paths]
