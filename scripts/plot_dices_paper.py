from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.analysis.dices import filter_llm_annotations, load_llm_annotations, summarize_llm_annotations
from src.plotting.dices import format_dices_label, plot_stat_at_one, plot_stat_line
from src.datasets import (
    BAD_PATH_TARGETS,
    GOOD_PATH_TARGETS,
    build_target_label,
    feature_target_labels,
    get_paper_target_labels,
)


MODEL_FILE_NAMES = {
    "gpt-oss:20b": "gpt-oss:latest",
    "gpt-oss:120b": "gpt-oss:120b",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1_reasoning=high": "gpt-5.1_reasoning=high",
    "gemma3:1b": "gemma3:1b",
    "gemma3:12b": "gemma3:12b",
    "gemma3:27b": "gemma3:27b",
    "deepseek-r1:1.5b": "deepseek-r1:1.5b",
    "deepseek-r1:7b": "deepseek-r1:7b",
    "deepseek-r1:32b": "deepseek-r1:32b",
    "qwen3:1.7b": "qwen3:1.7b",
    "qwen3:8b": "qwen3:8b",
    "qwen3:32b": "qwen3:32b",
    "qwen3-r:1.7b": "qwen3-r:1.7b",
    "qwen3-r:8b": "qwen3-r:8b",
    "qwen3-r:32b": "qwen3-r:32b",
}


ALL_MODELS = [
    "gemma3:1b", "gemma3:12b", "gemma3:27b",
    "gpt-oss:20b", "gpt-oss:120b", "gpt-5.1", "gpt-5.1_reasoning=high",
    "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:32b",
    "qwen3:1.7b", "qwen3:8b", "qwen3:32b",
    "qwen3-r:1.7b", "qwen3-r:8b", "qwen3-r:32b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the DICES paper figures.")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "results" / "figures" / "paper" / "DICE", help="Directory for output PDFs.")
    parser.add_argument("--force", action="store_true", help="Recompute cached bootstrap files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_annotations = load_llm_annotations()
    if llm_annotations.empty:
        print("No DICES LLM annotations found — skipping DICES figures.")
        print("To generate them, run: python scripts/run_dices.py")
        return

    for feature, feature_slug in [("rater_race", "race"), ("rater_age", "age")]:
        target_labels = feature_target_labels(feature)
        for model in ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1"]:
            filtered = filter_llm_annotations(
                llm_annotations=llm_annotations,
                target_labels=target_labels,
                models=[model],
                experiment_group="main",
            )
            summary = summarize_llm_annotations(
                annotations=filtered,
                group_columns=["target_group"],
                cache_prefix=f"dices__{feature_slug}__{model.replace(':', '_')}",
                force=args.force,
            )
            summary_map = {
                label: summary[summary["target_group"] == label]
                for label in target_labels
                if label in summary["target_group"].unique()
            }
            plot_stat_line(
                summary_map=summary_map,
                stat="mse",
                output_path=output_dir / f"mse_{feature_slug}_{MODEL_FILE_NAMES[model]}.pdf",
            )

    good_path_labels = [build_target_label(path) for path in GOOD_PATH_TARGETS]
    bad_path_labels = [build_target_label(path) for path in BAD_PATH_TARGETS]

    for path_name, labels in [("good_path", good_path_labels), ("bad_path", bad_path_labels)]:
        filtered = filter_llm_annotations(
            llm_annotations=llm_annotations,
            target_labels=labels,
            models=ALL_MODELS,
            experiment_group="main",
        )
        summary = summarize_llm_annotations(
            annotations=filtered,
            group_columns=["model_name", "target_group"],
            cache_prefix=f"dices__{path_name}",
            force=args.force,
        )
        for model in ALL_MODELS:
            summary_map = {
                label: summary[(summary["model_name"] == model) & (summary["target_group"] == label)]
                for label in labels
                if not summary[(summary["model_name"] == model) & (summary["target_group"] == label)].empty
            }
            plot_stat_line(
                summary_map=summary_map,
                stat="mse",
                output_path=output_dir / f"mse_{path_name}_{MODEL_FILE_NAMES[model]}.pdf",
                legend_order=labels,
            )

    all_paths = [
        [
            build_target_label({"rater_education": "College degree or higher"}),
            build_target_label({"rater_race": "Black/African American", "rater_education": "College degree or higher"}),
            build_target_label({"rater_gender": "Woman", "rater_race": "Black/African American", "rater_education": "College degree or higher"}),
        ],
        [
            build_target_label({"rater_education": "College degree or higher"}),
            build_target_label({"rater_race": "LatinX, Latino, Hispanic or Spanish Origin", "rater_education": "College degree or higher"}),
            build_target_label({"rater_gender": "Woman", "rater_race": "LatinX, Latino, Hispanic or Spanish Origin", "rater_education": "College degree or higher"}),
        ],
        [
            build_target_label({"rater_age": "gen z"}),
            build_target_label({"rater_gender": "Man", "rater_age": "gen z"}),
            build_target_label({"rater_gender": "Man", "rater_age": "gen z", "rater_education": "High school or below"}),
        ],
    ]

    target_union = sorted({target for path in all_paths for target in path})
    filtered = filter_llm_annotations(
        llm_annotations=llm_annotations,
        target_labels=target_union,
        models=["gpt-5.1", "gpt-5.1_reasoning=high"],
        experiment_group="main",
    )
    summary = summarize_llm_annotations(
        annotations=filtered,
        group_columns=["model_name", "target_group"],
        cache_prefix="dices__path_stats",
        force=args.force,
    )
    for stat in ["bias", "var"]:
        for model in ["gpt-5.1", "gpt-5.1_reasoning=high"]:
            summary_map = {}
            for path in all_paths:
                path_summary = summary[
                    (summary["model_name"] == model) & (summary["target_group"].isin(path))
                ].copy()
                if path_summary.empty:
                    continue
                pretty_label = " -> ".join(format_dices_label(target) for target in path)
                summary_map[pretty_label] = path_summary
            plot_stat_at_one(
                summary_map=summary_map,
                stat=stat,
                output_path=output_dir / f"{stat}_at_1_{MODEL_FILE_NAMES[model]}.pdf",
            )

    print(f"Saved DICES paper figures to {output_dir}")


if __name__ == "__main__":
    main()
