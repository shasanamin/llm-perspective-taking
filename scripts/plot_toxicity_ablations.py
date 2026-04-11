from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT

from src.analysis.toxicity import load_llm_annotations, summarize_generic_llm_annotations
from src.plotting.style import MODEL_MIXTURE_FAMILY, MODEL_MIXTURE_SIZE
from src.plotting.toxicity import (
    mixture_groups,
    plot_mixture_error_reduction,
    plot_stat_bar,
    plot_stat_line,
)
from src.datasets import get_ground_truth_frame


PROMPT_ORDER = ["question", "definition", "levels", "examples"]
TEMPERATURE_ORDER = [0.3, 0.6, 0.9, 1.2, 1.5]
GPT_MODELS = ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1"]
ALL_MODELS = [
    "gemma3:1b", "gemma3:12b", "gemma3:27b",
    "gpt-oss:20b", "gpt-oss:120b", "gpt-5.1",
    "gpt-5.1_reasoning=low", "gpt-5.1_reasoning=medium", "gpt-5.1_reasoning=high",
    "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:32b",
    "qwen3:1.7b", "qwen3:8b", "qwen3:32b",
    "qwen3-r:1.7b", "qwen3-r:8b", "qwen3-r:32b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the toxicity ablation figures.")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "results" / "figures" / "paper" / "toxicity_ablations", help="Directory for output PDFs.")
    parser.add_argument("--force", action="store_true", help="Recompute cached bootstrap files.")
    return parser.parse_args()


def build_llm_frame(llm_annotations, experiment_groups, models):
    ground_truth = get_ground_truth_frame("female")[["comment_id", "ground_truth"]]
    frame = llm_annotations[
        (llm_annotations["target_group_canonical"] == "female")
        & (llm_annotations["experiment_group"].isin(experiment_groups))
        & (llm_annotations["model_name"].isin(models))
    ].copy()
    frame = frame.rename(columns={"model_name": "system"})
    frame = frame.merge(ground_truth, on="comment_id", how="inner")
    return frame


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_annotations = load_llm_annotations()

    prompt_frame = build_llm_frame(llm_annotations, experiment_groups=["main", "prompt"], models=GPT_MODELS)
    prompt_summary = summarize_generic_llm_annotations(
        annotations=prompt_frame,
        group_columns=["system", "prompt_mode"],
        cache_prefix="toxicity_prompt",
        force=args.force,
    )
    prompt_summary.to_csv(PROJECT_ROOT / "results" / "processed" / "summaries" / "toxicity_prompt.csv", index=False)

    prompt_display_names = {
        "question": "L1-Question",
        "definition": "L2-Definition",
        "levels": "L3-Levels",
        "examples": "L4-Examples",
    }
    for stat in ["mse", "var", "bias"]:
        for model in GPT_MODELS:
            summary_map = {
                prompt_mode: prompt_summary[
                    (prompt_summary["system"] == model) & (prompt_summary["prompt_mode"] == prompt_mode)
                ]
                for prompt_mode in PROMPT_ORDER
            }
            if stat == "bias":
                plot_stat_bar(
                    summary_map=summary_map,
                    output_path=output_dir / f"{stat}_prompt_{model}.pdf",
                    legend_order=PROMPT_ORDER,
                    display_names=prompt_display_names,
                    show_legend=model != "gpt-oss:20b",
                )
            else:
                plot_stat_line(
                    summary_map=summary_map,
                    stat=stat,
                    output_path=output_dir / f"{stat}_prompt_{model}.pdf",
                    legend_order=PROMPT_ORDER,
                    display_names=prompt_display_names,
                )

    temperature_frame = build_llm_frame(llm_annotations, experiment_groups=["main", "temperature"], models=GPT_MODELS)
    temperature_summary = summarize_generic_llm_annotations(
        annotations=temperature_frame,
        group_columns=["system", "temperature"],
        cache_prefix="toxicity_temperature",
        force=args.force,
    )
    temperature_summary.to_csv(PROJECT_ROOT / "results" / "processed" / "summaries" / "toxicity_temperature.csv", index=False)

    temperature_display_names = {temperature: f"T={temperature}" for temperature in TEMPERATURE_ORDER}
    for stat in ["mse", "var", "bias"]:
        for model in GPT_MODELS:
            summary_map = {
                temperature: temperature_summary[
                    (temperature_summary["system"] == model) & (temperature_summary["temperature"] == temperature)
                ]
                for temperature in TEMPERATURE_ORDER
            }
            if stat == "bias":
                plot_stat_bar(
                    summary_map=summary_map,
                    output_path=output_dir / f"{stat}_temperature_{model}.pdf",
                    legend_order=TEMPERATURE_ORDER,
                    display_names=temperature_display_names,
                    show_legend=model == "gpt-oss:20b",
                )
            else:
                plot_stat_line(
                    summary_map=summary_map,
                    stat=stat,
                    output_path=output_dir / f"{stat}_temperature_{model}.pdf",
                    legend_order=TEMPERATURE_ORDER,
                    display_names=temperature_display_names,
                )

    base_frame = build_llm_frame(llm_annotations, experiment_groups=["main"], models=ALL_MODELS)

    for mixture_name, mixture_map in [("mixture_family", MODEL_MIXTURE_FAMILY), ("mixture_size", MODEL_MIXTURE_SIZE)]:
        pooled_frames = []
        for system in ALL_MODELS:
            pooled = base_frame[base_frame["system"] == system].copy()
            pooled_frames.append(pooled)

        expanded_mixtures = {}
        for key, models in mixture_map.items():
            group_labels = mixture_groups(models)
            expanded_mixtures[key] = group_labels
            for pair in combinations(models, 2):
                label = " & ".join(pair)
                pooled = base_frame[base_frame["system"].isin(pair)].copy()
                pooled["system"] = label
                pooled_frames.append(pooled)
            pooled = base_frame[base_frame["system"].isin(models)].copy()
            pooled["system"] = " & ".join(models)
            pooled_frames.append(pooled)

        pooled_annotations = pd.concat(pooled_frames, ignore_index=True)

        summary = summarize_generic_llm_annotations(
            annotations=pooled_annotations,
            group_columns=["system"],
            cache_prefix=f"toxicity_{mixture_name}",
            force=args.force,
        )
        summary.to_csv(PROJECT_ROOT / "results" / "processed" / "summaries" / f"toxicity_{mixture_name}.csv", index=False)
        plot_mixture_error_reduction(
            summary=summary,
            mixture_map=expanded_mixtures,
            output_dir=output_dir,
            filename_prefix=f"{mixture_name}_",
        )

    print(f"Saved toxicity ablation figures to {output_dir}")


if __name__ == "__main__":
    main()
