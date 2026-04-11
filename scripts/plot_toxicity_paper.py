from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.analysis.toxicity import load_llm_annotations, summarize_main_experiment
from src.plotting.style import MODEL_FAMILIES
from src.plotting.toxicity import (
    plot_grouped_bars,
    plot_llm_vs_humans,
    plot_model_family,
    plot_nonbinary_reasoning_bias,
    plot_two_target_comparison,
)
from src.datasets import display_target_label


GPT_MODELS = ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1"]
ALL_MODELS = list(dict.fromkeys(model for family in MODEL_FAMILIES.values() for model in family))
TARGETS = ["female", "male", "non-binary"]
TARGET_FILE_SLUGS = {"female": "female", "male": "male", "non-binary": "nonbinary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the main toxicity paper figures.")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "results" / "figures" / "paper", help="Directory for output PDFs.")
    parser.add_argument("--force", action="store_true", help="Recompute cached bootstrap files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_dir = PROJECT_ROOT / "results" / "processed" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_annotations = load_llm_annotations()

    gpt_summary_by_target = {}
    for target in TARGETS:
        summary = summarize_main_experiment(
            llm_annotations=llm_annotations,
            target_label=target,
            models=GPT_MODELS,
            force=args.force,
        )
        summary.to_csv(summary_dir / f"toxicity_main__{target}__gpt.csv", index=False)
        gpt_summary_by_target[display_target_label(target)] = summary
        plot_llm_vs_humans(
            summary=summary,
            output_path=output_dir / f"llm_human_{target}_gpt.pdf",
        )

    for stat in ["mse", "bias", "var"]:
        plot_grouped_bars(
            summary_by_target=gpt_summary_by_target,
            stat=stat,
            output_path=output_dir / f"{stat}_at_1_by_model_gpt.pdf",
        )

    plot_two_target_comparison(
        summary_by_target={
            "Male": gpt_summary_by_target["Male"],
            "Female": gpt_summary_by_target["Female"],
        },
        output_path=output_dir / "llm_human_male_female_gpt-oss:120b.pdf",
    )

    for target in TARGETS:
        summary = summarize_main_experiment(
            llm_annotations=llm_annotations,
            target_label=target,
            models=ALL_MODELS,
            force=args.force,
        )
        summary.to_csv(summary_dir / f"toxicity_main__{target}__all_models.csv", index=False)
        target_slug = TARGET_FILE_SLUGS[target]
        for stat in ["mse", "bias", "var"]:
            plot_model_family(
                summary=summary,
                stat=stat,
                output_path=output_dir / f"{stat}_at_1_model_family_{target_slug}.pdf",
            )
        if target == "non-binary":
            plot_nonbinary_reasoning_bias(
                summary=summary,
                output_path=output_dir / "bias_at_1_nonbinary_reasoning.pdf",
            )

    print(f"Saved toxicity paper figures to {output_dir}")


if __name__ == "__main__":
    main()
