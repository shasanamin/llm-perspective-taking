from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.analysis.toxicity import load_llm_annotations, summarize_main_experiment
from src.datasets import canonical_target_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute toxicity bootstrap summaries.")
    parser.add_argument("--target", required=True, help="Target group: female, male, or non-binary.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to include.")
    parser.add_argument("--experiment-group", default="main", help="Experiment group to analyze.")
    parser.add_argument("--output", help="Where to save the summary CSV.")
    parser.add_argument("--force", action="store_true", help="Recompute cached bootstrap files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_label = canonical_target_label(args.target)
    llm_annotations = load_llm_annotations()
    summary = summarize_main_experiment(
        llm_annotations=llm_annotations,
        target_label=target_label,
        models=args.models,
        experiment_group=args.experiment_group,
        force=args.force,
    )
    output_path = args.output or PROJECT_ROOT / "results" / "processed" / "summaries" / f"toxicity_main__{args.experiment_group}__{target_label}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
