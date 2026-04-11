from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.analysis.dices import filter_llm_annotations, load_llm_annotations, summarize_llm_annotations
from src.datasets import get_paper_target_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DICES bootstrap summaries.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to include.")
    parser.add_argument("--targets", nargs="*", help="Optional DICES target labels. Defaults to the paper target set.")
    parser.add_argument("--output", help="Where to save the summary CSV.")
    parser.add_argument("--force", action="store_true", help="Recompute cached bootstrap files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_annotations = load_llm_annotations()
    filtered = filter_llm_annotations(
        llm_annotations=llm_annotations,
        target_labels=args.targets or get_paper_target_labels(),
        models=args.models,
        experiment_group="main",
    )
    summary = summarize_llm_annotations(
        annotations=filtered,
        group_columns=["model_name", "target_group"],
        cache_prefix="dices__main",
        force=args.force,
    )
    output_path = args.output or PROJECT_ROOT / "results" / "processed" / "summaries" / "dices_main.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
