"""
Compute differential perspective-taking metrics for all toxicity experiments.

Usage:
    python scripts/analyze_differential_pt.py
    python scripts/analyze_differential_pt.py --output-dir results/processed/differential_pt/toxicity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.analysis.differential import (
    compare_human_vs_llms,
    run_all_toxicity,
)
from src.analysis.toxicity import load_llm_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run differential PT analysis for toxicity.")
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "results" / "processed" / "differential_pt" / "toxicity",
        help="Directory for output CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LLM annotations …")
    llm_annotations = load_llm_annotations()
    print(f"  → {len(llm_annotations)} records, {llm_annotations['model_name'].nunique()} models")

    print("\n" + "=" * 72)
    print("DIFFERENTIAL PERSPECTIVE-TAKING — TOXICITY")
    print("=" * 72)

    summary = run_all_toxicity(
        llm_annotations=llm_annotations,
        output_dir=output_dir,
    )

    print(f"\nSaved summary → {output_dir / 'summary.csv'}")

    # Statistical comparisons
    print("\n" + "-" * 72)
    print("Fisher z-tests: Human PT (In-Group) vs LLMs")
    print("-" * 72)

    comparisons = compare_human_vs_llms(summary)
    comparisons.to_csv(output_dir / "fisher_z_tests.csv", index=False)
    print(f"Saved {len(comparisons)} comparisons → fisher_z_tests.csv")

    # Print significant results
    sig = comparisons[comparisons["significant_005"]]
    if len(sig) > 0:
        print("\nSignificant results (human > LLM, p < 0.05):")
        for _, row in sig.iterrows():
            print(
                f"  {row['group1']} vs {row['group2']} | "
                f"Human ρ={row['r_human']:.3f} vs {row['llm_display']} ρ={row['r_llm']:.3f} | "
                f"z={row['fisher_z']:.2f}, p={row['p_one_sided_human_gt']:.3f}"
            )
    else:
        print("\nNo significant differences found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
