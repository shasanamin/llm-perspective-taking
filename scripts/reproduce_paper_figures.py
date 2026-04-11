"""Reproduce all paper figures from the bundled LLM annotation data.

Runs the analysis and plotting scripts in dependency order, writing
figures to results/figures/.  No API access is required — all
pre-generated LLM annotations are bundled under data/llm_annotations/.

Usage:
    python scripts/reproduce_paper_figures.py [--force]

Flags:
    --force    Recompute cached bootstrap CSV files instead of reusing them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SCRIPTS = [
    # ── Main paper figures ──────────────────────────────────────────────
    "plot_toxicity_paper.py",          # Figs 1-3, 6, 8
    "plot_toxicity_ablations.py",      # Figs 7, 9 + appendix ablations
    "plot_dices_paper.py",             # Figs 4-5  + appendix DICES

    # ── Differential perspective-taking (Figs 10-11) ────────────────────
    "analyze_differential_pt.py",      # prerequisite: compute DPT metrics
    "plot_differential_pt.py",         # scatter panel, correlation bars, scaling

    # ── Pretrained vs post-trained (Appendix Fig A13) ───────────────────
    "plot_hf_females.py",              # prerequisite: compute HF bootstrap summary
    "plot_pt_vs_it_paper.py",          # Gemma3 / Ministral3 panel figures

    # ── Reasoning paradox (supplementary) ───────────────────────────────
    "analyze_reasoning_paradox.py",    # delta heatmaps, trace analysis, effort sweep
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all paper figures from bundled data."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute cached bootstrap CSV files.",
    )
    return parser.parse_args()


def run_script(script_name: str, *extra_args: str) -> bool:
    """Run a single script.  Returns True on success, False on failure."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), *extra_args]
    print(f"\n{'─'*60}\nRunning {script_name} …\n{'─'*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {script_name} exited with code {result.returncode}")
        return False
    return True


def main() -> None:
    args = parse_args()
    extra = ["--force"] if args.force else []
    failed = []
    for script in SCRIPTS:
        if not run_script(script, *extra):
            failed.append(script)
    print(f"\n{'═'*60}\nAll figures written to {PROJECT_ROOT / 'results' / 'figures'}/\n{'═'*60}")
    if failed:
        print(f"\nWARNING: {len(failed)} script(s) had errors (likely missing data):")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
