from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.config import load_config
from src.generation import run_generation_job
from src.prompts import render_dices_prompt
from src.providers import build_provider
from src.datasets import get_paper_target_labels, load_items
from src.utils.common import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DICES perspective-taking generation.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--runtime-config", help="Optional runtime/provider config YAML.")
    parser.add_argument("--test", action="store_true", help="Override config and run a short smoke test.")
    parser.add_argument("--model", help="Override the configured model name.")
    parser.add_argument("--api-model", help="Optional provider-side model name when it differs from the recorded model name.")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode for Qwen-style runs.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort for providers that expose it directly.",
    )
    parser.add_argument("--output", help="Override the output JSONL path.")
    return parser.parse_args()


def apply_openai_reasoning_compatibility(config: dict) -> None:
    provider_kind = config["provider"]["kind"]
    api_model_name = str(config["model"].get("api_model_name", config["model"]["name"]))
    reasoning_effort = config["model"].get("reasoning_effort")

    if provider_kind != "openai_responses":
        return
    if not api_model_name.startswith("gpt-5.4"):
        return
    if reasoning_effort in (None, "none"):
        return

    config["model"]["temperature"] = None
    config["model"]["top_p"] = None


def main() -> None:
    args = parse_args()
    config = load_config(args.config, runtime_config_path=args.runtime_config)

    if args.model:
        config["model"]["name"] = args.model
    if args.api_model:
        config["model"]["api_model_name"] = args.api_model
    if args.test:
        config["generation"]["test_mode"] = True
    if args.reasoning:
        config["model"]["reasoning_enabled"] = True
    if args.reasoning_effort is not None:
        config["model"]["reasoning_effort"] = args.reasoning_effort
        config["model"]["reasoning_enabled"] = args.reasoning_effort != "none"

    model_name = config["model"]["name"]
    if config["model"].get("reasoning_enabled") and model_name.startswith("qwen3:"):
        config["model"]["api_model_name"] = model_name
        config["model"]["name"] = model_name.replace("qwen3:", "qwen3-r:")
        extra_body = dict(config["model"].get("extra_body", {}))
        extra_body["think"] = True
        config["model"]["extra_body"] = extra_body
    elif model_name.startswith("qwen3:"):
        extra_body = dict(config["model"].get("extra_body", {}))
        extra_body.setdefault("think", False)
        config["model"]["extra_body"] = extra_body

    apply_openai_reasoning_compatibility(config)

    items = load_items()
    target_groups = config["experiment"].get("target_groups")
    if not target_groups:
        target_groups = get_paper_target_labels()

    provider = build_provider(config["provider"])
    default_output = PROJECT_ROOT / "data" / "llm_annotations" / "dices" / "generated" / f"{slugify(config['experiment']['name'])}.jsonl"
    output_path = Path(args.output) if args.output else Path(config.get("output", {}).get("path", default_output))

    run_config = {
        "experiment_group": config["experiment"].get("experiment_group", "generated"),
        "run_name": config["experiment"]["name"],
        "provider_kind": config["provider"]["kind"],
        "prompt_mode": "dices_default",
    }

    run_generation_job(
        items=items,
        target_labels=target_groups,
        prompt_builder=render_dices_prompt,
        provider=provider,
        output_path=output_path,
        dataset_name="dices",
        model_config=config["model"],
        generation_config=config["generation"],
        run_config=run_config,
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
