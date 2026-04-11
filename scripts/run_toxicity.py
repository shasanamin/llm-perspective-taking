from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from src.config import load_config
from src.generation import run_generation_job
from src.prompts import render_toxicity_prompt
from src.providers import build_provider
from src.datasets import (
    build_example_payload,
    canonical_target_label,
    load_comments,
    prompt_target_label,
)
from src.utils.common import slugify
from src.utils.models import resolve_hf_local_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toxicity perspective-taking generation.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--runtime-config", help="Optional runtime/provider config YAML.")
    parser.add_argument("--test", action="store_true", help="Override config and run a short smoke test.")
    parser.add_argument("--model", help="Override the configured model name.")
    parser.add_argument("--api-model", help="Optional provider-side model name when it differs from the recorded model name.")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode for Qwen-style runs.")
    parser.add_argument(
        "--accept-unparsed-response",
        action="store_true",
        help="Treat a non-empty response as valid even if percentage parsing fails.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort for providers that expose it directly, such as OpenAI Responses.",
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

    # GPT-5.4 only accepts temperature/top_p when reasoning effort is "none".
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
    if args.accept_unparsed_response:
        config["model"]["accept_unparsed_response"] = True

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

    # Qwen3.5 reasoning / non-reasoning via HuggingFace (uses enable_thinking)
    if config["model"].get("reasoning_enabled") and model_name.startswith("qwen3.5:"):
        config["model"]["api_model_name"] = config["model"].get("api_model_name", model_name)
        config["model"]["name"] = model_name.replace("qwen3.5:", "qwen3.5-r:")
        extra_body = dict(config["model"].get("extra_body", {}))
        extra_body["enable_thinking"] = True
        config["model"]["extra_body"] = extra_body
    elif model_name.startswith("qwen3.5:"):
        extra_body = dict(config["model"].get("extra_body", {}))
        extra_body.setdefault("enable_thinking", False)
        config["model"]["extra_body"] = extra_body

    # Ministral 3 reasoning uses dedicated checkpoints instead of a runtime
    # thinking toggle. Allow the same config pattern as Qwen for convenience.
    if config["model"].get("reasoning_enabled") and model_name.startswith("ministral3:"):
        config["model"]["api_model_name"] = config["model"].get("api_model_name", model_name)
        config["model"]["name"] = model_name.replace("ministral3:", "ministral3-r:")

    apply_openai_reasoning_compatibility(config)

    if config["provider"]["kind"] == "huggingface_local":
        requested_model = str(config["model"].get("api_model_name", config["model"]["name"]))
        resolved_model = resolve_hf_local_model_name(requested_model)
        config["model"]["api_model_name"] = resolved_model
        config["provider"]["model_name"] = resolved_model

    comments = load_comments()
    prompt_mode = config["experiment"].get("prompt_mode", "levels")
    target_groups = config["experiment"].get("target_groups")
    if not target_groups:
        target_groups = [config["experiment"]["target_group"]]

    def prompt_builder(comment: str, target_label: str) -> str:
        examples = None
        if prompt_mode == "examples":
            examples = build_example_payload(
                canonical_target_label(target_label),
                config["experiment"]["example_indices"],
            )
        return render_toxicity_prompt(
            comment=comment,
            target_label=prompt_target_label(target_label),
            prompt_mode=prompt_mode,
            examples=examples,
        )

    provider = build_provider(config["provider"])
    default_output = PROJECT_ROOT / "data" / "llm_annotations" / "toxicity_detection" / "generated" / f"{slugify(config['experiment']['name'])}.jsonl"
    output_path = Path(args.output) if args.output else Path(config.get("output", {}).get("path", default_output))

    canonical_targets = [canonical_target_label(target) for target in target_groups]
    run_config = {
        "experiment_group": config["experiment"].get("experiment_group", "generated"),
        "run_name": config["experiment"]["name"],
        "provider_kind": config["provider"]["kind"],
        "prompt_mode": prompt_mode,
        "target_group_canonical": canonical_targets[0] if len(canonical_targets) == 1 else None,
    }

    run_generation_job(
        items=comments,
        target_labels=canonical_targets,
        prompt_builder=prompt_builder,
        provider=provider,
        output_path=output_path,
        dataset_name="toxicity_detection",
        model_config=config["model"],
        generation_config=config["generation"],
        run_config=run_config,
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
