from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd

from src.providers.base import GenerationRequest
from src.utils.common import utc_now_iso
from src.utils.jsonl import append_jsonl, iter_jsonl
from src.utils.models import infer_model_family
from src.utils.text import extract_percentage


def load_completed_keys(path: str | Path) -> set[tuple[str, str, int, str, int]]:
    completed = set()
    for record in iter_jsonl(path) or []:
        if record.get("parsed_percentage") is None and not record.get("accepted_without_parse", False):
            continue
        model_name = str(record.get("model_name", ""))
        run_name = str(record.get("run_name", model_name))
        completed.add(
            (
                run_name,
                model_name,
                int(record["comment_id"]),
                str(record["target_group"]),
                int(record["generation_index"]),
            )
        )
    return completed


def is_fatal_generation_error(message: str) -> bool:
    lowered = message.lower()
    fatal_markers = (
        "insufficient_quota",
        "invalid_api_key",
        "authentication",
        "permission",
        "model_not_found",
        "unsupported parameter",
        "unsupported value",
    )
    return any(marker in lowered for marker in fatal_markers)


def _generate_record(
    *,
    provider,
    api_model_name: str,
    model_name: str,
    dataset_name: str,
    run_name: str,
    run_config: dict,
    model_config: dict,
    prompt_mode: str,
    target_label: str,
    comment_id: int,
    comment: str,
    prompt: str,
    generation_index: int,
    max_retry: int,
    retry_sleep_seconds: float,
    request_sleep_seconds: float,
    capture_reasoning: bool,
) -> dict | None:
    response_text = ""
    reasoning_trace = ""
    reasoning_summary = ""
    reasoning_encrypted_content = ""
    reasoning_trace_format = ""
    parsed_percentage = None
    generation_time_s = None
    accepted_without_parse = False
    last_error = ""
    allow_unparsed_response = bool(model_config.get("accept_unparsed_response", False))

    for _ in range(max_retry):
        try:
            request = GenerationRequest(
                model_name=api_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_config.get("temperature"),
                top_p=model_config.get("top_p"),
                max_tokens=model_config.get("max_tokens"),
                reasoning_enabled=bool(model_config.get("reasoning_enabled", False)),
                extra_body=model_config.get("extra_body", {}),
                reasoning_effort=model_config.get("reasoning_effort"),
            )
            t_generation = time.time()
            response = provider.generate(request, capture_reasoning=capture_reasoning)
            generation_time_s = round(time.time() - t_generation, 3)
            response_text = response.text
            reasoning_trace = response.reasoning
            reasoning_summary = response.reasoning_summary
            reasoning_encrypted_content = response.reasoning_encrypted_content
            reasoning_trace_format = response.reasoning_format
            _, parsed_percentage = extract_percentage(response_text)
            if parsed_percentage is not None:
                time.sleep(request_sleep_seconds)
                break
            if allow_unparsed_response and response_text.strip():
                accepted_without_parse = True
                time.sleep(request_sleep_seconds)
                break
            last_error = f"Could not parse a percentage from: {response_text[:120]}"
        except Exception as exc:
            last_error = str(exc)
            if is_fatal_generation_error(last_error):
                break
            time.sleep(retry_sleep_seconds)

    if parsed_percentage is None and not accepted_without_parse:
        print(
            f"Skipping failed generation for model={model_name} target={target_label} "
            f"comment_id={comment_id} generation_index={generation_index}: {last_error}"
        )
        return None

    return {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "experiment_group": run_config["experiment_group"],
        "run_name": run_name,
        "source": run_config.get("source", "api_generation"),
        "created_at": utc_now_iso(),
        "provider_kind": run_config["provider_kind"],
        "model_name": model_name,
        "api_model_name": api_model_name,
        "model_family": infer_model_family(model_name),
        "target_group": target_label,
        "target_group_canonical": run_config.get("target_group_canonical", target_label),
        "comment_id": comment_id,
        "comment": comment,
        "generation_index": generation_index,
        "prompt_mode": prompt_mode,
        "temperature": model_config.get("temperature"),
        "top_p": model_config.get("top_p"),
        "max_tokens": model_config.get("max_tokens"),
        "reasoning_enabled": bool(model_config.get("reasoning_enabled", False)),
        "response_text": response_text,
        "parsed_percentage": parsed_percentage,
        "accepted_without_parse": accepted_without_parse,
        "reasoning_trace": reasoning_trace,
        "reasoning_summary": reasoning_summary,
        "reasoning_encrypted_content": reasoning_encrypted_content,
        "reasoning_trace_format": reasoning_trace_format,
        "request_extra_body": model_config.get("extra_body", {}),
        "generation_time_s": generation_time_s,
    }


def run_generation_job(
    items: pd.DataFrame,
    target_labels: list[str],
    prompt_builder: Callable[[str, str], str],
    provider,
    output_path: str | Path,
    dataset_name: str,
    model_config: dict,
    generation_config: dict,
    run_config: dict,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    test_mode = bool(generation_config.get("test_mode", False))
    if test_mode:
        items = items.head(int(generation_config.get("test_item_count", 2))).copy()

    n_generations = 1 if test_mode else int(generation_config.get("n_generations", 10))
    max_retry = int(generation_config.get("max_retry", 5))
    retry_sleep_seconds = float(generation_config.get("retry_sleep_seconds", 10))
    request_sleep_seconds = float(generation_config.get("request_sleep_seconds", 1))
    generation_concurrency = max(1, int(generation_config.get("generation_concurrency", 1)))
    capture_reasoning = bool(model_config.get("reasoning_enabled", False))

    model_name = model_config["name"]
    api_model_name = model_config.get("api_model_name", model_name)
    prompt_mode = run_config.get("prompt_mode", "levels")
    run_name = str(run_config["run_name"])
    completed = load_completed_keys(output_file)

    pending_jobs: list[dict] = []
    for target_label in target_labels:
        for row in items.itertuples(index=False):
            comment_id = int(row.comment_id)
            comment = row.comment
            prompt = prompt_builder(comment, target_label)

            for generation_index in range(n_generations):
                key = (run_name, model_name, comment_id, target_label, generation_index)
                if key in completed:
                    continue
                pending_jobs.append(
                    {
                        "key": key,
                        "target_label": target_label,
                        "comment_id": comment_id,
                        "comment": comment,
                        "prompt": prompt,
                        "generation_index": generation_index,
                    }
                )

    def build_record(job: dict) -> dict | None:
        return _generate_record(
            provider=provider,
            api_model_name=api_model_name,
            model_name=model_name,
            dataset_name=dataset_name,
            run_name=run_name,
            run_config=run_config,
            model_config=model_config,
            prompt_mode=prompt_mode,
            target_label=job["target_label"],
            comment_id=job["comment_id"],
            comment=job["comment"],
            prompt=job["prompt"],
            generation_index=job["generation_index"],
            max_retry=max_retry,
            retry_sleep_seconds=retry_sleep_seconds,
            request_sleep_seconds=request_sleep_seconds,
            capture_reasoning=capture_reasoning,
        )

    if generation_concurrency == 1 or provider.__class__.__name__ == "HuggingFaceLocalProvider":
        for job in pending_jobs:
            record = build_record(job)
            if record is None:
                continue
            append_jsonl([record], output_file)
            completed.add(job["key"])
    else:
        with ThreadPoolExecutor(max_workers=generation_concurrency) as executor:
            future_map = {executor.submit(build_record, job): job for job in pending_jobs}
            for future in as_completed(future_map):
                job = future_map[future]
                record = future.result()
                if record is None:
                    continue
                append_jsonl([record], output_file)
                completed.add(job["key"])

    return output_file
