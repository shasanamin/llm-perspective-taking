from __future__ import annotations

from src.providers.huggingface_local import HuggingFaceLocalProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.openai_responses import OpenAIResponsesProvider


def build_provider(provider_config: dict):
    provider_kind = provider_config["kind"]
    if provider_kind == "openai_compatible":
        return OpenAICompatibleProvider(
            api_url=provider_config["api_url"],
            api_key=provider_config["api_key"],
            timeout_seconds=int(provider_config.get("timeout_seconds", 90)),
            default_headers=provider_config.get("headers"),
            default_body=provider_config.get("default_body"),
        )
    if provider_kind == "openai_responses":
        return OpenAIResponsesProvider(
            api_key=provider_config["api_key"],
            timeout_seconds=int(provider_config.get("timeout_seconds", 90)),
            base_url=provider_config.get("base_url"),
        )
    if provider_kind == "huggingface_local":
        return HuggingFaceLocalProvider(
            model_name=provider_config["model_name"],
            cache_dir=provider_config.get("cache_dir"),
            device_map=provider_config.get("device_map", "auto"),
            torch_dtype=provider_config.get("torch_dtype", "auto"),
            trust_remote_code=bool(provider_config.get("trust_remote_code", True)),
            hf_token=provider_config.get("hf_token"),
        )
    raise ValueError(f"Unsupported provider kind: {provider_kind}")
