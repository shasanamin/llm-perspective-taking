from __future__ import annotations


MODEL_DISPLAY_NAMES = {
    "direct": "Direct Annotation",
    "perspective": "PT (In-Group)",
    "perspective_out": "PT (Out-Group)",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.1_reasoning=high": "GPT-5.1-R=H",
    "gpt-5.1_reasoning=medium": "GPT-5.1-R=M",
    "gpt-5.1_reasoning=low": "GPT-5.1-R=L",
    "gpt-5.1_reasoning=none": "GPT-5.1",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4_reasoning=high": "GPT-5.4-R=H",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4-mini_reasoning=high": "GPT-5.4m-R=H",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "gpt-5.4-nano_reasoning=low": "GPT-5.4n-R=L",
    "gpt-5.4-nano_reasoning=medium": "GPT-5.4n-R=M",
    "gpt-5.4-nano_reasoning=high": "GPT-5.4n-R=H",
    "gpt-5.4-nano_reasoning=xhigh": "GPT-5.4n-R=XH",
    "gpt-oss:20b": "GPT-OSS:20B",
    "gpt-oss:120b": "GPT-OSS:120B",
    "gpt-oss:latest": "GPT-OSS:20B",
    "qwen3:1.7b": "Qwen3 1.7B",
    "qwen3:8b": "Qwen3 8B",
    "qwen3:32b": "Qwen3 32B",
    "qwen3-r:1.7b": "Qwen3-R 1.7B",
    "qwen3-r:8b": "Qwen3-R 8B",
    "qwen3-r:32b": "Qwen3-R 32B",
    "deepseek-r1:1.5b": "DeepSeek-R1 1.5B",
    "deepseek-r1:7b": "DeepSeek-R1 7B",
    "deepseek-r1:32b": "DeepSeek-R1 32B",
    "gemma3:1b": "Gemma3 1B",
    "gemma3:12b": "Gemma3 12B",
    "gemma3:27b": "Gemma3 27B",
    # Qwen3.5 (HuggingFace local)
    "qwen3.5:0.8b": "Qwen3.5 0.8B",
    "qwen3.5:4b": "Qwen3.5 4B",
    "qwen3.5:9b": "Qwen3.5 9B",
    "qwen3.5-r:0.8b": "Qwen3.5-R 0.8B",
    "qwen3.5-r:4b": "Qwen3.5-R 4B",
    "qwen3.5-r:9b": "Qwen3.5-R 9B",
    "qwen3.5-base:0.8b": "Qwen3.5-Base 0.8B",
    "qwen3.5-base:4b": "Qwen3.5-Base 4B",
    "qwen3.5-base:9b": "Qwen3.5-Base 9B",
    # Llama 3.1 (HuggingFace local)
    "llama3.1:8b": "Llama3.1 8B",
    "llama3.1:8b-instruct": "Llama3.1 8B-Instruct",
    "llama3.1:70b": "Llama3.1 70B",
    "llama3.1:70b-instruct": "Llama3.1 70B-Instruct",
    # Gemma 3 (HuggingFace local)
    "gemma3-it:1b": "Gemma3-IT 1B",
    "gemma3-pt:1b": "Gemma3-PT 1B",
    "gemma3-it:12b": "Gemma3-IT 12B",
    "gemma3-pt:12b": "Gemma3-PT 12B",
    "gemma3-it:27b": "Gemma3-IT 27B",
    "gemma3-pt:27b": "Gemma3-PT 27B",
    # Ministral 3 (HuggingFace local)
    "ministral3-base:3b": "Ministral3-Base 3B",
    "ministral3:3b": "Ministral3 3B-Instruct",
    "ministral3-r:3b": "Ministral3-R 3B",
    "ministral3-base:8b": "Ministral3-Base 8B",
    "ministral3:8b": "Ministral3 8B-Instruct",
    "ministral3-r:8b": "Ministral3-R 8B",
    "ministral3-base:14b": "Ministral3-Base 14B",
    "ministral3:14b": "Ministral3 14B-Instruct",
    "ministral3-r:14b": "Ministral3-R 14B",
}


HF_LOCAL_MODEL_IDS = {
    # Qwen3.5 instruct / reasoning share the same post-trained checkpoint.
    "qwen3.5:0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5:4b": "Qwen/Qwen3.5-4B",
    "qwen3.5:9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-r:0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-r:4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-r:9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-base:0.8b": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3.5-base:4b": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5-base:9b": "Qwen/Qwen3.5-9B-Base",
    # Llama 3.1
    "llama3.1:8b": "meta-llama/Llama-3.1-8B",
    "llama3.1:8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Llama-3.1-70B",
    "llama3.1:70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    # Gemma 3
    "gemma3-it:1b": "google/gemma-3-1b-it",
    "gemma3-it:12b": "google/gemma-3-12b-it",
    "gemma3-it:27b": "google/gemma-3-27b-it",
    "gemma3-pt:1b": "google/gemma-3-1b-pt",
    "gemma3-pt:12b": "google/gemma-3-12b-pt",
    "gemma3-pt:27b": "google/gemma-3-27b-pt",
    # Ministral 3
    "ministral3-base:3b": "mistralai/Ministral-3-3B-Base-2512",
    "ministral3:3b": "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral3-r:3b": "mistralai/Ministral-3-3B-Reasoning-2512",
    "ministral3-base:8b": "mistralai/Ministral-3-8B-Base-2512",
    "ministral3:8b": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral3-r:8b": "mistralai/Ministral-3-8B-Reasoning-2512",
    "ministral3-base:14b": "mistralai/Ministral-3-14B-Base-2512",
    "ministral3:14b": "mistralai/Ministral-3-14B-Instruct-2512",
    "ministral3-r:14b": "mistralai/Ministral-3-14B-Reasoning-2512",
}


def normalize_model_name(model_name: str) -> str:
    if model_name == "gpt-oss:latest":
        return "gpt-oss:20b"
    if model_name.endswith("_reasoning=none"):
        return model_name[: -len("_reasoning=none")]
    return model_name


def infer_model_family(model_name: str) -> str:
    normalized = normalize_model_name(model_name).lower()
    if normalized.startswith("gpt"):
        return "gpt"
    if normalized.startswith("qwen3.5-r"):
        return "qwen3.5-r"
    if normalized.startswith("qwen3.5-base"):
        return "qwen3.5-base"
    if normalized.startswith("qwen3.5"):
        return "qwen3.5"
    if normalized.startswith("qwen3-r"):
        return "qwen3-r"
    if normalized.startswith("qwen3"):
        return "qwen3"
    if normalized.startswith("llama3.1"):
        return "llama3.1"
    if normalized.startswith("ministral3-r"):
        return "ministral3-r"
    if normalized.startswith("ministral3-base"):
        return "ministral3-base"
    if normalized.startswith("ministral3"):
        return "ministral3"
    if normalized.startswith("gemma3-it") or normalized.startswith("gemma3-pt"):
        return "gemma3"
    if normalized.startswith("gemma3"):
        return "gemma3"
    if normalized.startswith("deepseek-r1"):
        return "deepseek-r1"
    return "other"


def display_model_name(model_name: str) -> str:
    normalized = normalize_model_name(model_name)
    return MODEL_DISPLAY_NAMES.get(normalized, normalized)


def resolve_hf_local_model_name(model_name: str) -> str:
    normalized = normalize_model_name(model_name)
    return HF_LOCAL_MODEL_IDS.get(normalized, model_name)
