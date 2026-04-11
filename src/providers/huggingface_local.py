from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from src.providers.base import GenerationRequest, GenerationResponse


class _SanitizeLogitsProcessor:
    """Replace NaN / Inf logits with large negative values to prevent
    ``torch.multinomial`` crashes.

    Some model-dtype-hardware combinations produce occasional NaN logits
    (observed with transformers >= 5.x dev + PyTorch 2.6-2.10 + bf16 on A100).
    This processor is a cheap safety net.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        import torch as _torch
        if _torch.isnan(scores).any() or _torch.isinf(scores).any():
            scores = _torch.where(
                _torch.isnan(scores) | _torch.isinf(scores),
                _torch.full_like(scores, -1e9),
                scores,
            )
        return scores

# ---------------------------------------------------------------------------
# Registry of known model families with their recommended generation defaults.
# The provider merges these defaults under any explicit request-level overrides.
# Keys use prefixes matched against the HuggingFace model id (lowercased).
#
# NOTE on Qwen3.5 presence_penalty: The model card recommends presence_penalty=2.0
# (non-thinking) / 1.5 (thinking) via serving APIs.  presence_penalty is an
# additive API-level penalty and has NO direct equivalent in model.generate().
# transformers' repetition_penalty is a *multiplicative* penalty (1.0 = off,
# >1.0 applies a divisor to already-seen token logits).  We do NOT set
# repetition_penalty here because mapping presence_penalty→repetition_penalty
# is lossy and a value of 2.0 would be extremely aggressive.  The recommended
# repetition_penalty from the model card for direct generate() is 1.0 (default).
# ---------------------------------------------------------------------------
MODEL_FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    # Qwen3.5 instruct: recommended non-thinking text params from model card
    # Ref: https://huggingface.co/Qwen/Qwen3.5-0.8B
    # temperature=1.0, top_p=1.0, top_k=20; presence_penalty=2.0 (API only, not set here)
    "qwen/qwen3.5-": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "max_new_tokens": 2048,
    },
    # Qwen3.5 thinking mode overrides (applied on top when thinking enabled)
    # temperature=1.0, top_p=0.95, top_k=20; presence_penalty=1.5 (API only, not set here)
    "_qwen3.5_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 8192,
    },
    # Llama 3.1 instruct: recommended params from model card
    # Ref: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    "meta-llama/llama-3.1-": {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 2048,
    },
    # Gemma 3: model card does not prescribe specific sampling defaults;
    # using conservative settings similar to Llama 3.1
    # Ref: https://huggingface.co/google/gemma-3-1b-it
    "google/gemma-3-": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 2048,
    },
    # Ministral 3 family defaults are split by checkpoint type because the
    # official cards recommend different settings for Base / Instruct /
    # Reasoning variants.
    "_ministral3_base": {
        "temperature": 0.15,
        "top_p": 1.0,
        "max_new_tokens": 2048,
    },
    "_ministral3_instruct": {
        "temperature": 0.1,
        "top_p": 1.0,
        "max_new_tokens": 2048,
    },
    # Ministral 3 reasoning: official cards recommend temperature=0.7 and
    # top_p=0.95 with generous token budgets.
    "_ministral3_reasoning": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 8192,
    },
}


def _lookup_family_defaults(model_id: str) -> dict[str, Any]:
    """Return the best-matching family defaults for *model_id*."""
    lower = model_id.lower()
    if lower.startswith("mistralai/ministral-3-"):
        if "-reasoning-" in lower:
            return dict(MODEL_FAMILY_DEFAULTS["_ministral3_reasoning"])
        if "-instruct-" in lower:
            return dict(MODEL_FAMILY_DEFAULTS["_ministral3_instruct"])
        if "-base-" in lower:
            return dict(MODEL_FAMILY_DEFAULTS["_ministral3_base"])
    for prefix, defaults in MODEL_FAMILY_DEFAULTS.items():
        if prefix.startswith("_"):
            continue  # skip internal keys
        if lower.startswith(prefix):
            return dict(defaults)
    return {}


def _is_base_model(model_id: str) -> bool:
    """Heuristic: base/pretrained models don't have chat templates."""
    lower = model_id.lower()
    # Qwen3.5 base
    if "-base" in lower:
        return True
    # Gemma 3 pretrained
    if lower.startswith("google/gemma-3-") and lower.endswith("-pt"):
        return True
    # Llama 3.1 base (without -Instruct)
    if "llama-3.1-" in lower and "instruct" not in lower:
        return True
    # Ministral 3 base
    if lower.startswith("mistralai/ministral-3-") and "-base-" in lower:
        return True
    return False


def _is_ministral3_model(model_id: str) -> bool:
    return model_id.lower().startswith("mistralai/ministral-3-")


def _is_ministral3_reasoning_model(model_id: str) -> bool:
    lower = model_id.lower()
    return lower.startswith("mistralai/ministral-3-") and "-reasoning-" in lower


def _model_dtype_kwargs(transformers_version: str, dtype: Any) -> dict[str, Any]:
    """Handle the 4.x -> 5.x rename from ``torch_dtype`` to ``dtype``."""
    major = int(transformers_version.split(".", maxsplit=1)[0])
    if major >= 5:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}




class HuggingFaceLocalProvider:
    """Provider for running HuggingFace models locally on GPU.

    Supports three usage modes:

    1. **Instruct/chat models** (Qwen3.5, Llama-3.1-Instruct, Gemma-3-it):
       Uses ``tokenizer.apply_chat_template`` for proper prompt formatting.

    2. **Base/pretrained models** (Qwen3.5-Base, Llama-3.1, Gemma-3-pt):
       Uses plain text completion (no chat template).

    3. **Qwen3.5 thinking mode**: Passes ``enable_thinking=True`` through the
       chat template so the model produces a ``<think>...</think>`` block
       before its answer.  The thinking content is extracted and returned
       via ``GenerationResponse.reasoning``.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        hf_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token or None  # treat empty string as None

        self._model = None
        self._tokenizer = None
        self._is_base = _is_base_model(model_name)
        self._is_ministral3 = _is_ministral3_model(model_name)
        self._is_ministral3_reasoning = _is_ministral3_reasoning_model(model_name)

    # ------------------------------------------------------------------
    # Lazy model / tokenizer loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        dtype = self.torch_dtype
        if dtype == "auto":
            dtype = torch.bfloat16  # safe default for A100

        # NOTE: transformers 5.x dev renamed torch_dtype → dtype.
        # Use `dtype=` here (not `torch_dtype=`) to avoid deprecation warning.

        print(f"[HuggingFaceLocalProvider] Loading model: {self.model_name}")
        print(f"  cache_dir={self.cache_dir}  device_map={self.device_map}  dtype={dtype}")

        model_kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "token": self.hf_token,
        }
        model_kwargs.update(_model_dtype_kwargs(transformers_version, dtype))
        tokenizer_kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "token": self.hf_token,
        }

        if self._is_ministral3:
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

            self._tokenizer = MistralCommonBackend.from_pretrained(
                self.model_name,
                **tokenizer_kwargs,
            )
            self._model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        self._model.eval()
        try:
            loaded_on = str(next(self._model.parameters()).device)
        except StopIteration:
            loaded_on = "unknown"
        print(f"[HuggingFaceLocalProvider] Model loaded successfully (first param device: {loaded_on})")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt_chat(
        self, messages: list[dict[str, str]], enable_thinking: bool | None = None,
    ) -> str:
        """Build prompt using the tokenizer's chat template."""
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3.5 thinking mode
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking

        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _build_prompt_base(self, messages: list[dict[str, str]]) -> str:
        """Build a plain-text prompt for base/pretrained models."""
        instruction = "\n\n".join(msg["content"].strip() for msg in messages if msg.get("content")).strip()
        lower = self.model_name.lower()

        # Gemma PT models sometimes continue unrelated instruction-tuning
        # corpora when prompted with markdown-style "### Instruction" wrappers.
        # A simpler completion cue is more stable for these checkpoints.
        if lower.startswith("google/gemma-3-") and lower.endswith("-pt"):
            return (
                f"{instruction}\n\n"
                "Return only one percentage like 75%.\n"
                "Percentage:"
            )

        # Other base models respond more reliably to a completion-style wrapper
        # than to raw chat-formatted instructions.
        return (
            "Below is an instruction. Complete it with a direct answer.\n\n"
            "Return only a single percentage in the format NN%.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    def _build_tokenized_inputs(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool | None = None,
    ) -> tuple[dict[str, Any], int]:
        if self._is_ministral3:
            if self._is_base:
                prompt_text = self._build_prompt_base(messages)
                input_ids = self._tokenizer.encode(prompt_text, return_tensors="pt")
                return {"input_ids": input_ids}, int(input_ids.shape[1])

            tokenized = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = tokenized["input_ids"]
            return dict(tokenized), int(input_ids.shape[1])

        if self._is_base:
            prompt_text = self._build_prompt_base(messages)
        else:
            prompt_text = self._build_prompt_chat(messages, enable_thinking=enable_thinking)
        tokenized = self._tokenizer(prompt_text, return_tensors="pt")
        return dict(tokenized), int(tokenized["input_ids"].shape[1])

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self, request: GenerationRequest, capture_reasoning: bool = False,
    ) -> GenerationResponse:
        import torch
        self._ensure_loaded()

        # Determine thinking mode from extra_body
        enable_thinking = None
        if "enable_thinking" in request.extra_body:
            enable_thinking = bool(request.extra_body.get("enable_thinking"))

        # Resolve generation parameters: family defaults < request overrides
        family = _lookup_family_defaults(self.model_name)
        if enable_thinking:
            thinking_defaults = MODEL_FAMILY_DEFAULTS.get("_qwen3.5_thinking", {})
            family.update(thinking_defaults)

        temperature = request.temperature if request.temperature is not None else family.get("temperature")
        top_p = request.top_p if request.top_p is not None else family.get("top_p")
        top_k = family.get("top_k")  # not in GenerationRequest, use family default
        repetition_penalty = family.get("repetition_penalty")
        max_new_tokens = request.max_tokens or family.get("max_new_tokens") or 2048

        do_sample = temperature is not None and temperature > 0

        # Resolve target device.  For device_map="auto" models spread across
        # multiple GPUs, model.device is not reliable; we take the device of
        # the first parameter (always cuda:0 with accelerate auto mapping).
        try:
            target_device = next(self._model.parameters()).device
        except StopIteration:
            target_device = torch.device("cuda")

        # Tokenize
        inputs, prompt_len = self._build_tokenized_inputs(
            request.messages,
            enable_thinking=enable_thinking,
        )
        moved_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if key == "pixel_values":
                    moved_inputs[key] = value.to(dtype=torch.bfloat16, device=target_device)
                else:
                    moved_inputs[key] = value.to(target_device)
            else:
                moved_inputs[key] = value
        input_ids = moved_inputs["input_ids"]
        attention_mask = moved_inputs.get("attention_mask")

        # Build generate kwargs
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw"):
            if key in moved_inputs:
                gen_kwargs[key] = moved_inputs[key]

        # Suppress padding-side warnings for models without pad_token
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id

        # Guard against NaN/Inf in logits (observed with transformers dev + bf16)
        gen_kwargs["logits_processor"] = [_SanitizeLogitsProcessor()]

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # Decode only the new tokens
        new_token_ids = output_ids[0, prompt_len:]
        full_output = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        # Extract reasoning content when the output format exposes it.
        reasoning = ""
        text = full_output
        if enable_thinking or self._is_ministral3_reasoning:
            text, reasoning = _extract_reasoning(full_output)
        if enable_thinking and not reasoning:
            text, reasoning = _extract_thinking(full_output)

        return GenerationResponse(text=text, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_MISTRAL_THINK_RE = re.compile(r"\[THINK\](.*?)\[/THINK\]", re.DOTALL)


def _extract_thinking(output: str) -> tuple[str, str]:
    """Split Qwen3.5 ``<think>...</think>`` reasoning from the final answer."""
    match = _THINK_RE.search(output)
    if not match:
        return output, ""
    reasoning = match.group(1).strip()
    # Everything after </think> is the actual answer
    answer = output[match.end():].strip()
    return answer, reasoning


def _extract_reasoning(output: str) -> tuple[str, str]:
    mistral_match = _MISTRAL_THINK_RE.search(output)
    if mistral_match:
        reasoning = mistral_match.group(1).strip()
        answer = output[mistral_match.end():].strip()
        return answer, reasoning
    return _extract_thinking(output)
