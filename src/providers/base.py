from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class GenerationRequest:
    model_name: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_enabled: bool = False
    extra_body: dict = field(default_factory=dict)
    reasoning_effort: str | None = None


@dataclass(slots=True)
class GenerationResponse:
    text: str
    reasoning: str = ""
    reasoning_summary: str = ""
    reasoning_encrypted_content: str = ""
    reasoning_format: str = ""
    raw_response: dict | None = None


class Provider(Protocol):
    def generate(self, request: GenerationRequest, capture_reasoning: bool = False) -> GenerationResponse:
        ...
