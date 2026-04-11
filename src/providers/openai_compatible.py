from __future__ import annotations

import json
from typing import Any

import requests

from src.providers.base import GenerationRequest, GenerationResponse
from src.utils.text import extract_thinking


class OpenAICompatibleProvider:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout_seconds: int = 90,
        default_headers: dict[str, str] | None = None,
        default_body: dict[str, Any] | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self.default_body = default_body or {}

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers)
        return headers

    def _build_body(self, request: GenerationRequest, stream: bool) -> dict[str, Any]:
        body = dict(self.default_body)
        body.update(request.extra_body)
        body["model"] = request.model_name
        body["messages"] = request.messages
        body["stream"] = stream

        options = dict(body.get("options", {}))
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            body["options"] = options
        return body

    def _parse_streaming_response(self, response: requests.Response) -> GenerationResponse:
        full_text = ""
        full_reasoning = ""
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            full_text += delta.get("content", "") or ""
            full_reasoning += delta.get("reasoning_content", "") or ""

        if not full_reasoning:
            full_reasoning = extract_thinking(full_text)
        return GenerationResponse(text=full_text, reasoning=full_reasoning)

    def generate(self, request: GenerationRequest, capture_reasoning: bool = False) -> GenerationResponse:
        stream = capture_reasoning
        response = requests.post(
            self.api_url,
            json=self._build_body(request, stream=stream),
            headers=self._headers(),
            timeout=self.timeout_seconds,
            stream=stream,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

        if stream:
            return self._parse_streaming_response(response)

        if not response.text or response.text.strip() == "null":
            raise RuntimeError("Provider returned an empty response body.")

        body = response.json()
        message = body["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or extract_thinking(text)
        return GenerationResponse(text=text, reasoning=reasoning, raw_response=body)
