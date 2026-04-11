from __future__ import annotations

from src.providers.base import GenerationRequest, GenerationResponse


class OpenAIResponsesProvider:
    def __init__(self, api_key: str, timeout_seconds: int = 90, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def generate(self, request: GenerationRequest, capture_reasoning: bool = False) -> GenerationResponse:
        client = self._client_instance()
        kwargs = {
            "model": request.model_name,
            "input": request.messages,
        }
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.reasoning_effort is not None:
            reasoning = {"effort": request.reasoning_effort}
            if capture_reasoning and request.reasoning_effort != "none":
                reasoning["summary"] = "detailed"
            kwargs["reasoning"] = reasoning
        if capture_reasoning and request.reasoning_effort not in (None, "none"):
            kwargs["include"] = ["reasoning.encrypted_content"]

        response = client.responses.create(**kwargs)
        response_dict = response.model_dump()
        text = getattr(response, "output_text", "") or ""
        reasoning = ""
        reasoning_summary = ""
        reasoning_encrypted_content = ""
        reasoning_format = ""
        if capture_reasoning:
            for item in response_dict.get("output", []):
                if item.get("type") == "reasoning":
                    summary = item.get("summary") or []
                    reasoning_summary = "\n".join(
                        chunk.get("text", "")
                        for chunk in summary
                        if isinstance(chunk, dict)
                    ).strip()
                    if reasoning_summary:
                        reasoning = reasoning_summary
                        reasoning_format = "summary_text"
                    else:
                        reasoning_encrypted_content = item.get("encrypted_content") or ""
                        if reasoning_encrypted_content:
                            reasoning = reasoning_encrypted_content
                            reasoning_format = "encrypted_content"
                    break

        return GenerationResponse(
            text=text,
            reasoning=reasoning,
            reasoning_summary=reasoning_summary,
            reasoning_encrypted_content=reasoning_encrypted_content,
            reasoning_format=reasoning_format,
            raw_response=response_dict,
        )
