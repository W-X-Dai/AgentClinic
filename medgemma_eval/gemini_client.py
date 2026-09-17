"""Gemini Developer API client for the AgentClinic visual benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ollama_client import CHOICE_SCHEMA


@dataclass(frozen=True)
class GeminiResponse:
    content: str
    latency_seconds: float
    finish_reason: str
    prompt_token_count: int | None
    candidates_token_count: int | None
    total_token_count: int | None
    model_version: str | None
    response_id: str | None


def load_env_file(path: Path) -> dict[str, str]:
    """Read a simple dotenv file without logging or exporting its secrets."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "UNKNOWN"
    reason = getattr(candidates[0], "finish_reason", None)
    value = getattr(reason, "value", None)
    return str(value or reason or "UNKNOWN").split(".")[-1]


def _usage_value(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None) if usage is not None else None
    return int(value) if isinstance(value, int) else None


class GeminiClient:
    """Small multimodal wrapper matching the backend's google-genai behavior."""

    def __init__(self, api_key: str, timeout: float) -> None:
        if not api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is missing")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is unavailable; use backend/venv/bin/python"
            ) from exc
        self._types = types
        self._client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    def model_info(self, model: str) -> dict[str, Any]:
        info = self._client.models.get(model=model)
        return {
            "name": getattr(info, "name", None),
            "display_name": getattr(info, "display_name", None),
            "version": getattr(info, "version", None),
            "input_token_limit": getattr(info, "input_token_limit", None),
            "output_token_limit": getattr(info, "output_token_limit", None),
            "supported_actions": getattr(info, "supported_actions", None),
        }

    def generate_choice(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image: Path | None,
        seed: int,
        max_output_tokens: int,
    ) -> GeminiResponse:
        types = self._types
        parts: list[Any] = [types.Part.from_text(text=user_prompt)]
        if image is not None:
            suffix = image.suffix.lower()
            mime_type = "image/png" if suffix == ".png" else "image/jpeg"
            parts.append(
                types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_type)
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max(256, max_output_tokens),
            seed=seed,
            response_mime_type="application/json",
            response_json_schema=CHOICE_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        started = time.monotonic()
        response = self._client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )
        latency = time.monotonic() - started
        content = str(response.text or "").strip()
        reason = _finish_reason(response)
        if not content:
            raise RuntimeError(f"Gemini returned no text (finish_reason={reason})")
        usage = getattr(response, "usage_metadata", None)
        return GeminiResponse(
            content=content,
            latency_seconds=latency,
            finish_reason=reason,
            prompt_token_count=_usage_value(usage, "prompt_token_count"),
            candidates_token_count=_usage_value(usage, "candidates_token_count"),
            total_token_count=_usage_value(usage, "total_token_count"),
            model_version=getattr(response, "model_version", None),
            response_id=getattr(response, "response_id", None),
        )
