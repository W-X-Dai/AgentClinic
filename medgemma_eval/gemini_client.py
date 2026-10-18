"""Gemini Developer API client for the AgentClinic visual benchmark."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ollama_client import CHOICE_SCHEMA

# Gemini 3.5+ is configured only through ``thinking_config.thinking_level``;
# ``temperature``/``top_p``/``top_k`` and ``thinking_budget`` are deprecated there.
THINKING_LEVELS = ("minimal", "low", "medium", "high")
DEFAULT_THINKING_LEVEL = "low"
THINKING_LEVEL_ENV = "AGENTCLINIC_GEMINI_THINKING_LEVEL"
_THINKING_LEVEL_ONLY_FROM = (3, 5)
_MIN_OUTPUT_TOKENS = 256
# Thinking tokens share ``max_output_tokens`` with the short JSON answer; these
# floors keep the answer from being truncated at each level.
_THINKING_OUTPUT_FLOOR = {"minimal": 512, "low": 1024, "medium": 2048, "high": 4096}


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


def validate_thinking_level(value: str | None) -> str:
    """Normalize a thinking level, defaulting to ``DEFAULT_THINKING_LEVEL``."""
    normalized = (value or "").strip().lower() or DEFAULT_THINKING_LEVEL
    if normalized not in THINKING_LEVELS:
        raise ValueError(
            f"thinking level must be one of {', '.join(THINKING_LEVELS)}; got {value!r}"
        )
    return normalized


def uses_thinking_level(model: str) -> bool:
    """Return True for Gemini 3.5+ model names (``gemini-3.5-*``, ``gemini-4``)."""
    match = re.match(r"^(?:models/)?gemini-(\d+)(?:\.(\d+))?", model.lower())
    if not match:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= _THINKING_LEVEL_ONLY_FROM


def generation_settings(
    model: str, thinking_level: str | None, max_output_tokens: int
) -> dict[str, Any]:
    """Describe the generation settings actually sent for one request.

    Older Gemini models keep the previous request (no sampling or thinking
    fields), so earlier runs remain reproducible. No model receives temperature.
    """
    settings: dict[str, Any] = {
        "max_output_tokens": max(_MIN_OUTPUT_TOKENS, max_output_tokens)
    }
    if uses_thinking_level(model):
        level = validate_thinking_level(thinking_level)
        settings["thinking_level"] = level
        settings["max_output_tokens"] = max(
            settings["max_output_tokens"], _THINKING_OUTPUT_FLOOR[level]
        )
    return settings


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
        thinking_level: str | None = None,
    ) -> GeminiResponse:
        types = self._types
        parts: list[Any] = [types.Part.from_text(text=user_prompt)]
        if image is not None:
            suffix = image.suffix.lower()
            mime_type = "image/png" if suffix == ".png" else "image/jpeg"
            parts.append(
                types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_type)
            )

        settings = generation_settings(model, thinking_level, max_output_tokens)
        thinking_config = (
            types.ThinkingConfig(thinking_level=settings["thinking_level"])
            if "thinking_level" in settings
            else None
        )
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=settings["max_output_tokens"],
            thinking_config=thinking_config,
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
