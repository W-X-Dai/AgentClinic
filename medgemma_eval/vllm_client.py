"""Dependency-free client for a vLLM OpenAI-compatible chat server."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ollama_client import CHOICE_SCHEMA


@dataclass(frozen=True)
class VLLMResponse:
    content: str
    latency_seconds: float
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_model: str | None
    response_id: str | None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


class VLLMClient:
    def __init__(self, base_url: str, timeout: float, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key.strip()

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            if self.api_key:
                detail = detail.replace(self.api_key, "<redacted>")
            raise RuntimeError(f"vLLM HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"vLLM request failed: {error}") from error
        return json.loads(payload) if payload else None

    def health(self) -> bool:
        try:
            self._request("GET", "/health")
        except (RuntimeError, json.JSONDecodeError):
            return False
        return True

    def models(self) -> list[str]:
        response = self._request("GET", "/v1/models")
        values = response.get("data") if isinstance(response, dict) else None
        if not isinstance(values, list):
            raise TypeError("vLLM /v1/models returned an invalid payload")
        return [
            str(item["id"])
            for item in values
            if isinstance(item, dict) and item.get("id")
        ]

    def version(self) -> str | None:
        try:
            response = self._request("GET", "/version")
        except RuntimeError:
            return None
        if not isinstance(response, dict):
            return None
        value = response.get("version") or response.get("vllm_version")
        return str(value) if value else None

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image: Path | None,
        seed: int,
        max_tokens: int,
        response_format: str,
    ) -> VLLMResponse:
        if image is None:
            user_content: str | list[dict[str, Any]] = user_prompt
        else:
            suffix = image.suffix.lower()
            mime_type = "image/png" if suffix == ".png" else "image/jpeg"
            data_url = f"data:{mime_type};base64," + base64.b64encode(
                image.read_bytes()
            ).decode("ascii")
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        if response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "choice_response",
                    "schema": CHOICE_SCHEMA,
                    "strict": True,
                },
            }
        elif response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif response_format != "none":
            raise ValueError(f"unsupported response format: {response_format}")

        started = time.monotonic()
        response = self._request("POST", "/v1/chat/completions", payload)
        latency = time.monotonic() - started
        if not isinstance(response, dict):
            raise TypeError("vLLM chat completion returned an invalid payload")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vLLM chat completion returned no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not content or not str(content).strip():
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            raise RuntimeError(f"vLLM returned no content (finish_reason={reason})")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        return VLLMResponse(
            content=str(content).strip(),
            latency_seconds=latency,
            finish_reason=str(choice.get("finish_reason"))
            if isinstance(choice, dict) and choice.get("finish_reason") is not None
            else None,
            prompt_tokens=_integer(usage.get("prompt_tokens")),
            completion_tokens=_integer(usage.get("completion_tokens")),
            total_tokens=_integer(usage.get("total_tokens")),
            response_model=str(response.get("model"))
            if response.get("model")
            else None,
            response_id=str(response.get("id")) if response.get("id") else None,
        )
