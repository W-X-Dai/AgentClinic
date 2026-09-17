"""Minimal dependency-free Ollama client with structured-choice parsing."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
    },
    "required": ["choice"],
}


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    latency_seconds: float
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    eval_count: int | None


class OllamaClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error

    def version(self) -> str:
        return str(self._request("GET", "/api/version")["version"])

    def model_info(self, model: str) -> dict[str, Any]:
        return dict(self._request("POST", "/api/show", {"model": model}))

    def model_digest(self, model: str) -> str | None:
        response = self._request("GET", "/api/tags")
        candidates = response.get("models") if isinstance(response, dict) else None
        if not isinstance(candidates, list):
            return None
        requested = model if ":" in model else f"{model}:latest"
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            names = {str(candidate.get("name", "")), str(candidate.get("model", ""))}
            if model in names or requested in names:
                digest = candidate.get("digest")
                return str(digest) if digest else None
        return None

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image: Path | None,
        seed: int,
        num_predict: int,
        keep_alive: str,
    ) -> OllamaResponse:
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image is not None:
            user_message["images"] = [
                base64.b64encode(image.read_bytes()).decode("ascii")
            ]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "stream": False,
            "format": CHOICE_SCHEMA,
            "keep_alive": keep_alive,
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        started = time.monotonic()
        response = self._request("POST", "/api/chat", payload)
        latency = time.monotonic() - started
        return OllamaResponse(
            content=str(response["message"]["content"]),
            latency_seconds=latency,
            total_duration_ns=response.get("total_duration"),
            load_duration_ns=response.get("load_duration"),
            prompt_eval_count=response.get("prompt_eval_count"),
            eval_count=response.get("eval_count"),
        )


def parse_choice(content: str, options: tuple[str, ...]) -> tuple[int | None, str]:
    """Parse a choice conservatively and report the successful strategy."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        choice = value.get("choice")
        if isinstance(choice, str) and re.fullmatch(r"[A-Ea-e]", choice.strip()):
            return ord(choice.strip().upper()) - ord("A"), "json_choice"

    truncated_json = re.match(
        r'^\s*\{\s*["\']choice["\']\s*:\s*["\']([A-Ea-e])["\'](?:\s*[,}])',
        content,
    )
    if truncated_json:
        return ord(truncated_json.group(1).upper()) - ord("A"), "leading_json_choice"

    stripped = content.strip()
    direct = re.fullmatch(
        r"(?:answer\s*[:=-]?\s*)?\(?([A-Ea-e])\)?[.]?", stripped, re.IGNORECASE
    )
    if direct:
        return ord(direct.group(1).upper()) - ord("A"), "direct_label"

    answer_match = re.search(
        r"(?:answer|choice)\s*(?:is|[:=-])\s*\(?([A-Ea-e])\)?\b",
        content,
        re.IGNORECASE,
    )
    if answer_match:
        return ord(answer_match.group(1).upper()) - ord("A"), "label_pattern"

    normalized_content = " ".join(re.sub(r"[^\w]+", " ", content.lower()).split())
    matches = []
    for index, option in enumerate(options):
        normalized_option = " ".join(re.sub(r"[^\w]+", " ", option.lower()).split())
        if normalized_option and normalized_option in normalized_content:
            matches.append(index)
    if len(matches) == 1:
        return matches[0], "unique_option_text"
    return None, "unparsed"
