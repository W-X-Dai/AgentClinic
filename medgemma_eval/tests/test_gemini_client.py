from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.gemini_client import (
    GeminiClient,
    generation_settings,
    load_env_file,
    validate_thinking_level,
)
from medgemma_eval.run_gemini_benchmark import _safe_error


class GeminiConfigurationTests(unittest.TestCase):
    def test_load_env_file_handles_quotes_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# ignored\nexport GEMINI_API_KEY='secret-value'\n"
                'GEMINI_MODEL="gemini-test"\n',
                encoding="utf-8",
            )
            values = load_env_file(path)

        self.assertEqual(values["GEMINI_API_KEY"], "secret-value")
        self.assertEqual(values["GEMINI_MODEL"], "gemini-test")

    def test_api_key_is_redacted_from_errors(self) -> None:
        message = _safe_error(
            RuntimeError("request rejected for key-secret"), "key-secret"
        )
        self.assertNotIn("key-secret", message)
        self.assertIn("<redacted>", message)


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text='{"answer": "A"}',
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=None,
            model_version=None,
            response_id=None,
        )


def _offline_client() -> tuple[GeminiClient, _FakeModels]:
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - needs backend/venv
        raise unittest.SkipTest("google-genai is unavailable") from exc
    models = _FakeModels()
    client = GeminiClient.__new__(GeminiClient)
    client._types = types
    client._client = SimpleNamespace(models=models)
    return client, models


def _generate(client: GeminiClient, model: str, **kwargs) -> None:
    client.generate_choice(
        model=model,
        system_prompt="synthetic system",
        user_prompt="synthetic question",
        image=None,
        seed=7,
        max_output_tokens=kwargs.pop("max_output_tokens", 300),
        **kwargs,
    )


class GeminiRequestConfigTests(unittest.TestCase):
    def test_newer_model_sends_only_thinking_level(self) -> None:
        client, models = _offline_client()
        _generate(client, "gemini-3.6-flash")

        config = models.calls[0]["config"]
        sent = config.model_dump(exclude_none=True)
        for deprecated in ("temperature", "top_p", "top_k"):
            self.assertNotIn(deprecated, sent)
        self.assertEqual(
            sent["thinking_config"],
            {"thinking_level": config.thinking_config.thinking_level},
        )
        self.assertEqual(
            str(config.thinking_config.thinking_level), "ThinkingLevel.LOW"
        )
        self.assertEqual(config.max_output_tokens, 1024)
        self.assertIsNotNone(config.response_json_schema)

    def test_configured_level_raises_output_floor(self) -> None:
        client, models = _offline_client()
        _generate(client, "gemini-3.5-pro", thinking_level="high")

        config = models.calls[0]["config"]
        self.assertEqual(
            str(config.thinking_config.thinking_level), "ThinkingLevel.HIGH"
        )
        self.assertEqual(config.max_output_tokens, 4096)

    def test_older_model_request_is_unchanged(self) -> None:
        client, models = _offline_client()
        _generate(client, "gemini-2.5-flash", thinking_level="high")

        config = models.calls[0]["config"]
        self.assertIsNone(config.thinking_config)
        self.assertIsNone(config.temperature)
        self.assertEqual(config.max_output_tokens, 300)

    def test_generation_settings_and_validation(self) -> None:
        self.assertEqual(
            generation_settings("gemini-3.6-flash", None, 2000),
            {"max_output_tokens": 2000, "thinking_level": "low"},
        )
        self.assertEqual(
            generation_settings("gemini-2.5-flash", "high", 100),
            {"max_output_tokens": 256},
        )
        self.assertEqual(validate_thinking_level(" Minimal "), "minimal")
        with self.assertRaises(ValueError):
            validate_thinking_level("auto")


if __name__ == "__main__":
    unittest.main()
