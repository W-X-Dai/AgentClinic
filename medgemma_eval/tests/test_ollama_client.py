from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.ollama_client import OllamaClient, parse_choice

OPTIONS = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")


class ParseChoiceTests(unittest.TestCase):
    def test_structured_json(self) -> None:
        self.assertEqual(
            (2, "json_choice"), parse_choice('{"choice":"C","reason":"x"}', OPTIONS)
        )

    def test_truncated_json_keeps_leading_choice(self) -> None:
        content = '{"choice":"D","reason":"long explanation that was truncated'
        self.assertEqual((3, "leading_json_choice"), parse_choice(content, OPTIONS))

    def test_direct_label(self) -> None:
        self.assertEqual((1, "direct_label"), parse_choice("Answer: B", OPTIONS))

    def test_unique_option_text_fallback(self) -> None:
        self.assertEqual(
            (3, "unique_option_text"), parse_choice("I select Delta.", OPTIONS)
        )

    def test_ambiguous_or_invalid_text_is_not_guessed(self) -> None:
        self.assertEqual((None, "unparsed"), parse_choice("Alpha or Beta", OPTIONS))
        self.assertEqual((None, "unparsed"), parse_choice("Answer: F", OPTIONS))

    def test_model_digest_accepts_explicit_or_latest_name(self) -> None:
        client = OllamaClient("http://unused", 1)
        client._request = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "models": [{"name": "medgemma:27b", "digest": "abc123"}]
        }
        self.assertEqual("abc123", client.model_digest("medgemma:27b"))


if __name__ == "__main__":
    unittest.main()
