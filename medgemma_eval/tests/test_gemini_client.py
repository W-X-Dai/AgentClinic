from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.gemini_client import load_env_file
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


if __name__ == "__main__":
    unittest.main()
