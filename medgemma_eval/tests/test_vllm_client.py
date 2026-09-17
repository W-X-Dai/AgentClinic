from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.vllm_client import VLLMClient


class FakeVLLMClient(VLLMClient):
    def __init__(self) -> None:
        super().__init__("http://localhost:8000", 10)
        self.payload: dict[str, Any] | None = None

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        self.payload = body
        return {
            "id": "response-1",
            "model": "vision-model",
            "choices": [
                {
                    "message": {"content": '{"choice":"C"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }


class VLLMClientTests(unittest.TestCase):
    def test_multimodal_payload_uses_data_url_and_json_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "case.jpg"
            image.write_bytes(b"image-bytes")
            client = FakeVLLMClient()
            response = client.chat(
                model="vision-model",
                system_prompt="system",
                user_prompt="question",
                image=image,
                seed=7,
                max_tokens=32,
                response_format="json_schema",
            )

        assert client.payload is not None
        content = client.payload["messages"][1]["content"]
        expected = base64.b64encode(b"image-bytes").decode("ascii")
        self.assertEqual(
            content[1]["image_url"]["url"], f"data:image/jpeg;base64,{expected}"
        )
        self.assertEqual(client.payload["response_format"]["type"], "json_schema")
        self.assertEqual(response.content, '{"choice":"C"}')
        self.assertEqual(response.total_tokens, 14)

    def test_text_payload_does_not_include_image_content(self) -> None:
        client = FakeVLLMClient()
        client.chat(
            model="text-model",
            system_prompt="system",
            user_prompt="question",
            image=None,
            seed=7,
            max_tokens=32,
            response_format="none",
        )

        assert client.payload is not None
        self.assertEqual(client.payload["messages"][1]["content"], "question")
        self.assertNotIn("response_format", client.payload)


if __name__ == "__main__":
    unittest.main()
