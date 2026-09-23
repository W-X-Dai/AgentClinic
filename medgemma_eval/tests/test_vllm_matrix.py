from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.run_vllm_matrix import (
    build_serve_command,
    detect_vision,
    inspect_model,
    load_manifest,
)


class VLLMMatrixTests(unittest.TestCase):
    def test_a100_manifest_attempts_all_models_without_rtx_workaround(self) -> None:
        manifest = load_manifest(
            AGENTCLINIC_ROOT / "medgemma_eval" / "vllm_models.a100.json"
        )
        models = {model["name"]: model for model in manifest["models"]}
        bf16 = models["nemotron-3-nano-omni-30b-bf16"]
        nvfp4 = models["nemotron-3-nano-omni-30b-nvfp4"]

        self.assertEqual(bf16["max_model_len"], 4096)
        self.assertNotIn("--moe-backend", bf16["serve_args"])
        self.assertTrue(nvfp4.get("enabled", True))
        self.assertEqual(len(manifest["models"]), 11)

    def test_detects_vision_from_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Gemma3ForConditionalGeneration"],
                        "model_type": "gemma3",
                        "vision_config": {"hidden_size": 1},
                    }
                ),
                encoding="utf-8",
            )
            inspection = inspect_model(model)

        self.assertTrue(inspection["has_vision_config"])
        self.assertTrue(detect_vision(model, inspection))

    def test_build_command_has_stable_served_name_and_generation_config(self) -> None:
        manifest = {
            "server": {
                "command": ["vllm", "serve"],
                "host": "127.0.0.1",
                "port": 9000,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.85,
                "max_model_len": 4096,
            }
        }
        model = {
            "name": "local-model",
            "path": "/mnt/models/local-model",
            "serve_args": ["--trust-remote-code"],
        }
        command = build_serve_command(manifest, model)

        self.assertEqual(command[:3], ["vllm", "serve", "/mnt/models/local-model"])
        self.assertIn("--served-model-name", command)
        self.assertIn("--generation-config", command)
        self.assertEqual(command[-1], "--trust-remote-code")

    def test_manifest_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {"name": "same", "path": "/one"},
                            {"name": "same", "path": "/two"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
