from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.compare_visual_models import model_metrics, pair_records


def _row(model: str, case_id: str, mode: str, correct: bool) -> dict[str, object]:
    return {
        "model": model,
        "case_id": case_id,
        "mode": mode,
        "prompt_version": "v1",
        "seed": 1,
        "dataset_sha256": "dataset",
        "displayed_correct_index": 0,
        "displayed_options": ["a", "b", "c", "d", "e"],
        "option_order_original_indices": [0, 1, 2, 3, 4],
        "prompt_sha256": f"prompt-{case_id}-{mode}",
        "image_case_id": case_id if mode != "context_only" else None,
        "image_sha256": f"image-{case_id}" if mode != "context_only" else None,
        "subtypes": ["xray"],
        "status": "ok",
        "correct": correct,
    }


class ModelComparisonTests(unittest.TestCase):
    def test_candidate_helpful_and_harmful_are_paired(self) -> None:
        baseline = []
        candidate = []
        for mode in (
            "context_only",
            "context_image",
            "context_mismatched",
            "image_only",
        ):
            baseline.extend(
                [
                    _row("baseline", "one", mode, False),
                    _row("baseline", "two", mode, True),
                ]
            )
            candidate.extend(
                [
                    _row("candidate", "one", mode, True),
                    _row("candidate", "two", mode, False),
                ]
            )

        metrics = model_metrics(pair_records(baseline, candidate))
        overall = next(
            row
            for row in metrics
            if row["scope_kind"] == "overall" and row["mode"] == "context_image"
        )
        self.assertEqual(overall["candidate_helpful"], 1)
        self.assertEqual(overall["candidate_harmful"], 1)
        self.assertEqual(overall["candidate_delta"], 0)

    def test_pairing_rejects_different_prompts(self) -> None:
        baseline = [_row("baseline", "one", "context_only", True)]
        candidate = [_row("candidate", "one", "context_only", True)]
        candidate[0]["prompt_sha256"] = "different"

        with self.assertRaisesRegex(ValueError, "prompt_sha256"):
            pair_records(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
