from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.analyze_visual_results import _comparison, binomial_upper_p


def result(case_id: str, mode: str, correct: bool, prediction: int) -> dict:
    return {
        "case_id": case_id,
        "mode": mode,
        "correct": correct,
        "predicted_index": prediction,
        "subtypes": ["xray"],
        "task_type": "diagnosis",
        "model": "model",
        "prompt_version": "v3",
        "seed": 1,
        "dataset_sha256": "data",
    }


class VisualAnalysisTests(unittest.TestCase):
    def test_matched_image_gain_counts_paired_flips(self) -> None:
        rows = [
            result("a", "context_only", False, 0),
            result("a", "context_image", True, 1),
            result("b", "context_only", True, 2),
            result("b", "context_image", True, 2),
        ]
        overall = next(
            row
            for row in _comparison(
                rows,
                name="gain",
                baseline_mode="context_only",
                target_mode="context_image",
            )
            if row["scope_kind"] == "overall"
        )
        self.assertEqual(0.5, overall["delta"])
        self.assertEqual(1, overall["helpful"])
        self.assertEqual(0, overall["harmful"])
        self.assertEqual(1, overall["answer_changed"])

    def test_image_only_exact_chance_test(self) -> None:
        self.assertLess(binomial_upper_p(5, 5), 0.001)
        self.assertEqual(1.0, binomial_upper_p(0, 5))


if __name__ == "__main__":
    unittest.main()
