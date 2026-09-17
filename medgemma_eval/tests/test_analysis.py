from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.analyze_results import (
    aggregate,
    mcnemar_exact_p,
    paired,
    validate_comparable_config,
    wilson_interval,
)


def record(
    case_id: str, mode: str, correct: bool, status: str, subtypes: list[str]
) -> dict:
    return {
        "case_id": case_id,
        "mode": mode,
        "correct": correct,
        "status": status,
        "subtypes": subtypes,
        "task_type": "diagnosis",
        "model": "model",
        "prompt_version": "v1",
        "seed": 1,
        "dataset_sha256": "data",
    }


class AnalysisTests(unittest.TestCase):
    def test_failures_remain_in_accuracy_denominator(self) -> None:
        rows = [
            record("a", "multimodal", True, "ok", ["physical", "xray"]),
            record("b", "multimodal", False, "error", ["physical"]),
        ]
        overall = next(row for row in aggregate(rows) if row["scope_kind"] == "overall")
        self.assertEqual(2, overall["n"])
        self.assertEqual(0.5, overall["accuracy"])
        self.assertEqual(0.5, overall["scorable_rate"])
        subtype_rows = [
            row for row in aggregate(rows) if row["scope_kind"] == "subtype"
        ]
        self.assertEqual(
            {"physical", "xray"}, {row["scope_value"] for row in subtype_rows}
        )

    def test_paired_flip_counts(self) -> None:
        rows = [
            record("a", "text_only", False, "ok", ["physical"]),
            record("a", "multimodal", True, "ok", ["physical"]),
            record("b", "text_only", True, "ok", ["physical"]),
            record("b", "multimodal", False, "parse_error", ["physical"]),
        ]
        overall = next(row for row in paired(rows) if row["scope_kind"] == "overall")
        self.assertEqual(1, overall["image_helpful"])
        self.assertEqual(1, overall["image_harmful"])
        self.assertEqual(0.0, overall["vision_delta"])

    def test_wilson_interval_contains_observed_accuracy(self) -> None:
        low, high = wilson_interval(7, 10)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)

    def test_exact_mcnemar_uses_only_discordant_pairs(self) -> None:
        self.assertAlmostEqual(0.7265625, mcnemar_exact_p(3, 5))
        self.assertEqual(1.0, mcnemar_exact_p(0, 0))

    def test_mixed_prompt_versions_are_rejected(self) -> None:
        rows = [
            record("a", "text_only", True, "ok", ["physical"]),
            record("b", "text_only", True, "ok", ["physical"]),
        ]
        rows[1]["prompt_version"] = "v2"
        with self.assertRaisesRegex(ValueError, "multiple prompt"):
            validate_comparable_config(rows)


if __name__ == "__main__":
    unittest.main()
