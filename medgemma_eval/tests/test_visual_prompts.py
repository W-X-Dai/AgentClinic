from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.dataset import load_cases, permuted_options
from medgemma_eval.run_benchmark import _prompt


class VisualPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_cases(AGENTCLINIC_ROOT / "agentclinic_nejm_extended.jsonl")[2]
        cls.options, _, _ = permuted_options(cls.case, 123)

    def test_context_prompt_omits_original_image_report(self) -> None:
        prompt = _prompt(self.case, self.options, "context_only")
        self.assertIn(self.case.clinical_context, prompt)
        self.assertIn(self.case.question_target, prompt)
        self.assertNotIn("crazy paving", prompt.lower())

    def test_image_only_prompt_has_task_but_no_history(self) -> None:
        prompt = _prompt(self.case, self.options, "image_only")
        self.assertIn(self.case.question_target, prompt)
        self.assertNotIn(self.case.clinical_context, prompt)

    def test_matched_and_mismatched_prompts_are_identical(self) -> None:
        self.assertEqual(
            _prompt(self.case, self.options, "context_image"),
            _prompt(self.case, self.options, "context_mismatched"),
        )


if __name__ == "__main__":
    unittest.main()
