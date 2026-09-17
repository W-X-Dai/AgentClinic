from __future__ import annotations

import collections
import sys
import unittest
from pathlib import Path

AGENTCLINIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTCLINIC_ROOT))

from medgemma_eval.dataset import (
    classify_task,
    final_question,
    load_cases,
    mismatched_image_case,
    permuted_options,
    select_cases,
)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(AGENTCLINIC_ROOT / "agentclinic_nejm_extended.jsonl")

    def test_extended_dataset_contract(self) -> None:
        self.assertEqual(120, len(self.cases))
        self.assertEqual(120, len({case.case_id for case in self.cases}))
        self.assertTrue(all(len(case.options) == 5 for case in self.cases))
        self.assertTrue(all(0 <= case.correct_index < 5 for case in self.cases))

        subtype_counts = collections.Counter(
            subtype for case in self.cases for subtype in case.subtypes
        )
        self.assertEqual(13, len(subtype_counts))
        self.assertEqual(51, subtype_counts["physical"])
        self.assertEqual(19, subtype_counts["CT scan"])
        self.assertEqual(1, subtype_counts["echocardiogram"])

    def test_direct_diagnosis_subset_is_explicit(self) -> None:
        counts = collections.Counter(case.task_type for case in self.cases)
        self.assertEqual(90, counts["diagnosis"])
        self.assertEqual("test_or_association", self.cases[30].task_type)
        self.assertEqual("test_or_association", self.cases[45].task_type)
        self.assertEqual("test_or_association", self.cases[46].task_type)

    def test_task_classifier_does_not_need_gold_answer(self) -> None:
        self.assertEqual(
            "treatment", classify_task("Which treatment is most appropriate?")
        )
        self.assertEqual("etiology", classify_task("What is the most likely pathogen?"))
        self.assertEqual("diagnosis", classify_task("What is the diagnosis?"))

    def test_final_question_excludes_image_description(self) -> None:
        self.assertEqual(
            "Which of the following is the most likely diagnosis?",
            final_question(self.cases[2].question),
        )
        self.assertNotIn("crazy paving", self.cases[2].question_target.lower())

    def test_patient_context_does_not_contain_exact_gold_answer(self) -> None:
        self.assertTrue(
            all(
                case.correct_answer.casefold() not in case.clinical_context.casefold()
                for case in self.cases
            )
        )

    def test_mismatched_image_is_deterministic_and_subtype_matched(self) -> None:
        case = self.cases[2]
        first = mismatched_image_case(self.cases, case, 123)
        second = mismatched_image_case(self.cases, case, 123)
        self.assertEqual(first.case_id, second.case_id)
        self.assertNotEqual(case.case_id, first.case_id)
        self.assertTrue(set(case.subtypes).intersection(first.subtypes))

    def test_option_permutation_is_deterministic_and_preserves_gold(self) -> None:
        case = self.cases[0]
        first = permuted_options(case, 123)
        second = permuted_options(case, 123)
        self.assertEqual(first, second)
        options, displayed_correct, order = first
        self.assertEqual(case.correct_answer, options[displayed_correct])
        self.assertEqual(set(range(5)), set(order))

    def test_all_subtype_smoke_indices_cover_every_tag(self) -> None:
        smoke = select_cases(
            self.cases,
            set(),
            set(),
            {21, 30, 40, 45, 53, 100, 105},
            None,
        )
        all_tags = {tag for case in self.cases for tag in case.subtypes}
        smoke_tags = {tag for case in smoke for tag in case.subtypes}
        self.assertEqual(all_tags, smoke_tags)


if __name__ == "__main__":
    unittest.main()
