import unittest

from eval_kv_quantized_multiple_choice import (
    build_fewshot_prefix,
    choose_answer,
    clean_hellaswag_text,
    format_task_example,
)


class MultipleChoiceFormattingTest(unittest.TestCase):
    def test_formats_hellaswag(self):
        prompt, choices, gold = format_task_example(
            "hellaswag",
            {
                "ctx": "A person [title] opens a door.",
                "endings": ["walks inside.", "flies away."],
                "label": "0",
            },
        )
        self.assertEqual(prompt, "A person. opens a door.")
        self.assertEqual(choices, [" walks inside.", " flies away."])
        self.assertEqual(gold, 0)

    def test_formats_arc_label_order(self):
        prompt, choices, gold = format_task_example(
            "arc_challenge",
            {
                "question": "Which answer?",
                "choices": {"text": ["first", "second"], "label": ["A", "B"]},
                "answerKey": "B",
            },
        )
        self.assertEqual(prompt, "Question: Which answer?\nAnswer:")
        self.assertEqual(choices, [" first", " second"])
        self.assertEqual(gold, 1)

    def test_choose_answer(self):
        self.assertEqual(choose_answer([-3.0, -1.0, -2.0]), 1)

    def test_builds_fewshot_prefix_with_gold_answers(self):
        prefix = build_fewshot_prefix(
            "arc_challenge",
            [
                {
                    "question": "Pick one.",
                    "choices": {"text": ["wrong", "right"], "label": ["A", "B"]},
                    "answerKey": "B",
                }
            ],
        )
        self.assertEqual(prefix, "Question: Pick one.\nAnswer: right\n\n")

    def test_clean_hellaswag_text(self):
        self.assertEqual(clean_hellaswag_text(" A  [noise]  short text "), "A short text")


if __name__ == "__main__":
    unittest.main()
