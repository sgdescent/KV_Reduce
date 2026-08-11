import unittest

import torch

from eval_kv_quantized_multiple_choice import (
    assemble_passkey_prompt_ids,
    build_fewshot_prefix,
    choose_answer,
    clean_hellaswag_text,
    format_task_example,
    generate_passkey_examples,
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

    def test_generates_deterministic_passkeys_at_multiple_depths(self):
        first = generate_passkey_examples(
            num_examples=4,
            skip_examples=0,
            dataset_seed=17,
        )
        second = generate_passkey_examples(
            num_examples=4,
            skip_examples=0,
            dataset_seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual([row[1]["depth"] for row in first], [0.1, 0.5, 0.9, 0.1])
        for _source_idx, row in first:
            self.assertEqual(len(row["choices"]), 4)
            self.assertEqual(row["choices"][row["gold"]], row["passkey"])

    def test_generates_harder_sixteen_way_passkeys(self):
        examples = generate_passkey_examples(
            num_examples=3,
            skip_examples=0,
            dataset_seed=17,
            num_choices=16,
        )
        for _source_idx, row in examples:
            self.assertEqual(len(row["choices"]), 16)
            self.assertEqual(len(set(row["choices"])), 16)
            self.assertEqual(row["choices"][row["gold"]], row["passkey"])

    def test_rejects_degenerate_passkey_choices(self):
        with self.assertRaises(ValueError):
            generate_passkey_examples(
                num_examples=1,
                skip_examples=0,
                dataset_seed=17,
                num_choices=1,
            )

    def test_assembles_exact_length_passkey_prompt(self):
        prompt = assemble_passkey_prompt_ids(
            prefix_ids=torch.tensor([1, 1]),
            filler_ids=torch.tensor([2, 3]),
            key_ids=torch.tensor([9, 9]),
            query_ids=torch.tensor([8]),
            target_tokens=11,
            depth=0.5,
        )
        self.assertEqual(tuple(prompt.shape), (1, 11))
        self.assertEqual(prompt[0, -1].item(), 8)
        key_positions = (prompt[0] == 9).nonzero(as_tuple=False).flatten().tolist()
        self.assertEqual(key_positions, [5, 6])

    def test_formats_passkey_choices(self):
        prompt, choices, gold = format_task_example(
            "passkey",
            {"choices": ["123456", "654321"], "gold": 1},
        )
        self.assertIn("pass key", prompt)
        self.assertEqual(choices, [" 123456", " 654321"])
        self.assertEqual(gold, 1)

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
