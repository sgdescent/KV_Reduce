import unittest

import torch
from transformers import DynamicCache

from eval_kv_quantized_multiple_choice import (
    PASSKEY_VARIANT_CONFUSABLE,
    assemble_confusable_passkey_prompt_ids,
    assemble_passkey_prompt_ids,
    build_fewshot_prefix,
    choose_answer,
    clean_hellaswag_text,
    clone_cache,
    format_task_example,
    generate_passkey_examples,
    prepare_prefill_cache,
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

    def test_dynamic_cache_clone_is_independent_without_conversion(self):
        legacy = tuple(
            (torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))
            for _ in range(3)
        )
        cache = DynamicCache(ddp_cache_data=legacy)
        cloned = clone_cache(cache)

        self.assertIsInstance(cloned, DynamicCache)
        self.assertNotEqual(cloned.layers[0].keys.data_ptr(), cache.layers[0].keys.data_ptr())
        cloned.crop(5)
        self.assertEqual(cache.get_seq_length(), 8)
        self.assertEqual(cloned.get_seq_length(), 5)

    def test_full_precision_prefill_reuses_rewindable_cache(self):
        legacy = ((torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)),)
        cache = DynamicCache(ddp_cache_data=legacy)

        prepared = prepare_prefill_cache(
            cache,
            k_bits=[16],
            v_bits=[16],
            key_quant_axis="per_channel",
            key_group_size=4,
            key_residual_length=4,
            value_quant_scheme="affine",
        )

        self.assertIs(prepared, cache)

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

    def test_generates_confusable_associative_passkeys(self):
        examples = generate_passkey_examples(
            num_examples=3,
            skip_examples=0,
            dataset_seed=17,
            num_choices=16,
            variant=PASSKEY_VARIANT_CONFUSABLE,
        )
        for _source_idx, row in examples:
            self.assertEqual(row["variant"], PASSKEY_VARIANT_CONFUSABLE)
            self.assertEqual(len(row["choices"]), 16)
            self.assertEqual(len(row["distractor_records"]), 15)
            self.assertEqual(row["choices"][row["gold"]], row["passkey"])
            self.assertEqual(
                len({record["tag"] for record in row["distractor_records"]} | {row["target_tag"]}),
                16,
            )
            for choice in row["choices"]:
                if choice != row["passkey"]:
                    self.assertEqual(
                        sum(left != right for left, right in zip(choice, row["passkey"])),
                        1,
                    )

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

    def test_assembles_exact_length_confusable_passkey_prompt(self):
        prompt = assemble_confusable_passkey_prompt_ids(
            prefix_ids=torch.tensor([1, 1]),
            filler_ids=torch.tensor([2, 3]),
            target_record_ids=torch.tensor([9, 9]),
            distractor_record_ids=[torch.tensor([4]), torch.tensor([5])],
            query_ids=torch.tensor([8]),
            target_tokens=15,
            depth=0.5,
        )
        self.assertEqual(tuple(prompt.shape), (1, 15))
        self.assertEqual(prompt[0, -1].item(), 8)
        self.assertEqual((prompt[0] == 9).sum().item(), 2)

    def test_formats_passkey_choices(self):
        prompt, choices, gold = format_task_example(
            "passkey",
            {"choices": ["123456", "654321"], "gold": 1},
        )
        self.assertIn("pass key", prompt)
        self.assertEqual(choices, [" 123456", " 654321"])
        self.assertEqual(gold, 1)

    def test_formats_longbench_passage_retrieval(self):
        context = "\n\n".join(
            f"Paragraph {idx}: passage {idx}." for idx in range(1, 31)
        )
        prompt, choices, gold = format_task_example(
            "longbench_passage_retrieval",
            {
                "context": context,
                "input": "A summary of passage 17.",
                "answers": ["Paragraph 17"],
            },
        )
        self.assertIn("Here are 30 paragraphs", prompt)
        self.assertIn("A summary of passage 17.", prompt)
        self.assertEqual(len(choices), 30)
        self.assertEqual(choices[16], " Paragraph 17")
        self.assertEqual(gold, 16)

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
