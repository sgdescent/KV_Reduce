import unittest

from aggregate_kv_passkey import paired_accuracy_differences, parse_int_list


class PasskeyAggregationTest(unittest.TestCase):
    def test_parses_integer_list(self):
        self.assertEqual(parse_int_list("4096, 8192,16384"), [4096, 8192, 16384])

    def test_computes_context_and_depth_matched_differences(self):
        rows = [
            {
                "prompt_tokens": "4096",
                "passkey_depth": "0.1",
                "seed": "0",
                "source_idx": "4",
                "config": "none",
                "raw_correct": "1",
            },
            {
                "prompt_tokens": "4096",
                "passkey_depth": "0.1",
                "seed": "0",
                "source_idx": "4",
                "config": "k4v4",
                "raw_correct": "0",
            },
            {
                "prompt_tokens": "8192",
                "passkey_depth": "0.1",
                "seed": "0",
                "source_idx": "4",
                "config": "none",
                "raw_correct": "0",
            },
            {
                "prompt_tokens": "8192",
                "passkey_depth": "0.1",
                "seed": "0",
                "source_idx": "4",
                "config": "k4v4",
                "raw_correct": "1",
            },
        ]
        values = paired_accuracy_differences(
            rows,
            config="k4v4",
            baseline="none",
            context=4096,
            depth=0.1,
        )
        self.assertEqual(values, [-1.0])


if __name__ == "__main__":
    unittest.main()
