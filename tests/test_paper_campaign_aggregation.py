import unittest

from spec_kv_statistics import sample_count_status


class PaperCampaignAggregationTest(unittest.TestCase):
    def test_sample_count_status_uses_unique_prompt_ids(self):
        summary = {"config": {"num_prompts": 4}}
        rows = [
            {"prompt_idx": "0", "config": "none"},
            {"prompt_idx": "0", "config": "k4v4"},
            {"prompt_idx": "1", "config": "none"},
            {"prompt_idx": "1", "config": "k4v4"},
        ]
        status = sample_count_status(summary, rows)
        self.assertEqual(status["observed_num_prompts"], 2)
        self.assertEqual(status["prompt_shortfall"], 2)
        self.assertTrue(status["underfilled"])

    def test_sample_count_status_defaults_to_observed_count(self):
        rows = [{"prompt_idx": "7", "config": "none"}]
        status = sample_count_status({}, rows)
        self.assertEqual(status["requested_num_prompts"], 1)
        self.assertFalse(status["underfilled"])


if __name__ == "__main__":
    unittest.main()
