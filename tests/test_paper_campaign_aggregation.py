import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from paper.aggregate_campaign import collect_campaign, pair_sort_key
from paper.campaign_provenance import sample_count_status


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

    def test_collect_campaign_rejects_unrelated_diagnostic_outputs(self):
        with TemporaryDirectory() as tmp_dir:
            results_root = Path(tmp_dir)
            output_dir = results_root / "target_verification_diagnostic" / "wikitext_ctx1024"
            output_dir.mkdir(parents=True)
            (output_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "runtime": {
                            "evaluator_version": "cached_dynamic_v4",
                            "target_cache_reused": True,
                            "draft_cache_reused": True,
                            "target_reference_generation_in_timing": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            metrics, comparisons, rejected = collect_campaign(
                results_root,
                bootstrap_samples=0,
                seed=7,
                tie_tolerance=1e-3,
            )

            self.assertEqual(metrics, [])
            self.assertEqual(comparisons, [])
            self.assertEqual(rejected[0]["reason"], "unrecognized model pair")

    def test_pair_sort_key_keeps_unknown_pairs_after_known_pairs(self):
        pairs = ["target_verification_diagnostic", "qwen25_3b_15b"]
        self.assertEqual(
            sorted(pairs, key=pair_sort_key),
            ["qwen25_3b_15b", "target_verification_diagnostic"],
        )


if __name__ == "__main__":
    unittest.main()
