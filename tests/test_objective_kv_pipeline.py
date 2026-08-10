import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acceptance_risk_statistics import paired_drop_statistics
from aggregate_objective_kv_matrix import classify_exactness
from benchmark_spec_kv_quantization import last_token_logits_kwargs
from kv_cache_quantization import parse_csv_ints, parse_quant_config_specs
from prepare_objective_kv_matrix import heuristic_component_bits


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_profile(path: Path, risk_field: str, risks: dict[tuple[int, str, int], float]) -> None:
    rows = []
    for (layer, component, bits), risk in risks.items():
        rows.append(
            {
                "candidate": f"layer{layer}_{component}{bits}",
                "layer": layer,
                "component": component,
                "bits": bits,
                risk_field: risk,
                "cache_mib_saved": (16 - bits) / 16,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class ObjectiveKVPipelineTest(unittest.TestCase):
    def test_last_token_logits_kwarg_matches_model_api(self) -> None:
        class CurrentModel:
            def forward(self, input_ids, logits_to_keep=None):
                return None

        class LegacyModel:
            def forward(self, input_ids, num_logits_to_keep=None):
                return None

        class UnsupportedModel:
            def forward(self, input_ids):
                return None

        self.assertEqual(last_token_logits_kwargs(CurrentModel()), {"logits_to_keep": 1})
        self.assertEqual(last_token_logits_kwargs(LegacyModel()), {"num_logits_to_keep": 1})
        self.assertEqual(last_token_logits_kwargs(UnsupportedModel()), {})

    def test_matched_memory_heuristics_prioritize_k_or_v(self) -> None:
        self.assertEqual(heuristic_component_bits(4, prioritize="k"), (4, 4))
        self.assertEqual(heuristic_component_bits(4, prioritize="v"), (4, 4))
        self.assertEqual(heuristic_component_bits(6, prioritize="k"), (8, 4))
        self.assertEqual(heuristic_component_bits(6, prioritize="v"), (4, 8))
        self.assertEqual(heuristic_component_bits(10, prioritize="k"), (16, 4))
        self.assertEqual(heuristic_component_bits(12, prioritize="v"), (8, 16))

    def test_paired_acceptance_risk_reports_upper_confidence_bound(self) -> None:
        baseline = [
            {"prompt_idx": 0, "accept_rate": 0.5},
            {"prompt_idx": 1, "accept_rate": 0.8},
        ]
        candidate = [
            {"prompt_idx": 0, "accept_rate": 0.4},
            {"prompt_idx": 1, "accept_rate": 0.6},
        ]
        stats = paired_drop_statistics(baseline, candidate)

        self.assertAlmostEqual(stats["accept_rate_drop_prompt_mean"], 0.15)
        self.assertAlmostEqual(stats["accept_rate_drop_prompt_se"], 0.05)
        self.assertAlmostEqual(stats["accept_rate_drop_ucb95"], 0.248)
        self.assertAlmostEqual(stats["accept_rate_drop_ucb95_clipped"], 0.248)

        mass_stats = paired_drop_statistics(
            [
                {"prompt_idx": 0, "round_accept_mass": 0.9},
                {"prompt_idx": 1, "round_accept_mass": 0.8},
            ],
            [
                {"prompt_idx": 0, "round_accept_mass": 0.8},
                {"prompt_idx": 1, "round_accept_mass": 0.75},
            ],
            metric="round_accept_mass",
            prefix="accept_mass_drop",
        )
        self.assertAlmostEqual(mass_stats["accept_mass_drop_prompt_mean"], 0.075)
        self.assertGreater(mass_stats["accept_mass_drop_ucb95_clipped"], 0.075)

    def test_exactness_classification_distinguishes_numerical_ties(self) -> None:
        exact = {"matches_target_greedy": "1.0", "mismatch_min_top1_margin": "nan"}
        tie = {"matches_target_greedy": "0.0", "mismatch_min_top1_margin": "0.0005"}
        non_tie = {"matches_target_greedy": "0.0", "mismatch_min_top1_margin": "0.125"}
        unknown = {"matches_target_greedy": "0.0", "mismatch_min_top1_margin": "nan"}

        self.assertEqual(classify_exactness(exact, tie_margin=1e-3), "exact")
        self.assertEqual(classify_exactness(tie, tie_margin=1e-3), "numerical_tie")
        self.assertEqual(classify_exactness(non_tie, tie_margin=1e-3), "non_tie_or_unknown")
        self.assertEqual(classify_exactness(unknown, tie_margin=1e-3), "non_tie_or_unknown")

    def test_semicolon_quant_configs(self) -> None:
        configs = parse_quant_config_specs("none;k8v4;k4v8", num_layers=3)
        self.assertEqual([config[0] for config in configs], ["none", "k8v4", "k4v8"])
        self.assertEqual(parse_csv_ints("8;4"), [8, 4])

    def test_fixed_budget_allocator_changes_layout_by_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_csv = root / "quality.csv"
            acceptance_csv = root / "acceptance.csv"
            quality_out = root / "quality_out"
            acceptance_out = root / "acceptance_out"
            quality_risks = {
                (0, "k", 8): 0.01,
                (0, "k", 4): 0.02,
                (0, "v", 8): 0.10,
                (0, "v", 4): 0.30,
            }
            acceptance_risks = {
                (0, "k", 8): 0.10,
                (0, "k", 4): 0.30,
                (0, "v", 8): 0.01,
                (0, "v", 4): 0.02,
            }
            write_profile(quality_csv, "quality_risk", quality_risks)
            write_profile(acceptance_csv, "accept_rate_drop", acceptance_risks)

            for profile, field, out_dir, name in (
                (quality_csv, "quality_risk", quality_out, "quality"),
                (acceptance_csv, "accept_rate_drop", acceptance_out, "acceptance"),
            ):
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "search_kv_bit_allocation.py"),
                        "--profile_csv",
                        str(profile),
                        "--num_layers",
                        "1",
                        "--risk_field",
                        field,
                        "--target_profiled_mean_bits",
                        "10",
                        "--name",
                        name,
                        "--out_dir",
                        str(out_dir),
                    ],
                    check=True,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )

            quality = json.loads((quality_out / "allocation.json").read_text())
            acceptance = json.loads((acceptance_out / "allocation.json").read_text())
            self.assertEqual(quality["achieved_profiled_mean_bits"], 10.0)
            self.assertEqual(acceptance["achieved_profiled_mean_bits"], 10.0)
            self.assertNotEqual((quality["k_bits"], quality["v_bits"]), (acceptance["k_bits"], acceptance["v_bits"]))

    def test_missing_risk_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.csv"
            write_profile(profile, "accept_rate_drop", {(0, "k", 4): 0.1})
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "search_kv_bit_allocation.py"),
                    "--profile_csv",
                    str(profile),
                    "--risk_field",
                    "quality_risk",
                    "--out_dir",
                    str(root / "out"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Risk field", result.stderr)


if __name__ == "__main__":
    unittest.main()
