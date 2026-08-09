import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kv_cache_quantization import parse_quant_config_specs


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
    def test_semicolon_quant_configs(self) -> None:
        configs = parse_quant_config_specs("none;k8v4;k4v8", num_layers=3)
        self.assertEqual([config[0] for config in configs], ["none", "k8v4", "k4v8"])

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
