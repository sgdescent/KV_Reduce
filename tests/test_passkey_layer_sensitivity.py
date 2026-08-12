import json
import tempfile
import unittest
from pathlib import Path

from aggregate_passkey_layer_sensitivity import paired_component_effects
from prepare_passkey_layer_sensitivity import parse_layers, prepare_allocations


class PasskeyLayerSensitivityTest(unittest.TestCase):
    def test_prepare_allocations_changes_one_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare_allocations(
                num_layers=6,
                layer_spec="top:2",
                base_k_bits=4,
                base_v_bits=4,
                reduced_bits=2,
                out_dir=Path(tmp),
            )
            self.assertEqual(manifest["selected_layers"], [4, 5])
            self.assertEqual(len(manifest["candidates"]), 4)
            k_payload = json.loads((Path(tmp) / "layer4_k2.json").read_text())
            v_payload = json.loads((Path(tmp) / "layer4_v2.json").read_text())
            self.assertEqual(k_payload["k_bits"], [16, 16, 16, 16, 2, 4])
            self.assertEqual(k_payload["v_bits"], [16, 16, 16, 16, 4, 4])
            self.assertEqual(v_payload["k_bits"], [16, 16, 16, 16, 4, 4])
            self.assertEqual(v_payload["v_bits"], [16, 16, 16, 16, 2, 4])

    def test_layer_parser_validates_ranges(self) -> None:
        self.assertEqual(parse_layers("top:3", 8), [5, 6, 7])
        self.assertEqual(parse_layers("1,3,3", 8), [1, 3])
        with self.assertRaises(ValueError):
            parse_layers("top:9", 8)

    def test_paired_component_effects_align_examples(self) -> None:
        rows = []
        values = {
            ("0", "10"): {"base": 1, "k": 0, "v": 1},
            ("0", "11"): {"base": 1, "k": 1, "v": 0},
        }
        for (seed, source_idx), configs in values.items():
            for config, correct in configs.items():
                rows.append(
                    {
                        "seed": seed,
                        "source_idx": source_idx,
                        "config": config,
                        "raw_correct": str(correct),
                    }
                )
        effects = paired_component_effects(
            rows,
            baseline="base",
            k_config="k",
            v_config="v",
        )
        self.assertEqual(effects["k_harm"], [1.0, 0.0])
        self.assertEqual(effects["v_harm"], [0.0, 1.0])
        self.assertEqual(effects["k_minus_v_harm"], [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
