import unittest

from aggregate_quantizer_factorial import factorial_effects


def row(acceptance: float, kl: float):
    return {"acceptance_delta": acceptance, "quality_kl": kl}


class QuantizerFactorialTest(unittest.TestCase):
    def test_computes_main_effects_and_interaction(self) -> None:
        key = (1024, "k4v8")
        cells = {
            "per_token_symmetric": {key: row(-0.30, 2.0)},
            "per_token_affine": {key: row(-0.28, 1.8)},
            "per_channel_symmetric": {key: row(-0.01, 0.02)},
            "per_channel_affine": {key: row(0.00, 0.01)},
        }

        result = factorial_effects(cells)[0]

        self.assertAlmostEqual(
            result["key_axis_effect_under_symmetric/acceptance_delta"], 0.29
        )
        self.assertAlmostEqual(
            result["value_affine_effect_under_per_channel/quality_kl"], -0.01
        )
        self.assertAlmostEqual(result["interaction/acceptance_delta"], -0.01)


if __name__ == "__main__":
    unittest.main()
