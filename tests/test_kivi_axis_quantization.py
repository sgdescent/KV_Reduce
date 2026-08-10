import unittest
from types import SimpleNamespace

import torch

from kv_cache_quantization import (
    PER_CHANNEL_AXIS,
    PER_TOKEN_AXIS,
    estimate_model_kv_cache_bytes,
    quantize_dequantize_per_vector_symmetric,
    quantize_key_cache_kivi_style,
)


class KiviAxisQuantizationTest(unittest.TestCase):
    def test_per_channel_affine_keys_isolate_channel_outliers(self) -> None:
        keys = torch.tensor(
            [[[[100.0, 1.0], [80.0, 0.8], [60.0, 0.6], [40.0, 0.4]]]]
        )
        per_token = quantize_dequantize_per_vector_symmetric(keys, bits=4)
        per_channel = quantize_key_cache_kivi_style(
            keys,
            bits=4,
            group_size=4,
            residual_length=0,
        )

        token_mse = torch.mean((per_token[..., 1] - keys[..., 1]) ** 2)
        channel_mse = torch.mean((per_channel[..., 1] - keys[..., 1]) ** 2)
        self.assertLess(float(channel_mse), float(token_mse))

    def test_recent_residual_window_remains_full_precision(self) -> None:
        torch.manual_seed(7)
        keys = torch.randn(1, 2, 64, 8)
        quantized = quantize_key_cache_kivi_style(
            keys,
            bits=4,
            group_size=16,
            residual_length=20,
        )

        # floor((64 - 20) / 16) * 16 = 32 quantized prefix tokens.
        self.assertFalse(torch.equal(quantized[..., :32, :], keys[..., :32, :]))
        torch.testing.assert_close(quantized[..., 32:, :], keys[..., 32:, :])

    def test_incremental_quantization_matches_one_shot_groups(self) -> None:
        torch.manual_seed(11)
        keys = torch.randn(1, 2, 192, 8)
        first = quantize_key_cache_kivi_style(
            keys[..., :160, :],
            bits=4,
            group_size=32,
            residual_length=128,
        )
        extended = torch.cat((first, keys[..., 160:, :]), dim=-2)
        incremental = quantize_key_cache_kivi_style(
            extended,
            bits=4,
            group_size=32,
            residual_length=128,
            previous_seq_len=160,
        )
        one_shot = quantize_key_cache_kivi_style(
            keys,
            bits=4,
            group_size=32,
            residual_length=128,
        )

        torch.testing.assert_close(incremental, one_shot)

    def test_memory_estimate_counts_scales_and_residual(self) -> None:
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=2,
            num_attention_heads=4,
            hidden_size=32,
            head_dim=8,
        )
        per_token = estimate_model_kv_cache_bytes(
            config=config,
            seq_len=1024,
            dtype_name="bf16",
            k_bits_by_layer=[4, 4],
            v_bits_by_layer=[4, 4],
            key_quant_axis=PER_TOKEN_AXIS,
        )
        per_channel = estimate_model_kv_cache_bytes(
            config=config,
            seq_len=1024,
            dtype_name="bf16",
            k_bits_by_layer=[4, 4],
            v_bits_by_layer=[4, 4],
            key_quant_axis=PER_CHANNEL_AXIS,
            key_group_size=32,
            key_residual_length=128,
        )

        self.assertEqual(per_channel["key_quantized_prefix_tokens"], 896.0)
        self.assertEqual(per_channel["key_residual_tokens"], 128.0)
        self.assertNotEqual(per_channel["quantized_cache_bytes"], per_token["quantized_cache_bytes"])
        self.assertLess(per_channel["quantized_cache_bytes"], per_channel["native_cache_bytes"])


if __name__ == "__main__":
    unittest.main()
