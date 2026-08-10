import unittest
from types import SimpleNamespace

import torch

from benchmark_spec_kv_quantization import estimate_total_kv_memory
from kv_cache_quantization import (
    AFFINE_QUANT,
    PER_CHANNEL_AXIS,
    PER_TOKEN_AXIS,
    estimate_model_kv_cache_bytes,
    quantize_dequantize_per_vector_symmetric,
    quantize_dequantize_per_vector_affine,
    quantize_key_cache_kivi_style,
)


class KiviAxisQuantizationTest(unittest.TestCase):
    def test_affine_values_use_the_full_unsigned_range(self) -> None:
        values = torch.tensor([[[[-3.0, -2.0, 1.0, 5.0]]]])
        symmetric = quantize_dequantize_per_vector_symmetric(values, bits=2)
        affine = quantize_dequantize_per_vector_affine(values, bits=2)

        affine_mse = torch.mean((affine - values) ** 2)
        symmetric_mse = torch.mean((symmetric - values) ** 2)
        self.assertLess(float(affine_mse), float(symmetric_mse))

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

    def test_speculative_tail_does_not_promote_uncommitted_key_group(self) -> None:
        torch.manual_seed(13)
        keys = torch.randn(1, 2, 36, 8)
        committed = quantize_key_cache_kivi_style(
            keys[..., :32, :],
            bits=4,
            group_size=4,
            residual_length=8,
        )
        speculative = torch.cat((committed, keys[..., 32:, :]), dim=-2)

        unchanged = quantize_key_cache_kivi_style(
            speculative,
            bits=4,
            group_size=4,
            residual_length=8,
            previous_seq_len=32,
            quantization_seq_len=32,
        )
        torch.testing.assert_close(unchanged, speculative)

        promoted = quantize_key_cache_kivi_style(
            unchanged,
            bits=4,
            group_size=4,
            residual_length=8,
            previous_seq_len=32,
            quantization_seq_len=36,
        )
        self.assertFalse(torch.equal(promoted[..., 24:28, :], keys[..., 24:28, :]))
        torch.testing.assert_close(promoted[..., 28:, :], keys[..., 28:, :])

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

        affine_values = estimate_model_kv_cache_bytes(
            config=config,
            seq_len=1024,
            dtype_name="bf16",
            k_bits_by_layer=[4, 4],
            v_bits_by_layer=[4, 4],
            key_quant_axis=PER_CHANNEL_AXIS,
            key_group_size=32,
            key_residual_length=128,
            value_quant_scheme=AFFINE_QUANT,
        )
        self.assertGreater(
            affine_values["quantized_cache_bytes"],
            per_channel["quantized_cache_bytes"],
        )

    def test_joint_memory_estimate_quantizes_target_and_draft(self) -> None:
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=2,
            num_attention_heads=4,
            hidden_size=32,
            head_dim=8,
        )
        target = SimpleNamespace(config=config)
        draft = SimpleNamespace(config=config)
        memory = estimate_total_kv_memory(
            big_model=target,
            small_model=draft,
            big_dtype="bf16",
            small_dtype="bf16",
            seq_len=1024,
            k_bits=[4, 4],
            v_bits=[4, 4],
            target_k_bits=[8, 8],
            target_v_bits=[8, 8],
            scale_bits=16,
            key_quant_axis=PER_CHANNEL_AXIS,
            key_group_size=32,
            key_residual_length=128,
            value_quant_scheme=AFFINE_QUANT,
        )
        self.assertGreater(memory["target_cache_saved_fraction"], 0.0)
        self.assertGreater(memory["draft_cache_saved_fraction"], 0.0)
        self.assertGreater(memory["total_cache_saved_fraction"], 0.0)
        self.assertLess(memory["quantized_total_cache_bytes"], memory["native_total_cache_bytes"])


if __name__ == "__main__":
    unittest.main()
