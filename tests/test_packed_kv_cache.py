import unittest

import torch

from packed_kv_cache import (
    pack_kivi_kv,
    pack_per_channel_grouped_affine,
    pack_per_token_affine,
    pack_unsigned,
    packed_num_bytes,
    unpack_unsigned,
)


class PackedKVCacheTest(unittest.TestCase):
    def test_unsigned_round_trip_for_supported_precisions(self) -> None:
        for bits in (2, 3, 4, 5, 7, 8):
            values = torch.arange(37, dtype=torch.int64).remainder(1 << bits).to(torch.uint8)
            payload = pack_unsigned(values, bits)
            restored = unpack_unsigned(payload, bits, values.numel())

            self.assertEqual(payload.numel(), packed_num_bytes(values.numel(), bits))
            self.assertTrue(torch.equal(restored, values))

    def test_per_token_affine_uses_actual_packed_storage(self) -> None:
        values = torch.linspace(-2.0, 3.0, 2 * 3 * 5 * 7).reshape(2, 3, 5, 7).to(torch.bfloat16)
        packed = pack_per_token_affine(values, bits=4)
        restored = packed.unpack()

        self.assertEqual(restored.shape, values.shape)
        self.assertEqual(restored.dtype, values.dtype)
        self.assertEqual(packed.payload.numel(), packed_num_bytes(values.numel(), 4))
        self.assertLess(torch.mean((restored.float() - values.float()).square()).item(), 0.02)
        self.assertLess(packed.storage_bytes, values.numel() * values.element_size())

    def test_grouped_keys_preserve_residual_tail_exactly(self) -> None:
        torch.manual_seed(7)
        keys = torch.randn(1, 2, 11, 8, dtype=torch.bfloat16)
        packed = pack_per_channel_grouped_affine(
            keys,
            bits=3,
            group_size=4,
            residual_length=3,
        )
        restored = packed.unpack()

        self.assertEqual(packed.prefix_length, 8)
        self.assertTrue(torch.equal(restored[..., 8:, :], keys[..., 8:, :]))
        self.assertLess(torch.mean((restored[..., :8, :].float() - keys[..., :8, :].float()).square()).item(), 0.08)

    def test_kivi_pair_reports_component_storage(self) -> None:
        keys = torch.randn(1, 2, 32, 16, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        packed_k, packed_v = pack_kivi_kv(
            keys,
            values,
            k_bits=4,
            v_bits=4,
            group_size=8,
            residual_length=8,
        )

        native_bytes = 2 * keys.numel() * keys.element_size()
        self.assertLess(packed_k.storage_bytes + packed_v.storage_bytes, native_bytes)
        self.assertEqual(packed_k.unpack().shape, keys.shape)
        self.assertEqual(packed_v.unpack().shape, values.shape)


if __name__ == "__main__":
    unittest.main()
