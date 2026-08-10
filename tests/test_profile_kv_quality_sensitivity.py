import unittest

from profile_kv_quality_sensitivity import build_parser


class ProfileKvQualitySensitivityTest(unittest.TestCase):
    def test_skip_sequences_defaults_to_zero(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.skip_sequences, 0)

    def test_skip_sequences_accepts_disjoint_shard_offset(self):
        args = build_parser().parse_args(["--skip_sequences", "512"])
        self.assertEqual(args.skip_sequences, 512)


if __name__ == "__main__":
    unittest.main()
