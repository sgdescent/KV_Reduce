import unittest

from aggregate_value_precision_sweep import parse_seed_filter


class SeedFilteredAggregationTest(unittest.TestCase):
    def test_empty_filter_selects_all_seeds(self) -> None:
        self.assertIsNone(parse_seed_filter(""))
        self.assertIsNone(parse_seed_filter(" , "))

    def test_seed_allowlist_is_normalized(self) -> None:
        self.assertEqual(parse_seed_filter("20, 22,20"), {20, 22})


if __name__ == "__main__":
    unittest.main()
