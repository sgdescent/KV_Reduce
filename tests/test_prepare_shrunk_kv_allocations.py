import csv
import json

import pytest

from prepare_shrunk_kv_allocations import (
    validate_allocation_layers,
    validate_profile_layers,
)


def test_validate_allocation_layers_rejects_wrong_model_depth(tmp_path):
    path = tmp_path / "allocation.json"
    path.write_text(
        json.dumps({"k_bits": [4] * 36, "v_bits": [4] * 36}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 28"):
        validate_allocation_layers(path, 28)


def test_validate_profile_layers_rejects_incomplete_depth(tmp_path):
    path = tmp_path / "profile.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "component"])
        writer.writeheader()
        for layer in range(3):
            writer.writerow({"layer": layer, "component": "k"})
            writer.writerow({"layer": layer, "component": "v"})

    validate_profile_layers(path, 3)
    with pytest.raises(ValueError, match="expected 0..3"):
        validate_profile_layers(path, 4)
