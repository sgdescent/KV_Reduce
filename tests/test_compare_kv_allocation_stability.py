import json

from compare_kv_allocation_stability import compare_bits, parse_allocation, stability_rows


def test_compare_bits_reports_component_and_total_disagreement():
    candidate = {"k": [4, 8], "v": [8, 4]}
    reference = {"k": [4, 4], "v": [8, 8]}

    result = compare_bits(candidate, reference)

    assert result["k_disagreement_fraction"] == 0.5
    assert result["v_disagreement_fraction"] == 0.5
    assert result["disagreement_fraction"] == 0.5
    assert result["mean_absolute_bit_difference"] == 2.0
    assert result["exact_allocation_match"] == 0.0


def test_stability_rows_use_last_allocation_as_reference(tmp_path):
    paths = []
    for label, k_bits, v_bits in (
        ("n4", [4, 8], [8, 4]),
        ("n8", [4, 4], [8, 8]),
        ("n16", [4, 4], [8, 8]),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"k_bits": k_bits, "v_bits": v_bits}))
        paths.append(("quality", label, path))

    rows = stability_rows(paths)

    assert [row["profile"] for row in rows] == ["n4", "n8"]
    assert rows[0]["reference"] == "n16"
    assert rows[0]["disagreement_fraction"] == 0.5
    assert rows[1]["exact_allocation_match"] == 1.0


def test_parse_allocation_accepts_objective_label_path():
    objective, label, path = parse_allocation("acceptance:n8=outputs/n8.json")

    assert objective == "acceptance"
    assert label == "n8"
    assert str(path) == "outputs/n8.json"
