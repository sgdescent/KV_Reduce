import json
from pathlib import Path

from aggregate_verifier_diagnostics import COUNT_FIELDS, MAX_FIELDS, main


def diagnostic(dtype: str, speculative_mismatches: int, *, num_prompts: int = 1) -> dict:
    payload = {
        "config": {
            "dtype": dtype,
            "attn_implementation": "sdpa",
            "seed": 0,
            "skip_prompts": 3,
        },
    }
    payload.update({field: 0 for field in COUNT_FIELDS})
    payload.update({field: 0.0 for field in MAX_FIELDS})
    payload["num_prompts"] = num_prompts
    payload["speculative_top1_mismatches"] = speculative_mismatches
    payload["speculative_independent_greedy_mismatches"] = speculative_mismatches
    payload["speculative_prompts_with_top1_mismatch"] = speculative_mismatches
    payload["speculative_prompts_with_independent_greedy_mismatch"] = speculative_mismatches
    return payload


def test_aggregate_groups_numerical_controls(tmp_path: Path, monkeypatch) -> None:
    for dtype, mismatches in (("bf16", 1), ("float32", 0)):
        (tmp_path / f"{dtype}.json").write_text(
            json.dumps(diagnostic(dtype, mismatches)), encoding="utf-8"
        )
    out_dir = tmp_path / "aggregate"
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_verifier_diagnostics.py",
            "--inputs",
            str(tmp_path / "*.json"),
            "--out_dir",
            str(out_dir),
        ],
    )

    main()

    result = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    grouped = {row["dtype"]: row for row in result["grouped"]}
    assert grouped["bf16"]["prompts_with_speculative_mismatch"] == 1
    assert grouped["float32"]["prompts_with_speculative_mismatch"] == 0


def test_aggregate_counts_prompts_within_batched_runs(tmp_path: Path, monkeypatch) -> None:
    payload = diagnostic("bf16", 3, num_prompts=32)
    (tmp_path / "bf16.json").write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "aggregate"
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_verifier_diagnostics.py",
            "--inputs",
            str(tmp_path / "*.json"),
            "--out_dir",
            str(out_dir),
        ],
    )

    main()

    result = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    grouped = result["grouped"][0]
    assert grouped["num_runs"] == 1
    assert grouped["num_prompts"] == 32
    assert grouped["prompts_with_speculative_mismatch"] == 3
