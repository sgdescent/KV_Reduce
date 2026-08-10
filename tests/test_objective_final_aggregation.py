import math
import csv
import json
import subprocess
import sys
from pathlib import Path

from aggregate_objective_kv_results import audit_acceptance_rows


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_row(prompt_idx, config, accept_rate, *, match=1, margin="nan"):
    return {
        "prompt_idx": str(prompt_idx),
        "config": config,
        "accept_rate": str(accept_rate),
        "matches_target_greedy": str(match),
        "mismatch_min_top1_margin": str(margin),
    }


def test_exactness_audit_keeps_ties_and_excludes_non_ties():
    rows = []
    values = {
        0: {"none": 0.5, "quality": 0.5, "acceptance": 0.6},
        1: {"none": 0.5, "quality": 0.4, "acceptance": 0.45},
        2: {"none": 0.5, "quality": 0.2, "acceptance": 0.9},
    }
    for prompt_idx, configs in values.items():
        for config, accept_rate in configs.items():
            if prompt_idx == 0:
                rows.append(make_row(prompt_idx, config, accept_rate))
            elif prompt_idx == 1:
                rows.append(make_row(prompt_idx, config, accept_rate, match=0, margin=0.0))
            else:
                rows.append(make_row(prompt_idx, config, accept_rate, match=0, margin=0.1))

    audit = audit_acceptance_rows(rows, quality_name="quality", acceptance_name="acceptance")

    assert audit["candidate_prompts"] == 3
    assert audit["valid_prompts"] == 2
    assert audit["excluded_non_tie_prompts"] == 1
    assert audit["row_counts"] == {
        "exact": 3,
        "numerical_tie": 3,
        "non_tie": 3,
        "invalid": 0,
    }
    assert math.isclose(audit["effects"]["quality_vs_native"]["mean"], -0.05)
    assert math.isclose(audit["effects"]["acceptance_vs_native"]["mean"], 0.025)
    assert math.isclose(audit["effects"]["acceptance_vs_quality"]["mean"], 0.075)


def test_exactness_audit_compares_optimized_policies_to_uniform():
    rows = []
    values = {
        0: {"none": 0.70, "k8v8": 0.65, "quality": 0.66, "acceptance": 0.68},
        1: {"none": 0.60, "k8v8": 0.55, "quality": 0.54, "acceptance": 0.57},
    }
    for prompt_idx, configs in values.items():
        for config, accept_rate in configs.items():
            rows.append(make_row(prompt_idx, config, accept_rate))

    audit = audit_acceptance_rows(
        rows,
        quality_name="quality",
        acceptance_name="acceptance",
        uniform_name="k8v8",
    )

    assert audit["candidate_prompts"] == 2
    assert audit["valid_prompts"] == 2
    assert math.isclose(audit["effects"]["uniform_vs_native"]["mean"], -0.05)
    assert math.isclose(audit["effects"]["quality_vs_uniform"]["mean"], 0.0, abs_tol=1e-12)
    assert math.isclose(audit["effects"]["acceptance_vs_uniform"]["mean"], 0.025)


def test_final_aggregate_reports_uniform_and_actual_memory_matching(tmp_path):
    quality_dir = tmp_path / "quality"
    acceptance_dir = tmp_path / "acceptance"
    out_dir = tmp_path / "out"
    quality_dir.mkdir()
    acceptance_dir.mkdir()

    quality_summaries = {
        "none": {"quantized_nll": 2.0},
        "k8v8": {
            "allocation/all_bits_mean": 8.0,
            "delta_nll": 0.02,
            "js": 0.01,
            "top1_match": 0.95,
        },
        "quality": {
            "allocation/all_bits_mean": 8.0,
            "delta_nll": 0.01,
            "js": 0.005,
            "top1_match": 0.97,
        },
        "acceptance": {
            "allocation/all_bits_mean": 8.0,
            "delta_nll": 0.03,
            "js": 0.015,
            "top1_match": 0.93,
        },
    }
    acceptance_summaries = {
        "none": {"overall_accept_rate": 0.70},
        "k8v8": {
            "overall_accept_rate": 0.65,
            "accepted_per_round": 2.0,
            "round_js": 0.03,
            "draft_cache_saved_fraction": 0.50,
            "total_cache_saved_fraction": 0.20,
        },
        "quality": {
            "overall_accept_rate": 0.66,
            "accepted_per_round": 2.1,
            "round_js": 0.02,
            "draft_cache_saved_fraction": 0.49,
            "total_cache_saved_fraction": 0.19,
        },
        "acceptance": {
            "overall_accept_rate": 0.68,
            "accepted_per_round": 2.2,
            "round_js": 0.01,
            "draft_cache_saved_fraction": 0.51,
            "total_cache_saved_fraction": 0.21,
        },
    }
    quality_summary = quality_dir / "summary.json"
    acceptance_summary = acceptance_dir / "summary.json"
    quality_summary.write_text(
        json.dumps({"runtime": {"evaluator_version": "quality-v1"}, "summaries": quality_summaries})
    )
    acceptance_summary.write_text(
        json.dumps({"runtime": {"evaluator_version": "spec-v1"}, "summaries": acceptance_summaries})
    )

    with (acceptance_dir / "benchmark_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_idx",
                "config",
                "accept_rate",
                "matches_target_greedy",
                "mismatch_min_top1_margin",
            ],
        )
        writer.writeheader()
        for config, accept_rate in (("none", 0.70), ("k8v8", 0.65), ("quality", 0.66), ("acceptance", 0.68)):
            writer.writerow(make_row(0, config, accept_rate))

    quality_allocation = tmp_path / "quality.json"
    acceptance_allocation = tmp_path / "acceptance.json"
    quality_allocation.write_text(
        json.dumps({"name": "quality", "risk_field": "quality_risk", "achieved_profiled_mean_bits": 8.0})
    )
    acceptance_allocation.write_text(
        json.dumps({"name": "acceptance", "risk_field": "acceptance_risk", "achieved_profiled_mean_bits": 8.0})
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "aggregate_objective_kv_results.py"),
            "--quality_summary",
            str(quality_summary),
            "--acceptance_summary",
            str(acceptance_summary),
            "--quality_allocation",
            str(quality_allocation),
            "--acceptance_allocation",
            str(acceptance_allocation),
            "--uniform_config",
            "k8v8",
            "--out_dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads((out_dir / "summary.json").read_text())
    assert [row["allocation"] for row in result["rows"]] == ["k8v8", "quality", "acceptance"]
    assert result["memory_matching"]["equal_nominal_mean_bits"] is True
    assert result["memory_matching"]["equal_actual_total_cache_bytes"] is False
    assert math.isclose(result["memory_matching"]["total_cache_saved_fraction_span"], 0.02)
    assert math.isclose(
        result["uniform_baseline_effects"]["acceptance_vs_uniform_acceptance"]["mean"],
        0.03,
    )
