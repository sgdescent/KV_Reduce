#!/usr/bin/env python3
"""Aggregate multi-budget, multi-context, multi-seed objective KV results."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EXACT_ACCEPTANCE_EVALUATOR_VERSION = "cached_dynamic_v6_sequential_target"
QUALITY_EVALUATOR_VERSION = "teacher_forced_cached_v1"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: List[float]) -> float:
    return 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def bootstrap_mean_ci(values: List[float], *, seed: int, samples: int = 2000) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    return mean(values), estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def hierarchical_bootstrap_mean_ci(
    clusters: Mapping[Tuple[int, int], Sequence[float]],
    *,
    seed: int,
    samples: int = 5000,
) -> Tuple[float, float, float, int]:
    """Bootstrap run cells first and examples second to avoid pseudoreplication."""
    nonempty = {key: list(values) for key, values in clusters.items() if values}
    values = [value for cluster in nonempty.values() for value in cluster]
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    if len(nonempty) == 1:
        estimate, low, high = bootstrap_mean_ci(values, seed=seed, samples=samples)
        return estimate, low, high, 1
    rng = random.Random(seed)
    keys = list(nonempty)
    estimates = []
    for _ in range(samples):
        sampled_values = []
        for _ in keys:
            cluster = nonempty[keys[rng.randrange(len(keys))]]
            sampled_values.extend(cluster[rng.randrange(len(cluster))] for _ in cluster)
        estimates.append(mean(sampled_values))
    estimates.sort()
    return (
        mean(values),
        estimates[int(0.025 * samples)],
        estimates[min(samples - 1, int(0.975 * samples))],
        len(nonempty),
    )


def load_manifest_cells(matrix_dir: Path) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """Load the matrix contract used to reject missing or underfilled cells."""
    manifest_path = matrix_dir / "manifest.tsv"
    if not manifest_path.exists():
        raise ValueError(f"Strict aggregation requires {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    required = {"objective", "budget", "context", "seed", "num_eval", "skip_blocks"}
    missing_fields = required - set(manifest_rows[0])
    if missing_fields:
        raise ValueError(f"Manifest is missing fields: {sorted(missing_fields)}")

    cells: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in manifest_rows:
        key = (int(row["budget"]), int(row["context"]), int(row["seed"]))
        objective = row["objective"]
        if objective not in {"quality", "acceptance"}:
            raise ValueError(f"Unexpected objective {objective!r} in {manifest_path}")
        cell = cells.setdefault(
            key,
            {
                "num_eval": int(row["num_eval"]),
                "objectives": {},
            },
        )
        if int(row["num_eval"]) != cell["num_eval"]:
            raise ValueError(f"Inconsistent num_eval for matrix cell {key}")
        if objective in cell["objectives"]:
            raise ValueError(f"Duplicate {objective} row for matrix cell {key}")
        cell["objectives"][objective] = row
    incomplete = {key: sorted(set(("quality", "acceptance")) - set(cell["objectives"])) for key, cell in cells.items()}
    incomplete = {key: objectives for key, objectives in incomplete.items() if objectives}
    if incomplete:
        raise ValueError(f"Manifest has incomplete objective pairs: {incomplete}")
    return cells


def audit_cell_coverage(
    *,
    quality_eval: Mapping[str, Any],
    acceptance_eval: Mapping[str, Any],
    quality_rows: Sequence[Mapping[str, str]],
    acceptance_rows: Sequence[Mapping[str, str]],
    expected_num_eval: int,
    tracked_names: Sequence[str],
) -> List[str]:
    """Return integrity errors for one budget/context/seed result cell."""
    issues = []
    expected_names = {"none", *tracked_names}
    if int(quality_eval.get("num_sequences", -1)) != expected_num_eval:
        issues.append(
            f"quality num_sequences={quality_eval.get('num_sequences')} expected={expected_num_eval}"
        )
    if int(acceptance_eval.get("num_prompts", -1)) != expected_num_eval:
        issues.append(
            f"acceptance num_prompts={acceptance_eval.get('num_prompts')} expected={expected_num_eval}"
        )
    quality_names = set(quality_eval.get("summaries", {}))
    acceptance_names = set(acceptance_eval.get("summaries", {}))
    if quality_names != expected_names:
        issues.append(f"quality configs={sorted(quality_names)} expected={sorted(expected_names)}")
    if acceptance_names != expected_names:
        issues.append(f"acceptance configs={sorted(acceptance_names)} expected={sorted(expected_names)}")
    if acceptance_eval.get("target_quant_configs") != ["none"]:
        issues.append("acceptance target_quant_configs must be ['none']")

    quality_counts: Dict[str, set[int]] = defaultdict(set)
    for row in quality_rows:
        quality_counts[str(row["candidate"])].add(int(row["sequence_idx"]))
    acceptance_counts: Dict[str, set[int]] = defaultdict(set)
    for row in acceptance_rows:
        acceptance_counts[str(row["config"])].add(int(row["prompt_idx"]))
    expected_indices = set(range(expected_num_eval))
    for name in expected_names:
        if quality_counts[name] != expected_indices:
            issues.append(
                f"quality rows for {name} cover {len(quality_counts[name])}/{expected_num_eval} sequences"
            )
        if acceptance_counts[name] != expected_indices:
            issues.append(
                f"acceptance rows for {name} cover {len(acceptance_counts[name])}/{expected_num_eval} prompts"
            )
    if len(quality_rows) != expected_num_eval * len(expected_names):
        issues.append(
            f"quality row count={len(quality_rows)} expected={expected_num_eval * len(expected_names)}"
        )
    if len(acceptance_rows) != expected_num_eval * len(expected_names):
        issues.append(
            f"acceptance row count={len(acceptance_rows)} expected={expected_num_eval * len(expected_names)}"
        )
    return issues


def classify_exactness(row: Dict[str, str], *, tie_margin: float) -> str:
    """Classify target-output agreement without hiding finite-precision ties."""
    if float(row["matches_target_greedy"]) >= 0.5:
        return "exact"
    try:
        margin = float(row["mismatch_min_top1_margin"])
    except (KeyError, TypeError, ValueError):
        margin = float("nan")
    if math.isfinite(margin) and margin <= tie_margin:
        return "numerical_tie"
    return "non_tie_or_unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate the objective-aware KV matrix.")
    parser.add_argument("--matrix_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--exactness_tie_margin",
        type=float,
        default=1e-3,
        help="Maximum target top-1 margin treated as a finite-precision numerical tie.",
    )
    parser.add_argument(
        "--acceptance_evaluator_version",
        default=EXACT_ACCEPTANCE_EVALUATOR_VERSION,
        help="Only aggregate acceptance artifacts produced by this evaluator.",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Fail unless every manifest cell, configuration, and example is present.",
    )
    parser.add_argument(
        "--require_exact_target",
        action="store_true",
        help="Fail if any speculative row differs from BF16 target greedy decoding.",
    )
    return parser


def make_plot(grouped_rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    contexts = sorted({int(row["context"]) for row in grouped_rows})
    budgets = sorted({int(row["budget"]) for row in grouped_rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.0 * len(contexts), 4.3), squeeze=False)
    colors = {
        "quality": "#26456E",
        "acceptance": "#D1495B",
        "k_priority": "#2A9D8F",
        "v_priority": "#E9C46A",
    }
    objective_order = [
        objective
        for objective in ("quality", "acceptance", "k_priority", "v_priority")
        if any(row["allocation_objective"] == objective for row in grouped_rows)
    ]
    for axis, context in zip(axes[0], contexts):
        for objective in objective_order:
            subset = sorted(
                [row for row in grouped_rows if int(row["context"]) == context and row["allocation_objective"] == objective],
                key=lambda row: int(row["budget"]),
            )
            axis.errorbar(
                [int(row["budget"]) for row in subset],
                [float(row["spec_accept_rate_mean"]) for row in subset],
                yerr=[float(row["spec_accept_rate_ci95"]) for row in subset],
                marker="o",
                linewidth=2,
                color=colors[objective],
                label=f"{objective}-optimized",
            )
        axis.set_title(f"Context {context}")
        axis.set_xlabel("Profiled mean KV bits")
        axis.set_ylabel("Speculative acceptance")
        axis.grid(alpha=0.25)
    axes[0][0].legend()
    fig.suptitle("Objective-Aware KV Allocation Across Context and Memory", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"objective_matrix_acceptance.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    args = build_parser().parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    missing = []
    rejected = []
    prompt_effects: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"acceptance": [], "quality_kl": [], "quality_delta_nll": []}
    )
    kv_prompt_effects: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"acceptance": [], "quality_kl": [], "quality_delta_nll": []}
    )
    native_prompt_effects: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
    acceptance_prompt_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
        lambda: {"candidate_pairs": 0, "excluded_non_tie": 0, "used": 0}
    )
    kv_acceptance_prompt_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
        lambda: {"candidate_pairs": 0, "excluded_non_tie": 0, "used": 0}
    )
    native_acceptance_prompt_counts: Dict[Tuple[int, int, str], Dict[str, int]] = defaultdict(
        lambda: {"candidate_pairs": 0, "excluded_non_tie": 0, "used": 0}
    )
    exactness_counts: Dict[Tuple[int, int, int], Dict[str, int]] = defaultdict(
        lambda: {"exact": 0, "numerical_tie": 0, "non_tie_or_unknown": 0, "invalid_prompts": 0}
    )
    exactness_examples: List[Dict[str, Any]] = []
    prompt_effect_clusters: Dict[Tuple[int, str], Dict[Tuple[int, int], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    kv_prompt_effect_clusters: Dict[Tuple[int, str], Dict[Tuple[int, int], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    native_prompt_effect_clusters: Dict[Tuple[int, str], Dict[Tuple[int, int], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    manifest_cells = load_manifest_cells(matrix_dir) if args.require_complete else {}
    seen_cells = set()
    integrity_violations: List[Dict[str, Any]] = []
    allocation_byte_audit: List[Dict[str, Any]] = []

    for budget_dir in sorted(matrix_dir.glob("budget_*")):
        budget = int(budget_dir.name.split("_", 1)[1])
        quality_allocation = read_json(budget_dir / "quality_allocation" / "allocation.json")
        acceptance_allocation = read_json(budget_dir / "acceptance_allocation" / "allocation.json")
        quality_target_bytes = quality_allocation.get("target_profiled_saved_bytes")
        acceptance_target_bytes = acceptance_allocation.get("target_profiled_saved_bytes")
        quality_achieved_bytes = quality_allocation.get("achieved_profiled_saved_bytes")
        acceptance_achieved_bytes = acceptance_allocation.get("achieved_profiled_saved_bytes")
        byte_issues = []
        if None in (
            quality_target_bytes,
            acceptance_target_bytes,
            quality_achieved_bytes,
            acceptance_achieved_bytes,
        ):
            byte_issues.append("objective allocations are missing metadata-aware byte budgets")
        else:
            if abs(float(quality_target_bytes) - float(acceptance_target_bytes)) > 0.5:
                byte_issues.append("quality and acceptance target bytes differ")
            if abs(float(quality_achieved_bytes) - float(acceptance_achieved_bytes)) > 0.5:
                byte_issues.append("quality and acceptance achieved bytes differ")
        allocation_byte_audit.append(
            {
                "budget": budget,
                "quality_target_profiled_saved_bytes": quality_target_bytes,
                "acceptance_target_profiled_saved_bytes": acceptance_target_bytes,
                "quality_achieved_profiled_saved_bytes": quality_achieved_bytes,
                "acceptance_achieved_profiled_saved_bytes": acceptance_achieved_bytes,
                "issues": byte_issues,
            }
        )
        if args.require_complete and byte_issues:
            raise ValueError(
                f"Objective matrix failed equal-byte gate at budget {budget}: {byte_issues}"
            )
        allocations = [("quality", quality_allocation), ("acceptance", acceptance_allocation)]
        for objective in ("k_priority", "v_priority"):
            allocation_path = budget_dir / f"{objective}_allocation" / "allocation.json"
            if allocation_path.exists():
                allocations.append((objective, read_json(allocation_path)))
        allocation_names = {objective: str(allocation["name"]) for objective, allocation in allocations}
        for context_dir in sorted(budget_dir.glob("ctx_*")):
            context = int(context_dir.name.split("_", 1)[1])
            for seed_dir in sorted(context_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_", 1)[1])
                cell_key = (budget, context, seed)
                expected_cell = manifest_cells.get(cell_key)
                if args.require_complete and expected_cell is None:
                    integrity_violations.append(
                        {"path": str(seed_dir), "issues": ["cell is not declared in manifest"]}
                    )
                    continue
                quality_path = seed_dir / "quality" / "summary.json"
                acceptance_path = seed_dir / "acceptance" / "summary.json"
                if not quality_path.exists() or not acceptance_path.exists():
                    missing.append(str(seed_dir))
                    continue
                quality_eval = read_json(quality_path)
                acceptance_eval = read_json(acceptance_path)
                quality_version = quality_eval.get("runtime", {}).get("evaluator_version")
                acceptance_version = acceptance_eval.get("runtime", {}).get("evaluator_version")
                if (
                    quality_version != QUALITY_EVALUATOR_VERSION
                    or acceptance_version != args.acceptance_evaluator_version
                ):
                    rejected.append(
                        {
                            "path": str(seed_dir),
                            "quality_version": quality_version,
                            "acceptance_version": acceptance_version,
                        }
                    )
                    continue
                for allocation_objective, allocation in allocations:
                    name = str(allocation["name"])
                    quality = quality_eval["summaries"][name]
                    acceptance = acceptance_eval["summaries"][name]
                    rows.append(
                        {
                            "budget": budget,
                            "context": context,
                            "seed": seed,
                            "allocation_objective": allocation_objective,
                            "allocation_name": name,
                            "profiled_mean_bits": allocation["achieved_profiled_mean_bits"],
                            "target_profiled_saved_bytes": allocation.get(
                                "target_profiled_saved_bytes"
                            ),
                            "achieved_profiled_saved_bytes": allocation.get(
                                "achieved_profiled_saved_bytes"
                            ),
                            "all_component_mean_bits": quality["allocation/all_bits_mean"],
                            "quality_delta_nll": quality["delta_nll"],
                            "quality_kl": quality["kl_p_to_q"],
                            "quality_top1_match": quality["top1_match"],
                            "spec_accept_rate": acceptance["overall_accept_rate"],
                            "spec_accepted_per_round": acceptance["accepted_per_round"],
                            "spec_round_js": acceptance["round_js"],
                            "draft_cache_saved_fraction": acceptance["draft_cache_saved_fraction"],
                            "total_cache_saved_fraction": acceptance["total_cache_saved_fraction"],
                        }
                    )

                quality_name = str(quality_allocation["name"])
                acceptance_name = str(acceptance_allocation["name"])
                k_priority_name = allocation_names.get("k_priority")
                v_priority_name = allocation_names.get("v_priority")
                tracked_names = set(allocation_names.values())
                acceptance_rows = read_csv(seed_dir / "acceptance" / "benchmark_rows.csv")
                quality_rows = read_csv(seed_dir / "quality" / "raw_sequence_rows.csv")
                if args.require_complete:
                    coverage_issues = audit_cell_coverage(
                        quality_eval=quality_eval,
                        acceptance_eval=acceptance_eval,
                        quality_rows=quality_rows,
                        acceptance_rows=acceptance_rows,
                        expected_num_eval=int(expected_cell["num_eval"]),
                        tracked_names=sorted(tracked_names),
                    )
                    if coverage_issues:
                        integrity_violations.append(
                            {"path": str(seed_dir), "issues": coverage_issues}
                        )
                        continue
                seen_cells.add(cell_key)
                acceptance_by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
                invalid_prompts = set()
                for row in acceptance_rows:
                    status = classify_exactness(row, tie_margin=args.exactness_tie_margin)
                    exactness_counts[(budget, context, seed)][status] += 1
                    if status == "non_tie_or_unknown":
                        invalid_prompts.add(row["prompt_idx"])
                        if len(exactness_examples) < 100:
                            exactness_examples.append(
                                {
                                    "budget": budget,
                                    "context": context,
                                    "seed": seed,
                                    "prompt_idx": int(row["prompt_idx"]),
                                    "config": row["config"],
                                    "mismatch_source": row.get("mismatch_source", ""),
                                    "mismatch_min_top1_margin": row.get("mismatch_min_top1_margin", "nan"),
                                }
                            )
                    if row["config"] in tracked_names or row["config"] == "none":
                        acceptance_by_prompt[row["prompt_idx"]][row["config"]] = float(row["accept_rate"])
                exactness_counts[(budget, context, seed)]["invalid_prompts"] = len(invalid_prompts)
                prompt_count = acceptance_prompt_counts[(budget, context)]
                for prompt_idx, pair in acceptance_by_prompt.items():
                    for objective, name in allocation_names.items():
                        if {"none", name}.issubset(pair):
                            native_count = native_acceptance_prompt_counts[(budget, context, objective)]
                            native_count["candidate_pairs"] += 1
                            if prompt_idx in invalid_prompts:
                                native_count["excluded_non_tie"] += 1
                            else:
                                native_count["used"] += 1
                                effect = pair[name] - pair["none"]
                                native_prompt_effects[(budget, context, objective)].append(effect)
                                native_prompt_effect_clusters[(budget, objective)][(context, seed)].append(
                                    effect
                                )
                    if {quality_name, acceptance_name}.issubset(pair):
                        prompt_count["candidate_pairs"] += 1
                        if prompt_idx in invalid_prompts:
                            prompt_count["excluded_non_tie"] += 1
                        else:
                            prompt_count["used"] += 1
                            effect = pair[acceptance_name] - pair[quality_name]
                            prompt_effects[(budget, context)]["acceptance"].append(effect)
                            prompt_effect_clusters[(budget, "acceptance")][(context, seed)].append(effect)
                    if (
                        k_priority_name is not None
                        and v_priority_name is not None
                        and {k_priority_name, v_priority_name}.issubset(pair)
                    ):
                        kv_count = kv_acceptance_prompt_counts[(budget, context)]
                        kv_count["candidate_pairs"] += 1
                        if prompt_idx in invalid_prompts:
                            kv_count["excluded_non_tie"] += 1
                        else:
                            kv_count["used"] += 1
                            effect = pair[k_priority_name] - pair[v_priority_name]
                            kv_prompt_effects[(budget, context)]["acceptance"].append(effect)
                            kv_prompt_effect_clusters[(budget, "acceptance")][(context, seed)].append(effect)

                quality_by_sequence: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
                for row in quality_rows:
                    if row["candidate"] in tracked_names:
                        quality_by_sequence[row["sequence_idx"]][row["candidate"]] = {
                            "kl": float(row["kl_p_to_q"]),
                            "delta_nll": float(row["delta_nll"]),
                        }
                for pair in quality_by_sequence.values():
                    if {quality_name, acceptance_name}.issubset(pair):
                        kl_effect = pair[acceptance_name]["kl"] - pair[quality_name]["kl"]
                        nll_effect = (
                            pair[acceptance_name]["delta_nll"]
                            - pair[quality_name]["delta_nll"]
                        )
                        prompt_effects[(budget, context)]["quality_kl"].append(kl_effect)
                        prompt_effects[(budget, context)]["quality_delta_nll"].append(nll_effect)
                        prompt_effect_clusters[(budget, "quality_kl")][(context, seed)].append(
                            kl_effect
                        )
                        prompt_effect_clusters[(budget, "quality_delta_nll")][(context, seed)].append(
                            nll_effect
                        )
                    if (
                        k_priority_name is not None
                        and v_priority_name is not None
                        and {k_priority_name, v_priority_name}.issubset(pair)
                    ):
                        kl_effect = pair[v_priority_name]["kl"] - pair[k_priority_name]["kl"]
                        nll_effect = (
                            pair[v_priority_name]["delta_nll"]
                            - pair[k_priority_name]["delta_nll"]
                        )
                        kv_prompt_effects[(budget, context)]["quality_kl"].append(kl_effect)
                        kv_prompt_effects[(budget, context)]["quality_delta_nll"].append(nll_effect)
                        kv_prompt_effect_clusters[(budget, "quality_kl")][(context, seed)].append(
                            kl_effect
                        )
                        kv_prompt_effect_clusters[(budget, "quality_delta_nll")][(context, seed)].append(
                            nll_effect
                        )

    if args.require_complete:
        missing_manifest_cells = sorted(set(manifest_cells) - seen_cells)
        if missing_manifest_cells or missing or rejected or integrity_violations:
            raise ValueError(
                "Objective matrix failed completeness gate: "
                f"missing_manifest_cells={missing_manifest_cells}, missing_pairs={missing}, "
                f"rejected_pairs={rejected}, integrity_violations={integrity_violations}"
            )
    if not rows:
        raise ValueError("No complete matrix result pairs were found.")

    groups: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["budget"]), int(row["context"]), str(row["allocation_objective"]))].append(row)
    grouped_rows: List[Dict[str, Any]] = []
    for (budget, context, objective), group in sorted(groups.items()):
        accept = [float(row["spec_accept_rate"]) for row in group]
        delta_nll = [float(row["quality_delta_nll"]) for row in group]
        quality_kl = [float(row["quality_kl"]) for row in group]
        grouped_rows.append(
            {
                "budget": budget,
                "context": context,
                "allocation_objective": objective,
                "num_seeds": len(group),
                "spec_accept_rate_mean": mean(accept),
                "spec_accept_rate_ci95": ci95(accept),
                "quality_delta_nll_mean": mean(delta_nll),
                "quality_delta_nll_ci95": ci95(delta_nll),
                "quality_kl_mean": mean(quality_kl),
                "quality_kl_ci95": ci95(quality_kl),
                "total_cache_saved_fraction": mean(float(row["total_cache_saved_fraction"]) for row in group),
            }
        )

    paired: Dict[Tuple[int, int, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["budget"]), int(row["context"]), int(row["seed"]))][str(row["allocation_objective"])] = row
    effects: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    heuristic_effects: Dict[Tuple[int, int, str], List[Tuple[float, float, float]]] = defaultdict(list)
    kv_priority_effects: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = defaultdict(list)
    for (budget, context, _), pair in paired.items():
        if not {"quality", "acceptance"}.issubset(pair):
            continue
        effects[(budget, context)].append(
            (
                float(pair["acceptance"]["spec_accept_rate"]) - float(pair["quality"]["spec_accept_rate"]),
                float(pair["acceptance"]["quality_delta_nll"]) - float(pair["quality"]["quality_delta_nll"]),
            )
        )
        for heuristic in ("k_priority", "v_priority"):
            if heuristic not in pair:
                continue
            heuristic_effects[(budget, context, heuristic)].append(
                (
                    float(pair["acceptance"]["spec_accept_rate"])
                    - float(pair[heuristic]["spec_accept_rate"]),
                    float(pair[heuristic]["quality_kl"]) - float(pair["quality"]["quality_kl"]),
                    float(pair[heuristic]["quality_delta_nll"])
                    - float(pair["quality"]["quality_delta_nll"]),
                )
            )
        if {"k_priority", "v_priority"}.issubset(pair):
            kv_priority_effects[(budget, context)].append(
                (
                    float(pair["k_priority"]["spec_accept_rate"])
                    - float(pair["v_priority"]["spec_accept_rate"]),
                    float(pair["v_priority"]["quality_kl"])
                    - float(pair["k_priority"]["quality_kl"]),
                    float(pair["v_priority"]["quality_delta_nll"])
                    - float(pair["k_priority"]["quality_delta_nll"]),
                )
            )
    effect_rows = []
    for (budget, context), values in sorted(effects.items()):
        acceptance_advantage = [value[0] for value in values]
        quality_advantage = [value[1] for value in values]
        row = {
                "budget": budget,
                "context": context,
                "num_seeds": len(values),
                "acceptance_optimized_acceptance_advantage_mean": mean(acceptance_advantage),
                "acceptance_optimized_acceptance_advantage_ci95": ci95(acceptance_advantage),
                "quality_optimized_delta_nll_advantage_mean": mean(quality_advantage),
                "quality_optimized_delta_nll_advantage_ci95": ci95(quality_advantage),
            }
        paired = prompt_effects[(budget, context)]
        for metric, metric_values in paired.items():
            estimate, low, high = bootstrap_mean_ci(
                metric_values,
                seed=budget * 100000 + context * 10 + len(metric),
            )
            row[f"paired_{metric}_n"] = len(metric_values)
            row[f"paired_{metric}_mean"] = estimate
            row[f"paired_{metric}_ci_low"] = low
            row[f"paired_{metric}_ci_high"] = high
        prompt_count = acceptance_prompt_counts[(budget, context)]
        row["paired_acceptance_candidate_n"] = prompt_count["candidate_pairs"]
        row["paired_acceptance_excluded_non_tie_n"] = prompt_count["excluded_non_tie"]
        row["paired_acceptance_valid_n"] = prompt_count["used"]
        effect_rows.append(row)

    heuristic_effect_rows = []
    for (budget, context, heuristic), values in sorted(heuristic_effects.items()):
        acceptance_advantage = [value[0] for value in values]
        quality_kl_advantage = [value[1] for value in values]
        quality_delta_nll_advantage = [value[2] for value in values]
        heuristic_effect_rows.append(
            {
                "budget": budget,
                "context": context,
                "heuristic": heuristic,
                "num_seeds": len(values),
                "acceptance_allocation_acceptance_advantage_mean": mean(acceptance_advantage),
                "acceptance_allocation_acceptance_advantage_ci95": ci95(acceptance_advantage),
                "quality_allocation_kl_advantage_mean": mean(quality_kl_advantage),
                "quality_allocation_kl_advantage_ci95": ci95(quality_kl_advantage),
                "quality_allocation_delta_nll_advantage_mean": mean(quality_delta_nll_advantage),
                "quality_allocation_delta_nll_advantage_ci95": ci95(quality_delta_nll_advantage),
            }
        )

    kv_priority_effect_rows = []
    for (budget, context), values in sorted(kv_priority_effects.items()):
        row = {
            "budget": budget,
            "context": context,
            "num_seeds": len(values),
            "k_priority_acceptance_advantage_mean": mean(value[0] for value in values),
            "k_priority_acceptance_advantage_ci95": ci95([value[0] for value in values]),
            "k_priority_quality_kl_advantage_mean": mean(value[1] for value in values),
            "k_priority_quality_kl_advantage_ci95": ci95([value[1] for value in values]),
            "k_priority_quality_delta_nll_advantage_mean": mean(value[2] for value in values),
            "k_priority_quality_delta_nll_advantage_ci95": ci95([value[2] for value in values]),
        }
        for metric, metric_values in kv_prompt_effects[(budget, context)].items():
            estimate, low, high = bootstrap_mean_ci(
                metric_values,
                seed=budget * 200000 + context * 20 + len(metric),
            )
            row[f"paired_{metric}_n"] = len(metric_values)
            row[f"paired_{metric}_mean"] = estimate
            row[f"paired_{metric}_ci_low"] = low
            row[f"paired_{metric}_ci_high"] = high
        prompt_count = kv_acceptance_prompt_counts[(budget, context)]
        row["paired_acceptance_candidate_n"] = prompt_count["candidate_pairs"]
        row["paired_acceptance_excluded_non_tie_n"] = prompt_count["excluded_non_tie"]
        row["paired_acceptance_valid_n"] = prompt_count["used"]
        kv_priority_effect_rows.append(row)

    native_acceptance_effect_rows = []
    for (budget, context, objective), values in sorted(native_prompt_effects.items()):
        estimate, low, high = bootstrap_mean_ci(
            values,
            seed=budget * 300000 + context * 30 + len(objective),
        )
        prompt_count = native_acceptance_prompt_counts[(budget, context, objective)]
        native_acceptance_effect_rows.append(
            {
                "budget": budget,
                "context": context,
                "allocation_objective": objective,
                "paired_acceptance_n": len(values),
                "paired_acceptance_mean": estimate,
                "paired_acceptance_ci_low": low,
                "paired_acceptance_ci_high": high,
                "paired_acceptance_candidate_n": prompt_count["candidate_pairs"],
                "paired_acceptance_excluded_non_tie_n": prompt_count["excluded_non_tie"],
                "paired_acceptance_valid_n": prompt_count["used"],
            }
        )

    budget_prompt_effects: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"acceptance": [], "quality_kl": [], "quality_delta_nll": []}
    )
    for (budget, _), metrics in prompt_effects.items():
        for metric, values in metrics.items():
            budget_prompt_effects[budget][metric].extend(values)
    cross_context_rows = []
    for budget, metrics in sorted(budget_prompt_effects.items()):
        row: Dict[str, Any] = {"budget": budget}
        for metric, values in metrics.items():
            estimate, low, high, num_clusters = hierarchical_bootstrap_mean_ci(
                prompt_effect_clusters[(budget, metric)],
                seed=budget * 1000000 + len(metric),
                samples=5000,
            )
            row[f"paired_{metric}_n"] = len(values)
            row[f"paired_{metric}_clusters"] = num_clusters
            row[f"paired_{metric}_mean"] = estimate
            row[f"paired_{metric}_ci_low"] = low
            row[f"paired_{metric}_ci_high"] = high
        cross_context_rows.append(row)

    kv_budget_prompt_effects: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"acceptance": [], "quality_kl": [], "quality_delta_nll": []}
    )
    for (budget, _), metrics in kv_prompt_effects.items():
        for metric, values in metrics.items():
            kv_budget_prompt_effects[budget][metric].extend(values)
    kv_cross_context_rows = []
    for budget, metrics in sorted(kv_budget_prompt_effects.items()):
        row = {"budget": budget}
        for metric, values in metrics.items():
            estimate, low, high, num_clusters = hierarchical_bootstrap_mean_ci(
                kv_prompt_effect_clusters[(budget, metric)],
                seed=budget * 2000000 + len(metric),
                samples=5000,
            )
            row[f"paired_{metric}_n"] = len(values)
            row[f"paired_{metric}_clusters"] = num_clusters
            row[f"paired_{metric}_mean"] = estimate
            row[f"paired_{metric}_ci_low"] = low
            row[f"paired_{metric}_ci_high"] = high
        kv_cross_context_rows.append(row)

    native_budget_prompt_effects: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for (budget, _, objective), values in native_prompt_effects.items():
        native_budget_prompt_effects[(budget, objective)].extend(values)
    native_cross_context_rows = []
    for (budget, objective), values in sorted(native_budget_prompt_effects.items()):
        estimate, low, high, num_clusters = hierarchical_bootstrap_mean_ci(
            native_prompt_effect_clusters[(budget, objective)],
            seed=budget * 3000000 + len(objective),
            samples=5000,
        )
        native_cross_context_rows.append(
            {
                "budget": budget,
                "allocation_objective": objective,
                "paired_acceptance_n": len(values),
                "paired_acceptance_clusters": num_clusters,
                "paired_acceptance_mean": estimate,
                "paired_acceptance_ci_low": low,
                "paired_acceptance_ci_high": high,
            }
        )

    exactness_rows = []
    for (budget, context, seed), counts in sorted(exactness_counts.items()):
        exactness_rows.append({"budget": budget, "context": context, "seed": seed, **counts})
    exactness_totals = {
        key: sum(row[key] for row in exactness_rows)
        for key in ("exact", "numerical_tie", "non_tie_or_unknown", "invalid_prompts")
    }
    if args.require_exact_target and (
        exactness_totals["numerical_tie"] > 0
        or exactness_totals["non_tie_or_unknown"] > 0
    ):
        raise ValueError(
            "Objective matrix failed exact-target gate: "
            f"numerical_ties={exactness_totals['numerical_tie']}, "
            f"non_tie_or_unknown={exactness_totals['non_tie_or_unknown']}"
        )

    write_csv(rows, out_dir / "matrix_rows.csv")
    write_csv(grouped_rows, out_dir / "matrix_grouped.csv")
    write_csv(effect_rows, out_dir / "cross_objective_effects.csv")
    write_csv(cross_context_rows, out_dir / "cross_context_effects.csv")
    write_csv(heuristic_effect_rows, out_dir / "heuristic_effects.csv")
    write_csv(kv_priority_effect_rows, out_dir / "kv_priority_effects.csv")
    write_csv(kv_cross_context_rows, out_dir / "kv_priority_cross_context_effects.csv")
    write_csv(native_acceptance_effect_rows, out_dir / "native_acceptance_effects.csv")
    write_csv(native_cross_context_rows, out_dir / "native_acceptance_cross_context_effects.csv")
    write_csv(exactness_rows, out_dir / "exactness_audit.csv")
    payload = {
        "required_evaluator_versions": {
            "quality": QUALITY_EVALUATOR_VERSION,
            "acceptance": args.acceptance_evaluator_version,
        },
        "integrity_gates": {
            "require_complete": args.require_complete,
            "complete_matrix_gate": not missing and not rejected and not integrity_violations,
            "require_exact_target": args.require_exact_target,
            "exact_target_gate": (
                exactness_totals["numerical_tie"] == 0
                and exactness_totals["non_tie_or_unknown"] == 0
            ),
            "paired_within_objective_gate": True,
            "objective_allocations_byte_matched_gate": not any(
                audit["issues"] for audit in allocation_byte_audit
            ),
            "cross_context_ci": "hierarchical_cell_then_example_bootstrap",
        },
        "allocation_byte_audit": allocation_byte_audit,
        "num_complete_rows": len(rows),
        "num_missing_pairs": len(missing),
        "num_rejected_pairs": len(rejected),
        "missing_pairs": missing,
        "rejected_pairs": rejected,
        "exactness_tie_margin": args.exactness_tie_margin,
        "exactness_audit": {
            "totals": exactness_totals,
            "cells": exactness_rows,
            "non_tie_or_unknown_examples": exactness_examples,
        },
        "grouped": grouped_rows,
        "cross_objective_effects": effect_rows,
        "cross_context_effects": cross_context_rows,
        "heuristic_effects": heuristic_effect_rows,
        "kv_priority_effects": kv_priority_effect_rows,
        "kv_priority_cross_context_effects": kv_cross_context_rows,
        "native_acceptance_effects": native_acceptance_effect_rows,
        "native_acceptance_cross_context_effects": native_cross_context_rows,
        "plots": make_plot(grouped_rows, out_dir),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Aggregated {len(rows)} rows; missing pairs: {len(missing)}")


if __name__ == "__main__":
    main()
