"""
Standalone analysis script for post-hoc analysis of experiment results.
Run after experiments complete to generate tables, stats, and figures.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results
    python analyze_results.py --scenario neutral
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from src.simulation import load_all_results
from src.scoring import (
    summary_table,
    run_all_comparisons,
    aggregate_episode_scores,
    aggregate_nuclear_rate,
    aggregate_severity_distribution,
    compute_first_escalation_turn,
    FRONTIER_BASELINES,
)
from src.visualization import generate_all_plots
from src.config import MODELS, SEVERITY_LABELS


def detailed_analysis(results_by_model, scenario=None):
    """Comprehensive analysis output."""

    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS REPORT")
    if scenario:
        print(f"Scenario filter: {scenario}")
    print("=" * 70)

    # Filter by scenario if specified
    if scenario:
        filtered = {}
        for mk, eps in results_by_model.items():
            filtered_eps = [e for e in eps if e["scenario"] == scenario]
            if filtered_eps:
                filtered[mk] = filtered_eps
        results_by_model = filtered

    # 1. Summary table
    print("\n--- SUMMARY ---")
    print(summary_table(results_by_model))

    # 2. Per-scenario breakdown
    print("\n--- PER-SCENARIO BREAKDOWN ---")
    scenarios_found = set()
    for eps in results_by_model.values():
        for e in eps:
            scenarios_found.add(e["scenario"])

    for sc in sorted(scenarios_found):
        print(f"\n  Scenario: {sc}")
        for mk in sorted(results_by_model.keys()):
            sc_eps = [e for e in results_by_model[mk] if e["scenario"] == sc]
            if not sc_eps:
                continue
            scores = aggregate_episode_scores(sc_eps)
            nuc = aggregate_nuclear_rate(sc_eps)
            model_name = MODELS[mk].name if mk in MODELS else mk
            print(
                f"    {model_name:<20} n={len(sc_eps):>3}  "
                f"mean={scores.mean():>7.1f}  std={scores.std():>6.1f}  "
                f"min={scores.min():>5}  max={scores.max():>5}  "
                f"nuc={nuc*100:>5.1f}%"
            )

    # 3. Severity distribution
    print("\n--- SEVERITY DISTRIBUTION ---")
    for mk in sorted(results_by_model.keys()):
        dist = aggregate_severity_distribution(results_by_model[mk])
        model_name = MODELS[mk].name if mk in MODELS else mk
        print(f"  {model_name}:")
        for sev, frac in sorted(dist.items()):
            bar = "#" * int(frac * 50)
            print(f"    {SEVERITY_LABELS[sev]:<25} {frac*100:>5.1f}%  {bar}")

    # 4. First escalation turn analysis
    print("\n--- FIRST VIOLENT ESCALATION TURN (severity >= 5) ---")
    for mk in sorted(results_by_model.keys()):
        turns = []
        for ep in results_by_model[mk]:
            t = compute_first_escalation_turn(ep["actions"], threshold=5)
            if t > 0:
                turns.append(t)
        model_name = MODELS[mk].name if mk in MODELS else mk
        if turns:
            print(
                f"  {model_name:<20}  mean_turn={np.mean(turns):.1f}  "
                f"median={np.median(turns):.0f}  "
                f"never={len(results_by_model[mk])-len(turns)}/{len(results_by_model[mk])}"
            )
        else:
            print(f"  {model_name:<20}  Never escalated to violent level")

    # 5. Comparison with frontier baselines (per-action normalized)
    print("\n--- COMPARISON WITH FRONTIER BASELINES (per-action normalized) ---")
    print(f"  {'Model':<20} {'E/action':>8} {'Nuc %':>8}  {'vs GPT-4-Base':>15}")
    print("  " + "-" * 55)
    gpt4b_ref = FRONTIER_BASELINES["GPT-4-Base"]["mean_per_action"]
    for mk in sorted(results_by_model.keys()):
        from src.scoring import compute_mean_escalation_per_action
        per_act = compute_mean_escalation_per_action(results_by_model[mk])
        nuc = aggregate_nuclear_rate(results_by_model[mk])
        model_name = MODELS[mk].name if mk in MODELS else mk
        ratio = per_act / gpt4b_ref if gpt4b_ref > 0 else 0
        print(
            f"  {model_name:<20} {per_act:>8.2f} {nuc*100:>7.1f}%  "
            f"{ratio:>14.2f}x"
        )
    print("  " + "-" * 55)
    for name, data in FRONTIER_BASELINES.items():
        ratio = data["mean_per_action"] / gpt4b_ref
        print(
            f"  {name:<20} {data['mean_per_action']:>8.2f} "
            f"{data['nuclear_rate']*100:>7.1f}%  {ratio:>14.2f}x  (ref)"
        )

    # 6. Statistical comparisons
    if len(results_by_model) > 1:
        print("\n--- PAIRWISE STATISTICAL TESTS ---")
        comparisons = run_all_comparisons(results_by_model)
        for comp in comparisons:
            sig = "***" if comp["ks_test"]["p_value"] < 0.001 else \
                  "**" if comp["ks_test"]["p_value"] < 0.01 else \
                  "*" if comp["ks_test"]["p_value"] < 0.05 else "n.s."
            print(
                f"  {comp['comparison']:<35}  "
                f"KS p={comp['ks_test']['p_value']:.4f} {sig}  "
                f"W1={comp['wasserstein']['distance']:.1f}"
            )

    return results_by_model


def main():
    parser = argparse.ArgumentParser(description="Analyze wargame results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--scenario", default=None, help="Filter by scenario")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    all_results = load_all_results(args.results_dir)
    if not all_results:
        print(f"No results in {args.results_dir}/")
        sys.exit(1)

    print(f"Loaded {len(all_results)} episode results.")

    results_by_model = defaultdict(list)
    for r in all_results:
        results_by_model[r["model"]].append(r)

    detailed_analysis(dict(results_by_model), scenario=args.scenario)

    if not args.no_plots:
        generate_all_plots(dict(results_by_model))


if __name__ == "__main__":
    main()
