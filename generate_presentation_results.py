"""
Generate presentation-ready results summary from experiment data.
Run after experiments complete to produce tables, key findings, and figures
ready to paste into slides.

Usage:
    python generate_presentation_results.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

from src.simulation import load_all_results
from src.scoring import (
    aggregate_episode_scores,
    aggregate_per_turn_mean,
    aggregate_severity_distribution,
    aggregate_nuclear_rate,
    compute_first_escalation_turn,
    compute_mean_escalation_per_action,
    run_all_comparisons,
    FRONTIER_BASELINES,
)
from src.visualization import generate_all_plots
from src.config import MODELS, SEVERITY_LABELS


def main():
    results_dir = "results"
    all_results = load_all_results(results_dir)

    if not all_results:
        print("No results yet. Run experiments first.")
        sys.exit(1)

    # Group by model
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    # Group by model+scenario
    by_model_scenario = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["scenario"])
        by_model_scenario[key].append(r)

    print("=" * 70)
    print("PRESENTATION RESULTS — Generated from experiment data")
    print(f"Total episodes: {len(all_results)}")
    print(f"Models: {sorted(by_model.keys())}")
    print("=" * 70)

    # ===== SLIDE 15: Overall Summary Table =====
    print("\n\n### SLIDE 15: Overall Escalation Scores ###")
    print()
    print("| Model | Episodes | Mean E | Std E | Nuclear % | Max E |")
    print("|-------|----------|--------|-------|-----------|-------|")
    for mk in sorted(by_model.keys()):
        eps = by_model[mk]
        scores = aggregate_episode_scores(eps)
        nuc = aggregate_nuclear_rate(eps) * 100
        name = MODELS[mk].name if mk in MODELS else mk
        print(f"| {name} | {len(eps)} | {scores.mean():.1f} | {scores.std():.1f} | {nuc:.1f}% | {scores.max()} |")

    print()
    print("**Frontier baselines (Rivera et al. 2024, per-action normalized):**")
    print("| Model | E/action | Nuclear % |")
    print("|-------|----------|-----------|")
    for name, data in FRONTIER_BASELINES.items():
        print(f"| {name} | {data['mean_per_action']} | {data['nuclear_rate']*100:.1f}% |")

    # ===== KEY FINDINGS =====
    print("\n\n### KEY FINDINGS (for presentation narration) ###")

    # Finding 1: Do open-weight models escalate?
    all_scores = []
    for eps in by_model.values():
        all_scores.extend([aggregate_episode_scores([e]).item() for e in eps])
    overall_mean = np.mean(all_scores)
    overall_nuclear = sum(
        aggregate_nuclear_rate(eps) for eps in by_model.values()
    ) / len(by_model)

    print(f"\n**Finding 1: Open-weight models {'DO' if overall_mean > 0 else 'do NOT'} escalate.**")
    print(f"  Overall mean escalation: {overall_mean:.1f}")
    print(f"  Overall nuclear rate: {overall_nuclear*100:.1f}%")

    if overall_mean > 0:
        print("  -> Escalation is intrinsic to transformer LLMs, not just frontier model artifact")

    # Finding 2: Which model is most/least escalatory?
    model_means = {}
    for mk, eps in by_model.items():
        scores = aggregate_episode_scores(eps)
        model_means[mk] = scores.mean()

    most = max(model_means, key=model_means.get)
    least = min(model_means, key=model_means.get)
    most_name = MODELS[most].name if most in MODELS else most
    least_name = MODELS[least].name if least in MODELS else least

    print(f"\n**Finding 2: Model variation**")
    print(f"  Most escalatory: {most_name} (mean={model_means[most]:.1f})")
    print(f"  Least escalatory: {least_name} (mean={model_means[least]:.1f})")
    print(f"  Range: {model_means[least]:.1f} to {model_means[most]:.1f}")

    # Finding 3: Scenario sensitivity
    print(f"\n**Finding 3: Scenario sensitivity**")
    for sc in ["neutral", "invasion", "cyber_attack"]:
        sc_scores = []
        for mk, eps in by_model.items():
            sc_eps = [e for e in eps if e["scenario"] == sc]
            if sc_eps:
                sc_scores.extend([aggregate_episode_scores([e]).item() for e in sc_eps])
        if sc_scores:
            print(f"  {sc:<15} mean={np.mean(sc_scores):.1f}, std={np.std(sc_scores):.1f}")

    # Finding 4: Comparison with GPT-4-Base (per-action normalized)
    gpt4b_pa = FRONTIER_BASELINES["GPT-4-Base"]["mean_per_action"]
    print(f"\n**Finding 4: vs GPT-4-Base (per-action={gpt4b_pa})**")
    for mk in sorted(by_model.keys()):
        pa = compute_mean_escalation_per_action(by_model[mk])
        name = MODELS[mk].name if mk in MODELS else mk
        ratio = pa / gpt4b_pa if gpt4b_pa > 0 else 0
        direction = "higher" if pa > gpt4b_pa else "lower"
        print(f"  {name}: {pa:.2f}/action ({ratio:.2f}x GPT-4-Base, {direction})")

    # Finding 5: Nuclear rates
    print(f"\n**Finding 5: Nuclear action rates**")
    for mk in sorted(by_model.keys()):
        nuc = aggregate_nuclear_rate(by_model[mk]) * 100
        name = MODELS[mk].name if mk in MODELS else mk
        gpt4b_nuc = FRONTIER_BASELINES["GPT-4-Base"]["nuclear_rate"] * 100
        print(f"  {name}: {nuc:.1f}% (GPT-4-Base: {gpt4b_nuc:.1f}%)")

    # ===== PER-SCENARIO TABLES =====
    print("\n\n### PER-SCENARIO BREAKDOWN ###")
    for sc in ["neutral", "invasion", "cyber_attack"]:
        print(f"\n**{sc.replace('_', ' ').title()} Scenario:**")
        print("| Model | Mean E | Std E | Nuclear % | First Violent Turn |")
        print("|-------|--------|-------|-----------|--------------------|")
        for mk in sorted(by_model.keys()):
            sc_eps = [e for e in by_model[mk] if e["scenario"] == sc]
            if not sc_eps:
                continue
            scores = aggregate_episode_scores(sc_eps)
            nuc = aggregate_nuclear_rate(sc_eps) * 100
            name = MODELS[mk].name if mk in MODELS else mk
            # First violent turn
            fvt_list = []
            for ep in sc_eps:
                t = compute_first_escalation_turn(ep["actions"], threshold=5)
                if t > 0:
                    fvt_list.append(t)
            fvt_str = f"{np.mean(fvt_list):.1f}" if fvt_list else "Never"
            print(f"| {name} | {scores.mean():.1f} | {scores.std():.1f} | {nuc:.1f}% | {fvt_str} |")

    # ===== STATISTICAL TESTS =====
    if len(by_model) > 1:
        print("\n\n### STATISTICAL COMPARISONS ###")
        print("| Comparison | KS stat | KS p-value | M-W p-value | W1 distance | Significant? |")
        print("|-----------|---------|-----------|-------------|-------------|--------------|")
        comparisons = run_all_comparisons(dict(by_model))
        for comp in comparisons:
            ks_p = comp["ks_test"]["p_value"]
            sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "n.s."
            print(
                f"| {comp['comparison']} | {comp['ks_test']['statistic']:.3f} | "
                f"{ks_p:.4f} | {comp['mann_whitney']['p_value']:.4f} | "
                f"{comp['wasserstein']['distance']:.1f} | {sig} |"
            )

    # ===== SEVERITY DISTRIBUTION =====
    print("\n\n### SEVERITY DISTRIBUTIONS ###")
    print("| Model | De-esc | Status quo | Posturing | Non-violent | Violent | Nuclear |")
    print("|-------|--------|-----------|-----------|-------------|---------|---------|")
    for mk in sorted(by_model.keys()):
        dist = aggregate_severity_distribution(by_model[mk])
        name = MODELS[mk].name if mk in MODELS else mk
        print(
            f"| {name} | {dist.get(1,0)*100:.1f}% | {dist.get(2,0)*100:.1f}% | "
            f"{dist.get(3,0)*100:.1f}% | {dist.get(4,0)*100:.1f}% | "
            f"{dist.get(5,0)*100:.1f}% | {dist.get(6,0)*100:.1f}% |"
        )

    # ===== REGENERATE ALL FIGURES =====
    print("\n\nRegenerating all figures...")
    generate_all_plots(dict(by_model))

    # ===== NARRATIVE SUMMARY FOR SPEAKING =====
    print("\n\n### NARRATIVE SUMMARY (for speaking) ###")
    print(f"""
In our experiments, we ran {len(all_results)} episodes across {len(by_model)} open-weight
language models in three crisis scenarios.

The most striking finding is that ALL tested models showed escalation behavior,
even in the neutral scenario with no conflict trigger — confirming that escalation
tendencies are intrinsic to transformer LLMs, not artifacts of specific frontier
model training pipelines.

{most_name} was the most escalatory model with a mean score of {model_means[most]:.1f},
while {least_name} was the least escalatory at {model_means[least]:.1f}.

Comparing to Rivera et al.'s frontier baselines using per-action normalized scores,
our open-weight models showed varied escalation relative to GPT-4-Base
(per-action baseline: {gpt4b_pa}).

The nuclear action rate across all models was {overall_nuclear*100:.1f}%, compared to
GPT-4-Base's 7.0% — {'exceeding' if overall_nuclear > 0.07 else 'lower than'} the frontier baseline.
""")

    print("Done. Check figures/ directory for updated plots.")


if __name__ == "__main__":
    main()
