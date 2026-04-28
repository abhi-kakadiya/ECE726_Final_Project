"""
Plot generation for wargame experiment results.

Produces 7 types of figures: escalation time-series, boxplots, severity
distributions, nuclear rate comparison, scenario comparison, heatmaps,
and action frequency charts. All saved to figures/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for batch rendering
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from src.config import SEVERITY_LABELS, MODELS
from src.scoring import (
    aggregate_episode_scores,
    aggregate_per_turn_mean,
    aggregate_severity_distribution,
    aggregate_nuclear_rate,
    compute_action_frequency,
    FRONTIER_BASELINES,
)

sns.set_theme(style="whitegrid", font_scale=1.2)

# One color per model key for consistency across all plots
MODEL_COLORS = {
    "llama3.2": "#1f77b4",
    "phi3.5": "#ff7f0e",
    "gemma2": "#2ca02c",
    "qwen2.5": "#d62728",
    "llama3.1-8b": "#9467bd",
    "llama3.1-70b": "#8c564b",
}
FRONTIER_COLOR = "#888888"
FIGURES_DIR = "figures"


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _model_label(key: str) -> str:
    """Display name for a model key."""
    return MODELS[key].name if key in MODELS else key


def _model_color(key: str) -> str:
    return MODEL_COLORS.get(key, "#333333")


def _scenario_title(scenario: Optional[str]) -> str:
    if scenario:
        return f" — {scenario.replace('_', ' ').title()} Scenario"
    return ""


# -- Plot 1: Escalation over time --

def plot_escalation_over_time(
    results_by_model: Dict[str, List[Dict]],
    scenario: Optional[str] = None,
    save: bool = True,
) -> plt.Figure:
    """Line plot showing mean escalation score per turn for each model."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_key, episodes in sorted(results_by_model.items()):
        if scenario:
            episodes = [e for e in episodes if e["scenario"] == scenario]
        if not episodes:
            continue

        mean_scores, std_scores = aggregate_per_turn_mean(episodes)
        turns = np.arange(1, len(mean_scores) + 1)
        color = _model_color(model_key)

        ax.plot(turns, mean_scores, marker="o", color=color,
                label=_model_label(model_key), linewidth=2)
        ax.fill_between(turns, mean_scores - std_scores,
                        mean_scores + std_scores, alpha=0.15, color=color)

    ax.set_xlabel("Turn (Day)")
    ax.set_ylabel("Escalation Score")
    ax.set_title("Mean Escalation Score Over Time" + _scenario_title(scenario))
    ax.legend(loc="upper left")
    ax.set_xticks(range(1, 15))

    plt.tight_layout()
    if save:
        suffix = f"_{scenario}" if scenario else ""
        fig.savefig(f"{FIGURES_DIR}/escalation_over_time{suffix}.png", dpi=150)
        plt.close(fig)
    return fig


# -- Plot 2: Total escalation boxplot --

def plot_total_escalation_boxplot(
    results_by_model: Dict[str, List[Dict]],
    scenario: Optional[str] = None,
    save: bool = True,
) -> plt.Figure:
    """Box plot comparing total escalation score distributions."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    data, labels, colors = [], [], []

    for model_key in sorted(results_by_model.keys()):
        episodes = results_by_model[model_key]
        if scenario:
            episodes = [e for e in episodes if e["scenario"] == scenario]
        if not episodes:
            continue
        data.append(aggregate_episode_scores(episodes))
        labels.append(_model_label(model_key))
        colors.append(_model_color(model_key))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Total Escalation Score (per episode)")
    ax.set_title("Total Escalation Score Distribution" + _scenario_title(scenario))

    plt.tight_layout()
    if save:
        suffix = f"_{scenario}" if scenario else ""
        fig.savefig(f"{FIGURES_DIR}/escalation_boxplot{suffix}.png", dpi=150)
        plt.close(fig)
    return fig


# -- Plot 3: Severity distribution (stacked bar) --

def plot_severity_distribution(
    results_by_model: Dict[str, List[Dict]],
    scenario: Optional[str] = None,
    save: bool = True,
) -> plt.Figure:
    """Stacked bar chart of severity bin distributions per model."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(12, 6))

    model_keys = sorted(results_by_model.keys())
    x = np.arange(len(model_keys))
    width = 0.6

    distributions = {}
    for mk in model_keys:
        episodes = results_by_model[mk]
        if scenario:
            episodes = [e for e in episodes if e["scenario"] == scenario]
        distributions[mk] = aggregate_severity_distribution(episodes)

    # Color scale from green (de-escalation) through red to purple (nuclear)
    severity_colors = {
        1: "#2ecc71", 2: "#95a5a6", 3: "#f39c12",
        4: "#e67e22", 5: "#e74c3c", 6: "#8e44ad",
    }

    bottom = np.zeros(len(model_keys))
    for sev in range(1, 7):
        values = [distributions[mk].get(sev, 0) for mk in model_keys]
        ax.bar(x, values, width, bottom=bottom,
               label=f"{sev}: {SEVERITY_LABELS[sev]}", color=severity_colors[sev])
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(mk) for mk in model_keys], rotation=15)
    ax.set_ylabel("Fraction of Actions")
    ax.set_title("Action Severity Distribution by Model" + _scenario_title(scenario))
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save:
        suffix = f"_{scenario}" if scenario else ""
        fig.savefig(f"{FIGURES_DIR}/severity_distribution{suffix}.png", dpi=150)
        plt.close(fig)
    return fig


# -- Plot 4: Nuclear rate comparison (us vs. frontier) --

def plot_nuclear_rate_comparison(
    results_by_model: Dict[str, List[Dict]],
    save: bool = True,
) -> plt.Figure:
    """Bar chart: nuclear action rates for our models + Rivera's frontier baselines."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Our open-weight models
    model_keys = sorted(results_by_model.keys())
    rates = [aggregate_nuclear_rate(results_by_model[mk]) * 100 for mk in model_keys]
    labels = [_model_label(mk) for mk in model_keys]
    colors = [_model_color(mk) for mk in model_keys]

    # Append frontier baselines
    for name, data in FRONTIER_BASELINES.items():
        labels.append(name)
        rates.append(data["nuclear_rate"] * 100)
        colors.append(FRONTIER_COLOR)

    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Visual divider between our models and frontier
    divider = len(model_keys) - 0.5
    ax.axvline(x=divider, color="black", linestyle=":", alpha=0.5)
    text_y = max(rates) * 0.95 if max(rates) > 0 else 1.0
    ax.text(divider - 0.3, text_y, "Open-weight", ha="right", fontsize=10)
    ax.text(divider + 0.3, text_y, "Frontier", ha="left", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Nuclear Action Rate (%)")
    ax.set_title("Nuclear Action Rate: Open-Weight vs. Frontier Models")

    plt.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/nuclear_rate_comparison.png", dpi=150)
        plt.close(fig)
    return fig


# -- Plot 5: Scenario comparison (grouped bar) --

def plot_scenario_comparison(
    results_by_model: Dict[str, List[Dict]],
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart: mean escalation by model, grouped by scenario."""
    _ensure_dir()
    scenarios = ["neutral", "invasion", "cyber_attack"]
    model_keys = sorted(results_by_model.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_keys))
    width = 0.25
    scenario_colors = {"neutral": "#3498db", "invasion": "#e74c3c", "cyber_attack": "#f39c12"}

    for i, sc in enumerate(scenarios):
        means, stds = [], []
        for mk in model_keys:
            eps = [e for e in results_by_model[mk] if e["scenario"] == sc]
            if eps:
                scores = aggregate_episode_scores(eps)
                means.append(scores.mean())
                stds.append(scores.std())
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=sc.replace("_", " ").title(),
               color=scenario_colors[sc], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([_model_label(mk) for mk in model_keys], rotation=15)
    ax.set_ylabel("Mean Total Escalation Score")
    ax.set_title("Escalation by Model and Scenario")
    ax.legend()

    plt.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/scenario_comparison.png", dpi=150)
        plt.close(fig)
    return fig


# -- Plot 6: Escalation heatmap (model x turn) --

def plot_escalation_heatmap(
    results_by_model: Dict[str, List[Dict]],
    scenario: Optional[str] = None,
    save: bool = True,
) -> plt.Figure:
    """Heatmap: rows = models, columns = turns, cell = mean escalation."""
    _ensure_dir()
    model_keys = sorted(results_by_model.keys())
    num_turns = 14

    data_matrix = []
    labels = []
    for mk in model_keys:
        eps = results_by_model[mk]
        if scenario:
            eps = [e for e in eps if e["scenario"] == scenario]
        if not eps:
            continue
        mean_scores, _ = aggregate_per_turn_mean(eps, num_turns)
        data_matrix.append(mean_scores)
        labels.append(_model_label(mk))

    if not data_matrix:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(num_turns))
    ax.set_xticklabels([str(t + 1) for t in range(num_turns)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Turn (Day)")
    ax.set_ylabel("Model")
    ax.set_title("Escalation Heatmap (Mean Score per Turn)" + _scenario_title(scenario))
    plt.colorbar(im, ax=ax, label="Escalation Score")

    plt.tight_layout()
    if save:
        suffix = f"_{scenario}" if scenario else ""
        fig.savefig(f"{FIGURES_DIR}/escalation_heatmap{suffix}.png", dpi=150)
        plt.close(fig)
    return fig


# -- Plot 7: Top actions by frequency --

def plot_action_frequency_top(
    results_by_model: Dict[str, List[Dict]],
    top_n: int = 10,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of most frequently chosen actions, one panel per model."""
    _ensure_dir()
    fig, axes = plt.subplots(1, len(results_by_model),
                             figsize=(5 * len(results_by_model), 8))

    if len(results_by_model) == 1:
        axes = [axes]

    for ax, (mk, episodes) in zip(axes, sorted(results_by_model.items())):
        all_actions = []
        for ep in episodes:
            all_actions.extend(ep["actions"])
        freq = compute_action_frequency(all_actions)
        top_actions = list(freq.items())[:top_n]
        names = [a[0] for a in reversed(top_actions)]
        counts = [a[1] for a in reversed(top_actions)]

        ax.barh(names, counts, color=_model_color(mk), alpha=0.8)
        ax.set_xlabel("Frequency")
        ax.set_title(_model_label(mk))

    fig.suptitle(f"Top {top_n} Actions by Model", fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/action_frequency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# -- Generate everything --

def generate_all_plots(results_by_model: Dict[str, List[Dict]]):
    """Produce all 7 plot types and save to figures/ directory."""
    _ensure_dir()
    print("Generating plots...")

    plot_total_escalation_boxplot(results_by_model)
    print("  [1/7] Boxplot done")

    plot_nuclear_rate_comparison(results_by_model)
    print("  [2/7] Nuclear rate comparison done")

    plot_scenario_comparison(results_by_model)
    print("  [3/7] Scenario comparison done")

    for sc in ["neutral", "invasion", "cyber_attack"]:
        plot_escalation_over_time(results_by_model, scenario=sc)
    print("  [4/7] Escalation over time done")

    for sc in ["neutral", "invasion", "cyber_attack"]:
        plot_severity_distribution(results_by_model, scenario=sc)
    print("  [5/7] Severity distribution done")

    for sc in ["neutral", "invasion", "cyber_attack"]:
        plot_escalation_heatmap(results_by_model, scenario=sc)
    print("  [6/7] Heatmaps done")

    plot_action_frequency_top(results_by_model)
    print("  [7/7] Action frequency done")

    print(f"All plots saved to {FIGURES_DIR}/")
