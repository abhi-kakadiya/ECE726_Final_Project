"""
Main experiment runner for the open-weight wargame simulation.
Usage:
    python run_experiment.py                          # Run all models, all scenarios
    python run_experiment.py --models llama3.2        # Single model
    python run_experiment.py --scenarios neutral       # Single scenario
    python run_experiment.py --episodes 5              # Fewer episodes
    python run_experiment.py --quick                   # Quick test: 1 episode, 3 turns
"""

import argparse
import json
import os
import sys
import time

from src.config import MODELS, GROQ_MODELS, SCENARIOS, NATIONS_3, NUM_TURNS, NUM_EPISODES
from src.simulation import run_episode, save_episode
from src.llm_backend import OllamaBackend, GroqBackend, get_backend
from src.scoring import summary_table, run_all_comparisons
from src.visualization import generate_all_plots


def check_ollama():
    """Verify Ollama is running and required models are available."""
    backend = OllamaBackend("test", "http://localhost:11434")
    if not backend.is_available():
        print("ERROR: Ollama is not running!")
        print("Start it with: ollama serve")
        print("Then pull models:")
        for mk, mc in MODELS.items():
            print(f"  ollama pull {mc.ollama_id}")
        return False
    return True


def pull_models():
    """Pull all required Ollama models."""
    import subprocess
    for mk, mc in MODELS.items():
        print(f"Pulling {mc.name} ({mc.ollama_id})...")
        subprocess.run(["ollama", "pull", mc.ollama_id], check=True)
    print("All models pulled.")


def run_experiments(
    model_keys: list,
    scenario_keys: list,
    num_episodes: int,
    num_turns: int,
    results_dir: str = "results",
    groq_api_key: str = None,
    verbose: bool = True,
):
    """Run all specified experiments."""
    total_runs = len(model_keys) * len(scenario_keys) * num_episodes
    run_count = 0
    start_time = time.time()

    for model_key in model_keys:
        print(f"\n{'='*60}")
        model_config = MODELS.get(model_key) or GROQ_MODELS.get(model_key)
        print(f"MODEL: {model_config.name} ({model_config.ollama_id})")
        print(f"{'='*60}")

        backend = get_backend(model_key, groq_api_key=groq_api_key)

        for scenario_key in scenario_keys:
            print(f"\n--- Scenario: {scenario_key} ---")

            for ep_id in range(num_episodes):
                run_count += 1
                elapsed = time.time() - start_time
                eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0

                print(
                    f"\nEpisode {ep_id+1}/{num_episodes} "
                    f"[Run {run_count}/{total_runs}, "
                    f"ETA: {eta/60:.1f}min]"
                )

                # Check if result already exists
                fname = f"{model_key}_{scenario_key}_ep{ep_id}.json"
                fpath = os.path.join(results_dir, fname)
                if os.path.exists(fpath):
                    print(f"  Already exists, skipping: {fname}")
                    continue

                result = run_episode(
                    backend=backend,
                    model_name=model_key,
                    scenario_key=scenario_key,
                    episode_id=ep_id,
                    nations=NATIONS_3,
                    num_turns=num_turns,
                    verbose=verbose,
                )

                saved_path = save_episode(result, results_dir)
                print(
                    f"  Total escalation: {result.total_escalation_score} | "
                    f"Time: {result.total_latency_sec:.1f}s | "
                    f"Saved: {saved_path}"
                )

    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time/60:.1f} minutes.")


def main():
    parser = argparse.ArgumentParser(
        description="Run open-weight wargame escalation experiments"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        help="Model keys to test",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIOS.keys()),
        help="Scenario keys to test",
    )
    parser.add_argument(
        "--episodes", type=int, default=NUM_EPISODES,
        help="Number of episodes per model per scenario",
    )
    parser.add_argument(
        "--turns", type=int, default=NUM_TURNS,
        help="Number of turns per episode",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--groq-key", default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key for cloud models",
    )
    parser.add_argument(
        "--pull", action="store_true",
        help="Pull Ollama models before running",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 1 episode, 3 turns, 1 model",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip simulation, only analyze existing results",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if args.quick:
        args.models = [args.models[0]]
        args.scenarios = ["neutral"]
        args.episodes = 1
        args.turns = 3

    if not args.analyze_only:
        # Check Ollama
        ollama_models = [m for m in args.models if m in MODELS]
        if ollama_models:
            if not check_ollama():
                sys.exit(1)
            if args.pull:
                pull_models()

        # Run experiments
        run_experiments(
            model_keys=args.models,
            scenario_keys=args.scenarios,
            num_episodes=args.episodes,
            num_turns=args.turns,
            results_dir=args.results_dir,
            groq_api_key=args.groq_key,
            verbose=not args.quiet,
        )

    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    from src.simulation import load_all_results
    all_results = load_all_results(args.results_dir)

    if not all_results:
        print("No results found to analyze.")
        sys.exit(0)

    # Group by model
    results_by_model = {}
    for r in all_results:
        mk = r["model"]
        if mk not in results_by_model:
            results_by_model[mk] = []
        results_by_model[mk].append(r)

    # Summary table
    print("\n" + summary_table(results_by_model))

    # Statistical comparisons
    if len(results_by_model) > 1:
        print("\nPAIRWISE STATISTICAL COMPARISONS:")
        print("-" * 60)
        comparisons = run_all_comparisons(results_by_model)
        for comp in comparisons:
            print(f"\n{comp['comparison']}:")
            print(f"  Mean: {comp['mean_a']:.1f} vs {comp['mean_b']:.1f}")
            ks = comp["ks_test"]
            mw = comp["mann_whitney"]
            w1 = comp["wasserstein"]
            print(f"  KS test: D={ks['statistic']:.3f}, p={ks['p_value']:.4f}")
            print(f"  Mann-Whitney U: U={mw['statistic']:.1f}, p={mw['p_value']:.4f}")
            print(f"  Wasserstein-1: {w1['distance']:.2f}")

        # Save comparisons
        comp_path = os.path.join(args.results_dir, "statistical_comparisons.json")
        with open(comp_path, "w") as f:
            json.dump(comparisons, f, indent=2)
        print(f"\nComparisons saved to {comp_path}")

    # Generate plots
    generate_all_plots(results_by_model)


if __name__ == "__main__":
    main()
