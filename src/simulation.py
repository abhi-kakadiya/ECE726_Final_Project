"""
Core simulation engine for the wargame experiment.

Runs multi-nation, multi-turn episodes where each LLM agent controls
a nation and picks actions from the 27-action menu each turn. The
engine handles prompt assembly, LLM calls, action resolution, and
result serialization.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.config import (
    ACTIONS, Nation, NATIONS_3, SCENARIOS,
    NUM_TURNS, TEMPERATURE, MAX_TOKENS, SEED_BASE,
    escalation_score,
)
from src.prompts import build_system_prompt, build_world_prompt, build_turn_prompt
from src.llm_backend import LLMResponse

# Maps nation name -> int offset for deterministic per-nation seeding.
# Covers all 8 Rivera nations even though we only use 3.
_NATION_SEED_OFFSET = {
    "Blue": 0, "Red": 1, "Orange": 2,
    "Purple": 3, "White": 4, "Pink": 5,
    "Yellow": 6, "Green": 7,
}


@dataclass
class TurnAction:
    """Single action taken by one nation in one turn."""
    nation: str
    turn: int
    action_name: str
    action_number: int
    severity: int
    escalation_score: int
    target: Optional[str]
    reasoning: str
    raw_response: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class EpisodeResult:
    """Complete record of one simulation episode."""
    model: str
    scenario: str
    episode_id: int
    seed: int
    nations: List[str]
    num_turns: int
    actions: List[TurnAction]
    total_escalation_score: int = 0
    per_nation_scores: Dict[str, int] = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    total_latency_sec: float = 0.0

    def compute_scores(self):
        self.total_escalation_score = sum(a.escalation_score for a in self.actions)
        self.per_nation_scores = {}
        for n in self.nations:
            self.per_nation_scores[n] = sum(
                a.escalation_score for a in self.actions if a.nation == n
            )


def resolve_action(parsed: Optional[Dict]) -> tuple:
    """
    Map raw LLM output to a valid action from the 27-action menu.
    Returns (action_name, action_number, severity, target).

    Resolution priority: action_number > exact name match > fuzzy name.
    Falls back to 'Wait' (action #8, severity 2) if nothing matches.
    """
    if parsed is None:
        return "Wait", 8, 2, None

    action_num = parsed.get("action_number")
    action_name = parsed.get("action_name", "")
    target = parsed.get("target_nation")

    # Try by number first — most reliable
    if isinstance(action_num, int) and 1 <= action_num <= 27:
        action = ACTIONS[action_num - 1]
        return action.name, action_num, action.severity, target

    # Exact name match
    action_name_clean = action_name.strip().lower()
    for i, a in enumerate(ACTIONS):
        if a.name.lower() == action_name_clean:
            return a.name, i + 1, a.severity, target

    # Fuzzy: substring match (handles cases like "nuclear strike"
    # matching "Tactical nuclear strike")
    for i, a in enumerate(ACTIONS):
        if action_name_clean in a.name.lower() or a.name.lower() in action_name_clean:
            return a.name, i + 1, a.severity, target

    return "Wait", 8, 2, None


def run_episode(
    backend,
    model_name: str,
    scenario_key: str,
    episode_id: int,
    nations: Optional[List[Nation]] = None,
    num_turns: int = NUM_TURNS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> EpisodeResult:
    """Run one complete wargame episode (all turns, all nations)."""

    if nations is None:
        nations = NATIONS_3
    scenario = SCENARIOS[scenario_key]
    if seed is None:
        seed = SEED_BASE + episode_id

    result = EpisodeResult(
        model=model_name,
        scenario=scenario_key,
        episode_id=episode_id,
        seed=seed,
        nations=[n.name for n in nations],
        num_turns=num_turns,
        actions=[],
        start_time=datetime.now().isoformat(),
    )

    # System prompts are fixed per nation for the whole episode
    system_prompts = {n.name: build_system_prompt(n) for n in nations}
    world_prompt = build_world_prompt(scenario, nations)

    # Shared action history — all nations see all prior actions
    action_history: List[Dict] = []

    for turn in range(1, num_turns + 1):
        if verbose:
            print(f"  Turn {turn}/{num_turns}", end=" ", flush=True)

        turn_actions = []

        for nation in nations:
            turn_prompt = build_turn_prompt(nation, nations, turn, action_history)
            user_prompt = world_prompt + "\n\n" + turn_prompt

            # Each nation gets a unique seed per turn for reproducibility
            nation_seed = seed + turn * 100 + _NATION_SEED_OFFSET.get(nation.name, 0)

            response: LLMResponse = backend.generate(
                system_prompt=system_prompts[nation.name],
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=nation_seed,
            )

            action_name, action_num, severity, target = resolve_action(response.parsed)
            score = escalation_score(severity)

            reasoning = ""
            if response.parsed:
                reasoning = response.parsed.get("reasoning", "")

            turn_action = TurnAction(
                nation=nation.name,
                turn=turn,
                action_name=action_name,
                action_number=action_num,
                severity=severity,
                escalation_score=score,
                target=target,
                reasoning=reasoning,
                raw_response=response.raw_text,
                latency_sec=response.latency_sec,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            turn_actions.append(turn_action)
            result.actions.append(turn_action)

        # All nations' actions become visible next turn
        action_history.append({
            "turn": turn,
            "actions": [
                {
                    "nation": a.nation,
                    "action": a.action_name,
                    "target": a.target,
                }
                for a in turn_actions
            ],
        })

        if verbose:
            summary = " | ".join(f"{a.nation}:{a.action_name}" for a in turn_actions)
            print(f"[{summary}]")

    result.end_time = datetime.now().isoformat()
    result.total_latency_sec = sum(a.latency_sec for a in result.actions)
    result.compute_scores()

    return result


def save_episode(result: EpisodeResult, output_dir: str = "results"):
    """Serialize episode result to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{result.model}_{result.scenario}_ep{result.episode_id}.json"
    fpath = os.path.join(output_dir, fname)

    data = {
        "model": result.model,
        "scenario": result.scenario,
        "episode_id": result.episode_id,
        "seed": result.seed,
        "nations": result.nations,
        "num_turns": result.num_turns,
        "total_escalation_score": result.total_escalation_score,
        "per_nation_scores": result.per_nation_scores,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "total_latency_sec": result.total_latency_sec,
        "actions": [
            {
                "nation": a.nation,
                "turn": a.turn,
                "action_name": a.action_name,
                "action_number": a.action_number,
                "severity": a.severity,
                "escalation_score": a.escalation_score,
                "target": a.target,
                "reasoning": a.reasoning,
                "raw_response": a.raw_response,
                "latency_sec": a.latency_sec,
                "prompt_tokens": a.prompt_tokens,
                "completion_tokens": a.completion_tokens,
            }
            for a in result.actions
        ],
    }

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return fpath


def load_episode(fpath: str) -> Dict:
    """Load a single episode JSON."""
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results(results_dir: str = "results") -> List[Dict]:
    """Load all episode JSONs from a results directory.

    Only picks up files matching the pattern model_scenario_epN.json,
    so summary or metadata files in the same directory are skipped.
    """
    results = []
    if not os.path.exists(results_dir):
        return results
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        if "_ep" not in fname:
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            data = load_episode(fpath)
            if "actions" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results
