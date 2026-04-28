"""
Re-score existing experiment JSON files with corrected severity bin mapping.
Actions 9-11 (International arbitration, Formal peace negotiations, Form alliance)
were incorrectly assigned severity=2 (score=0), should be severity=3 (score=4).

This updates severity and escalation_score for affected actions in-place,
then recalculates total_escalation_score and per_nation_scores.
"""

import json
import os
import sys
from collections import defaultdict

from src.config import ACTION_BY_NAME, escalation_score

RESULTS_DIR = "results"


def rescore_episode(filepath: str) -> dict:
    """Re-score a single episode JSON file. Returns summary of changes."""
    with open(filepath, "r") as f:
        data = json.load(f)

    changes = 0
    for action in data["actions"]:
        name = action["action_name"]
        if name in ACTION_BY_NAME:
            correct_sev = ACTION_BY_NAME[name].severity
            correct_score = escalation_score(correct_sev)
            if action["severity"] != correct_sev or action["escalation_score"] != correct_score:
                changes += 1
                action["severity"] = correct_sev
                action["escalation_score"] = correct_score

    # Recalculate totals
    data["total_escalation_score"] = sum(a["escalation_score"] for a in data["actions"])
    per_nation = defaultdict(int)
    for a in data["actions"]:
        per_nation[a["nation"]] += a["escalation_score"]
    data["per_nation_scores"] = dict(per_nation)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return {"file": os.path.basename(filepath), "changes": changes, "new_total": data["total_escalation_score"]}


def main():
    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".json") and "_ep" in f)
    print(f"Re-scoring {len(files)} episode files...")

    total_changes = 0
    for fname in files:
        path = os.path.join(RESULTS_DIR, fname)
        result = rescore_episode(path)
        if result["changes"] > 0:
            print(f"  {result['file']}: {result['changes']} actions updated, new total={result['new_total']}")
            total_changes += result["changes"]

    print(f"\nDone. {total_changes} total action scores corrected across {len(files)} files.")


if __name__ == "__main__":
    main()
