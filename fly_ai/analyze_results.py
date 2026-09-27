"""Recompute paired Race evaluation statistics from an episodes CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_phase(rows: list[dict[str, str]], phase: str) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["phase"] != phase:
            continue
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"duplicate {phase} seed: {seed}")
        result[seed] = {
            "score": float(row["score"]),
            "steps": float(row["steps"]),
        }
    return result


def analyze(path: Path, bootstrap_reps: int = 50_000, seed: int = 2026) -> dict:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    random_eval = _read_phase(rows, "random_eval")
    agent_eval = _read_phase(rows, "agent_eval")
    if not random_eval or random_eval.keys() != agent_eval.keys():
        raise ValueError("random_eval and agent_eval must contain the same non-empty seed set")

    seeds = sorted(agent_eval)
    random_scores = np.array([random_eval[s]["score"] for s in seeds], dtype=np.float64)
    agent_scores = np.array([agent_eval[s]["score"] for s in seeds], dtype=np.float64)
    random_steps = np.array([random_eval[s]["steps"] for s in seeds], dtype=np.float64)
    agent_steps = np.array([agent_eval[s]["steps"] for s in seeds], dtype=np.float64)
    differences = agent_scores - random_scores

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(bootstrap_reps, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    random_mean = float(random_scores.mean())
    agent_mean = float(agent_scores.mean())
    return {
        "csv": str(path),
        "paired_seeds": len(seeds),
        "random_mean_score": random_mean,
        "agent_mean_score": agent_mean,
        "random_median_score": float(np.median(random_scores)),
        "agent_median_score": float(np.median(agent_scores)),
        "relative_score_gain_percent": float((agent_mean / random_mean - 1.0) * 100.0),
        "random_mean_steps": float(random_steps.mean()),
        "agent_mean_steps": float(agent_steps.mean()),
        "relative_steps_gain_percent": float((agent_steps.mean() / random_steps.mean() - 1.0) * 100.0),
        "paired_agent_wins": int(np.count_nonzero(differences > 0)),
        "paired_mean_score_difference": float(differences.mean()),
        "bootstrap_repetitions": int(bootstrap_reps),
        "bootstrap_seed": int(seed),
        "paired_mean_difference_ci95": [float(ci_low), float(ci_high)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")
    print(json.dumps(analyze(args.csv_path, args.bootstrap_reps, args.seed), indent=2))


if __name__ == "__main__":
    main()
