"""Run a deterministic smoke benchmark for the RGB flight autopilot."""

from __future__ import annotations

import argparse
import json

from .flight3d import Flight3DEnv, FlightCVController


def evaluate(episodes: int, max_steps: int, seed: int) -> dict:
    rows = []
    for episode in range(episodes):
        env = Flight3DEnv(seed=seed + episode, max_steps=max_steps)
        controller = FlightCVController()
        frame, _ = env.reset(seed + episode)
        total_reward = 0.0
        for step in range(max_steps):
            action = controller.act(frame)
            frame, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rows.append({
            "seed": seed + episode,
            "steps": step + 1,
            "rings": info["rings"],
            "missed": info["missed"],
            "crashed": bool(terminated),
            "reward": total_reward,
        })
    return {
        "episodes": rows,
        "mean_rings": sum(row["rings"] for row in rows) / len(rows),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "crash_rate": sum(row["crashed"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.episodes, args.max_steps, args.seed), indent=2))


if __name__ == "__main__":
    main()
