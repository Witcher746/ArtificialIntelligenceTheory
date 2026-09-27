"""Run a deterministic smoke benchmark for the RGB Neon Drive controller."""

from __future__ import annotations

import argparse
import json

from .neon_drive import NeonDriveCVController, NeonDriveEnv


def evaluate(episodes: int, max_steps: int, seed: int) -> dict:
    rows = []
    for episode in range(episodes):
        env = NeonDriveEnv(seed=seed + episode, max_steps=max_steps)
        driver = NeonDriveCVController()
        frame, _ = env.reset(seed + episode)
        total_reward = 0.0
        for step in range(max_steps):
            frame, reward, terminated, truncated, info = env.step(driver.act(frame))
            total_reward += reward
            if terminated or truncated:
                break
        rows.append({
            "seed": seed + episode,
            "steps": step + 1,
            "distance": info["distance"],
            "checkpoints": info["checkpoints"],
            "speed": info["speed"],
            "crashed": bool(terminated),
            "crash_reason": info["crash_reason"],
            "driver_offset": driver.target_offset,
            "driver_danger": driver.danger,
            "reward": total_reward,
        })
    return {
        "episodes": rows,
        "mean_distance": sum(row["distance"] for row in rows) / len(rows),
        "mean_checkpoints": sum(row["checkpoints"] for row in rows) / len(rows),
        "crash_rate": sum(row["crashed"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.episodes, args.max_steps, args.seed), indent=2))


if __name__ == "__main__":
    main()
