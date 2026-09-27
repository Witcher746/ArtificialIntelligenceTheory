"""Train and evaluate the virtual fly from rendered pixels.

Example:
    python -m fly_ai.train --game knife_hit --teacher-episodes 30 --rl-episodes 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Training renders to off-screen surfaces and must not create an OS window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from .brain import MushroomBodyAgent
from .envs import ENV_REGISTRY, make_env
from .vision import make_eye


@dataclass
class EpisodeResult:
    phase: str
    episode: int
    seed: int
    reward: float
    steps: int
    score: float
    level: int
    terminated: bool
    truncated: bool


def _summary(phase: str, episode: int, seed: int, total: float, steps: int,
             terminated: bool, truncated: bool, info: dict) -> EpisodeResult:
    return EpisodeResult(
        phase=phase,
        episode=episode,
        seed=seed,
        reward=float(total),
        steps=steps,
        score=float(info.get("score", info.get("removed", 0.0))),
        level=int(info.get("level", info.get("completed", 0))),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def teacher_training(env, eye, agent, episodes: int, seed_base: int) -> tuple[list[EpisodeResult], float]:
    results: list[EpisodeResult] = []
    losses: list[float] = []
    # Keep a bounded, class-separated replay buffer.  Decision actions such as
    # THROW are naturally much rarer than WAIT, so training on the raw stream
    # would produce a policy that never acts.
    buffer_limit = 5000
    buffers: list[list[np.ndarray]] = [[] for _ in range(env.n_actions)]
    seen = np.zeros(env.n_actions, dtype=np.int64)
    for episode in range(episodes):
        seed = seed_base + episode
        frame, _ = env.reset(seed=seed)
        eye.reset()
        total = 0.0
        terminated = truncated = False
        for step in range(env.max_steps):
            observation = eye.encode(frame)
            features = agent.features(observation)
            action = env.teacher_action()
            if env.teacher_sample_valid():
                seen[action] += 1
                if len(buffers[action]) < buffer_limit:
                    buffers[action].append(features.copy())
                else:
                    replace = int(agent.rng.integers(seen[action]))
                    if replace < buffer_limit:
                        buffers[action][replace] = features.copy()
            frame, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        results.append(_summary("teacher", episode, seed, total, step + 1, terminated, truncated, info))

    nonempty = [action for action, samples in enumerate(buffers) if samples]
    if nonempty:
        updates = min(80_000, max(20_000, 20 * max(len(buffers[action]) for action in nonempty)))
        for _ in range(updates):
            action = int(agent.rng.choice(nonempty))
            samples = buffers[action]
            features = samples[int(agent.rng.integers(len(samples)))]
            losses.append(agent.imitate(features, action))
        # Demonstrations are replayed with equal class probability, therefore
        # a learned global prior is a sampling artefact rather than evidence.
        # Keep decisions visual by centering the MBON output biases.
        agent.action_bias.fill(0.0)
    return results, float(np.mean(losses)) if losses else 0.0


def reinforcement_training(env, eye, agent, episodes: int, seed_base: int,
                           epsilon_start: float, epsilon_end: float) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for episode in range(episodes):
        seed = seed_base + episode
        frame, _ = env.reset(seed=seed)
        eye.reset()
        features = agent.features(eye.encode(frame))
        total = 0.0
        terminated = truncated = False
        epsilon = epsilon_end if episodes <= 1 else (
            epsilon_start + (epsilon_end - epsilon_start) * episode / (episodes - 1)
        )
        for step in range(env.max_steps):
            action = agent.act(features, epsilon=epsilon)
            next_frame, reward, terminated, truncated, info = env.step(action)
            next_features = agent.features(eye.encode(next_frame))
            agent.learn_td(features, action, reward, next_features, terminated or truncated)
            features = next_features
            total += reward
            if terminated or truncated:
                break
        results.append(_summary("reinforcement", episode, seed, total, step + 1, terminated, truncated, info))
    return results


def evaluate(env, eye, agent, episodes: int, seed_base: int, *, random_policy: bool = False) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    rng = np.random.default_rng(seed_base + 991)
    phase = "random_eval" if random_policy else "agent_eval"
    for episode in range(episodes):
        seed = seed_base + episode
        frame, _ = env.reset(seed=seed)
        eye.reset()
        total = 0.0
        terminated = truncated = False
        for step in range(env.max_steps):
            features = agent.features(eye.encode(frame))
            action = int(rng.integers(env.n_actions)) if random_policy else agent.act(features)
            frame, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        results.append(_summary(phase, episode, seed, total, step + 1, terminated, truncated, info))
    return results


def evaluate_teacher(env, episodes: int, seed_base: int) -> list[EpisodeResult]:
    """Measure the privileged demonstrator as an explicit upper baseline."""
    if not env.has_teacher:
        return []
    results: list[EpisodeResult] = []
    for episode in range(episodes):
        seed = seed_base + episode
        _, _ = env.reset(seed=seed)
        total = 0.0
        terminated = truncated = False
        for step in range(env.max_steps):
            _, reward, terminated, truncated, info = env.step(env.teacher_action())
            total += reward
            if terminated or truncated:
                break
        results.append(_summary("teacher_eval", episode, seed, total, step + 1, terminated, truncated, info))
    return results


def aggregate(rows: list[EpisodeResult]) -> dict:
    if not rows:
        return {"episodes": 0}
    return {
        "episodes": len(rows),
        "mean_reward": float(np.mean([r.reward for r in rows])),
        "mean_score": float(np.mean([r.score for r in rows])),
        "median_score": float(np.median([r.score for r in rows])),
        "mean_level": float(np.mean([r.level for r in rows])),
        "mean_steps": float(np.mean([r.steps for r in rows])),
        "termination_rate": float(np.mean([r.terminated for r in rows])),
    }


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.game, seed=args.seed, max_steps=args.max_steps)
    eye = make_eye(args.game)
    agent = MushroomBodyAgent(
        eye.output_dim,
        env.n_actions,
        n_kenyon=args.kenyon,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    rows: list[EpisodeResult] = []
    started = time.time()

    teacher_episodes = args.teacher_episodes if env.has_teacher else 0
    teacher_rows, mean_loss = teacher_training(env, eye, agent, teacher_episodes, args.seed)
    rows.extend(teacher_rows)
    rows.extend(reinforcement_training(
        env, eye, agent, args.rl_episodes, args.seed + 10_000,
        args.epsilon_start, args.epsilon_end,
    ))

    # Test seeds are deliberately disjoint from training seeds.
    random_rows = evaluate(env, eye, agent, args.eval_episodes, args.seed + 50_000, random_policy=True)
    agent_rows = evaluate(env, eye, agent, args.eval_episodes, args.seed + 50_000)
    rows.extend(random_rows)
    rows.extend(agent_rows)

    model_path = output_dir / f"{args.game}_fly_agent.npz"
    metadata = {
        "game": args.game,
        "action_names": list(env.action_names),
        "pixel_only_policy": True,
        "vision": {"facets_per_eye": eye.facets, "channels": ["contrast", "motion"]},
        "training_seed": args.seed,
    }
    agent.save(model_path, metadata)

    csv_path = output_dir / f"{args.game}_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    report = {
        "game": args.game,
        "model": str(model_path),
        "episodes_csv": str(csv_path),
        "pixel_only_policy": True,
        "teacher_used_only_during_training": bool(teacher_episodes),
        "teacher_episodes": teacher_episodes,
        "reinforcement_episodes": args.rl_episodes,
        "evaluation_episodes": args.eval_episodes,
        "mean_imitation_loss": mean_loss,
        "random": aggregate(random_rows),
        "trained": aggregate(agent_rows),
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
    }
    report_path = output_dir / f"{args.game}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compound-eye virtual fly on a Pygame game")
    parser.add_argument("--game", choices=sorted(ENV_REGISTRY), default="knife_hit")
    parser.add_argument("--teacher-episodes", type=int, default=30,
                        help="privileged demonstrations; currently available for knife_hit and race")
    parser.add_argument("--rl-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--kenyon", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--epsilon-start", type=float, default=0.08)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/fly_ai")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.teacher_episodes < 0 or args.rl_episodes < 0 or args.eval_episodes <= 0:
        raise SystemExit("Episode counts must be non-negative and eval-episodes must be positive")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
