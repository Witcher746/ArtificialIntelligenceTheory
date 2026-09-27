"""Crash-resumable overnight DQN training for Neon Drive."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import torch
from torch import nn
from torch.nn import functional as F

from .neon_dqn import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, NeonDQN, NeonFrameEncoder
from .neon_drive import NeonDriveCVController, NeonDriveEnv


CSV_FIELDS = (
    "timestamp", "phase", "episode", "global_step", "reward", "steps", "distance",
    "checkpoints", "overtakes", "near_misses", "weaves", "crash_reason", "epsilon",
    "teacher_probability", "mean_loss", "eval_score",
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def prevent_windows_sleep(enable: bool) -> None:
    if os.name != "nt":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    flags = es_continuous | es_system_required if enable else es_continuous
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def gpu_temperature() -> int | None:
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
            creationflags=flags,
        )
        return int(float(result.stdout.splitlines()[0].strip()))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class DiskReplayBuffer:
    """Bounded disk-backed replay that survives process and desktop crashes."""

    def __init__(self, root: Path, capacity: int) -> None:
        self.root = root
        self.capacity = int(capacity)
        self.shape = (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        expected = {"capacity": self.capacity, "shape": list(self.shape), "version": 1}
        fresh = not manifest_path.exists()
        if fresh:
            atomic_json(manifest_path, expected)
        else:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError(f"replay schema mismatch: expected {expected}, got {actual}")
        mode = "w+" if fresh else "r+"
        replay_shape = (self.capacity, *self.shape)
        self.states = np.memmap(root / "states.uint8", dtype=np.uint8, mode=mode, shape=replay_shape)
        self.next_states = np.memmap(root / "next_states.uint8", dtype=np.uint8, mode=mode, shape=replay_shape)
        self.actions = np.memmap(root / "actions.uint8", dtype=np.uint8, mode=mode, shape=(self.capacity,))
        self.rewards = np.memmap(root / "rewards.float32", dtype=np.float32, mode=mode, shape=(self.capacity,))
        self.dones = np.memmap(root / "dones.uint8", dtype=np.uint8, mode=mode, shape=(self.capacity,))
        self.expert = np.memmap(root / "expert.uint8", dtype=np.uint8, mode=mode, shape=(self.capacity,))
        self.position = 0
        self.size = 0

    def restore(self, position: int, size: int) -> None:
        self.position = int(position) % self.capacity
        self.size = int(np.clip(size, 0, self.capacity))

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
            done: bool, expert: bool) -> None:
        index = self.position
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.expert[index] = expert
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            np.asarray(self.states[indices]),
            np.asarray(self.actions[indices], dtype=np.int64),
            np.asarray(self.rewards[indices], dtype=np.float32),
            np.asarray(self.next_states[indices]),
            np.asarray(self.dones[indices], dtype=np.float32),
            np.asarray(self.expert[indices], dtype=np.bool_),
        )

    def flush(self) -> None:
        for array in (self.states, self.next_states, self.actions, self.rewards, self.dones, self.expert):
            array.flush()


def greedy_action(model: NeonDQN, state: np.ndarray, device: torch.device) -> int:
    with torch.inference_mode():
        tensor = torch.from_numpy(state[None]).to(device, non_blocking=True)
        return int(model(tensor).argmax(dim=1).item())


def optimise(
    model: NeonDQN,
    target: NeonDQN,
    optimizer: torch.optim.Optimizer,
    scaler,
    replay: DiskReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
    batch_size: int,
    gamma: float,
    tau: float,
) -> float:
    states, actions, rewards, next_states, dones, expert = replay.sample(batch_size, rng)
    states_t = torch.from_numpy(states).to(device, non_blocking=True)
    actions_t = torch.from_numpy(actions).to(device, non_blocking=True)
    rewards_t = torch.from_numpy(rewards).to(device, non_blocking=True)
    next_states_t = torch.from_numpy(next_states).to(device, non_blocking=True)
    dones_t = torch.from_numpy(dones).to(device, non_blocking=True)
    expert_t = torch.from_numpy(expert).to(device, non_blocking=True)
    use_amp = device.type == "cuda"
    amp_context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    optimizer.zero_grad(set_to_none=True)
    with amp_context:
        all_q = model(states_t)
        chosen_q = all_q.gather(1, actions_t[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = model(next_states_t).argmax(dim=1)
            next_q = target(next_states_t).gather(1, next_actions[:, None]).squeeze(1)
            expected_q = rewards_t + gamma * (1.0 - dones_t) * next_q
        td_loss = F.smooth_l1_loss(chosen_q, expected_q)
        imitation_loss = (F.cross_entropy(all_q[expert_t], actions_t[expert_t])
                          if bool(expert_t.any()) else torch.zeros((), device=device))
        loss = td_loss + 0.08 * imitation_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), model.parameters()):
            target_parameter.lerp_(parameter, tau)
    return float(loss.detach().item())


def evaluate(model: NeonDQN, device: torch.device, seed_base: int, episodes: int,
             max_steps: int) -> tuple[float, list[dict]]:
    env = NeonDriveEnv(seed=seed_base, max_steps=max_steps)
    encoder = NeonFrameEncoder()
    rows: list[dict] = []
    was_training = model.training
    model.eval()
    for episode in range(episodes):
        frame, _ = env.reset(seed_base + episode)
        encoder.reset()
        state = encoder.encode(frame)
        total = 0.0
        for step in range(max_steps):
            action = greedy_action(model, state, device)
            frame, reward, terminated, truncated, info = env.step(action)
            state = encoder.encode(frame)
            total += reward
            if terminated or truncated:
                break
        rows.append({"reward": total, **info})
    if was_training:
        model.train()
    env.close()
    scores = [row["score"] for row in rows]
    return float(np.mean(scores)), rows


def save_checkpoint(path: Path, model: NeonDQN, target: NeonDQN, optimizer,
                    scaler, episode: int, global_step: int, replay: DiskReplayBuffer,
                    rng: np.random.Generator, best_eval: float, args: argparse.Namespace) -> None:
    replay.flush()
    payload = {
        "version": 1,
        "game": "neon_drive",
        "pixel_only_policy": True,
        "n_actions": 7,
        "model_state": model.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "episode": int(episode),
        "global_step": int(global_step),
        "replay_position": int(replay.position),
        "replay_size": int(replay.size),
        "rng_state": rng.bit_generator.state,
        "best_eval": float(best_eval),
        "config": vars(args),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def epsilon_at(step: int, decay_steps: int) -> float:
    progress = min(1.0, step / max(1, decay_steps))
    return float(1.0 + progress * (0.05 - 1.0))


def teacher_probability_at(step: int, teacher_steps: int) -> float:
    progress = min(1.0, step / max(1, teacher_steps))
    return float(0.75 * (1.0 - progress))


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    latest_path = output / "checkpoint_latest.pt"
    best_path = output / "best.pt"
    csv_path = output / "episodes.csv"
    replay = DiskReplayBuffer(output / "replay", args.replay_capacity)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = NeonDQN(7).to(device)
    target = NeonDQN(7).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    episode = 0
    global_step = 0
    best_eval = float("-inf")
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        target.load_state_dict(checkpoint["target_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        episode = int(checkpoint["episode"])
        global_step = int(checkpoint["global_step"])
        best_eval = float(checkpoint.get("best_eval", best_eval))
        replay.restore(checkpoint["replay_position"], checkpoint["replay_size"])
        rng.bit_generator.state = checkpoint["rng_state"]
        print(f"RESUME episode={episode} step={global_step} replay={replay.size}", flush=True)

    env = NeonDriveEnv(seed=args.seed, max_steps=args.max_steps)
    encoder = NeonFrameEncoder()
    started = time.monotonic()
    deadline = started + args.hours * 3600.0
    last_checkpoint = started
    last_temperature_check = 0.0
    prevent_windows_sleep(True)
    atomic_json(output / "run_manifest.json", {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config": vars(args),
        "pixel_only_policy": True,
        "reward_signals": ["speed", "portal", "overtake", "near_miss", "weave", "crash"],
    })

    try:
        while time.monotonic() < deadline and not (output / "STOP").exists():
            frame, info = env.reset(args.seed + episode)
            encoder.reset()
            state = encoder.encode(frame)
            teacher = NeonDriveCVController()
            total_reward = 0.0
            losses: list[float] = []
            terminated = truncated = False
            epsilon = epsilon_at(global_step, args.epsilon_decay_steps)
            teacher_probability = teacher_probability_at(global_step, args.teacher_steps)
            for step in range(args.max_steps):
                now = time.monotonic()
                if now >= deadline:
                    truncated = True
                    break
                if now - last_temperature_check >= 45.0:
                    temperature = gpu_temperature()
                    last_temperature_check = now
                    if temperature is not None and temperature >= args.thermal_limit:
                        print(f"THERMAL PAUSE {temperature}C", flush=True)
                        while temperature is not None and temperature > args.thermal_resume:
                            time.sleep(20.0)
                            temperature = gpu_temperature()
                epsilon = epsilon_at(global_step, args.epsilon_decay_steps)
                teacher_probability = teacher_probability_at(global_step, args.teacher_steps)
                expert = bool(rng.random() < teacher_probability)
                if expert:
                    action = teacher.act(frame)
                elif rng.random() < epsilon:
                    action = int(rng.integers(env.n_actions))
                else:
                    action = greedy_action(model, state, device)
                next_frame, reward, terminated, truncated, info = env.step(action)
                next_state = encoder.encode(next_frame)
                done = terminated or truncated
                replay.add(state, action, float(np.clip(reward, -10.0, 10.0)), next_state, done, expert)
                state = next_state
                frame = next_frame
                total_reward += reward
                global_step += 1
                if replay.size >= max(args.warmup_steps, args.batch_size):
                    for _ in range(args.updates_per_step):
                        losses.append(optimise(
                            model, target, optimizer, scaler, replay, rng, device,
                            args.batch_size, args.gamma, args.tau,
                        ))
                if done:
                    break

            episode += 1
            mean_loss = float(np.mean(losses)) if losses else 0.0
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phase": "train",
                "episode": episode,
                "global_step": global_step,
                "reward": total_reward,
                "steps": step + 1,
                "distance": info.get("distance", 0.0),
                "checkpoints": info.get("checkpoints", 0),
                "overtakes": info.get("overtakes", 0),
                "near_misses": info.get("near_misses", 0),
                "weaves": info.get("weaves", 0),
                "crash_reason": info.get("crash_reason", ""),
                "epsilon": epsilon,
                "teacher_probability": teacher_probability,
                "mean_loss": mean_loss,
                "eval_score": "",
            }
            append_csv(csv_path, row)
            print(
                f"EP {episode:05d} step={global_step:07d} reward={total_reward:7.2f} "
                f"dist={row['distance']:7.1f} overtakes={row['overtakes']} "
                f"weaves={row['weaves']} eps={epsilon:.3f} loss={mean_loss:.4f}",
                flush=True,
            )

            eval_score = None
            if episode % args.eval_every == 0:
                eval_score, eval_rows = evaluate(
                    model, device, args.seed + 90_000, args.eval_episodes, args.max_steps
                )
                eval_summary = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": "eval",
                    "episode": episode,
                    "global_step": global_step,
                    "reward": float(np.mean([item["reward"] for item in eval_rows])),
                    "steps": float(np.mean([item["step"] for item in eval_rows])),
                    "distance": float(np.mean([item["distance"] for item in eval_rows])),
                    "checkpoints": float(np.mean([item["checkpoints"] for item in eval_rows])),
                    "overtakes": float(np.mean([item["overtakes"] for item in eval_rows])),
                    "near_misses": float(np.mean([item["near_misses"] for item in eval_rows])),
                    "weaves": float(np.mean([item["weaves"] for item in eval_rows])),
                    "crash_reason": "",
                    "epsilon": 0.0,
                    "teacher_probability": 0.0,
                    "mean_loss": mean_loss,
                    "eval_score": eval_score,
                }
                append_csv(csv_path, eval_summary)
                print(f"EVAL episode={episode} mean_score={eval_score:.3f}", flush=True)
                if eval_score > best_eval:
                    best_eval = eval_score
                    save_checkpoint(
                        best_path, model, target, optimizer, scaler, episode, global_step,
                        replay, rng, best_eval, args,
                    )
                    print(f"BEST {best_eval:.3f} -> {best_path}", flush=True)

            elapsed = time.monotonic() - started
            atomic_json(status_path, {
                "state": "RUNNING",
                "pid": os.getpid(),
                "episode": episode,
                "global_step": global_step,
                "replay_size": replay.size,
                "last_reward": total_reward,
                "last_distance": row["distance"],
                "last_overtakes": row["overtakes"],
                "last_weaves": row["weaves"],
                "epsilon": epsilon,
                "teacher_probability": teacher_probability,
                "best_eval_score": None if best_eval == float("-inf") else best_eval,
                "elapsed_hours": elapsed / 3600.0,
                "remaining_hours": max(0.0, (deadline - time.monotonic()) / 3600.0),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            if (episode <= 3 or episode % args.checkpoint_every == 0
                    or time.monotonic() - last_checkpoint >= 300.0):
                save_checkpoint(
                    latest_path, model, target, optimizer, scaler, episode, global_step,
                    replay, rng, best_eval, args,
                )
                last_checkpoint = time.monotonic()

        save_checkpoint(
            latest_path, model, target, optimizer, scaler, episode, global_step,
            replay, rng, best_eval, args,
        )
        atomic_json(status_path, {
            "state": "COMPLETE" if time.monotonic() >= deadline else "STOPPED",
            "pid": os.getpid(),
            "episode": episode,
            "global_step": global_step,
            "replay_size": replay.size,
            "best_eval_score": None if best_eval == float("-inf") else best_eval,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    except BaseException as exc:
        atomic_json(status_path, {
            "state": "FAILED",
            "pid": os.getpid(),
            "episode": episode,
            "global_step": global_step,
            "error": repr(exc),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise
    finally:
        env.close()
        replay.flush()
        prevent_windows_sleep(False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--output-dir", default="artifacts/fly_ai/neon_night")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--epsilon-decay-steps", type=int, default=180_000)
    parser.add_argument("--teacher-steps", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--thermal-limit", type=int, default=83)
    parser.add_argument("--thermal-resume", type=int, default=77)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hours <= 0 or args.max_steps <= 0 or args.replay_capacity < args.batch_size:
        raise SystemExit("invalid positive runtime/training parameters")
    run(args)


if __name__ == "__main__":
    main()
