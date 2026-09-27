"""A small mushroom-body-like reinforcement learning policy.

The visual projection is fixed and sparse, similar in spirit to expansion from
projection neurons to Kenyon cells.  A trainable MBON readout receives signed
prediction errors: positive errors are the engineering analogue of appetitive
dopamine, negative errors of aversive reinforcement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MushroomBodyAgent:
    """Sparse visual expansion followed by a linear action-value readout."""

    format_version = 1

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        *,
        n_kenyon: int = 4096,
        fan_in: int = 12,
        active_fraction: float = 0.15,
        gamma: float = 0.97,
        learning_rate: float = 0.025,
        seed: int = 42,
    ) -> None:
        if input_dim < fan_in:
            raise ValueError("input_dim must be at least fan_in")
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.n_kenyon = int(n_kenyon)
        self.fan_in = int(fan_in)
        self.active_fraction = float(active_fraction)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.kc_indices = np.empty((self.n_kenyon, self.fan_in), dtype=np.int32)
        for row in range(self.n_kenyon):
            self.kc_indices[row] = self.rng.choice(self.input_dim, self.fan_in, replace=False)
        self.kc_weights = self.rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(self.n_kenyon, self.fan_in),
        ) / np.sqrt(self.fan_in)
        self.mbon = np.zeros((self.n_kenyon, self.n_actions), dtype=np.float32)
        self.action_bias = np.zeros(self.n_actions, dtype=np.float32)
        self.updates = 0

    def features(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (self.input_dim,):
            raise ValueError(f"Expected observation ({self.input_dim},), got {x.shape}")
        activation = np.sum(x[self.kc_indices] * self.kc_weights, axis=1)
        np.maximum(activation, 0.0, out=activation)
        # A small direct PN→KC subpopulation preserves fine angular detail;
        # the rest remains a fixed random expansion. Positive and negative
        # contrast are represented by separate non-negative cells.
        direct = min(self.input_dim, self.n_kenyon // 2)
        activation[:direct] = np.maximum(x[:direct], 0.0)
        activation[direct:2 * direct] = np.maximum(-x[:direct], 0.0)

        active = max(1, int(round(self.n_kenyon * self.active_fraction)))
        if active < self.n_kenyon:
            threshold = np.partition(activation, -active)[-active]
            activation[activation < threshold] = 0.0
        norm = float(np.linalg.norm(activation))
        if norm > 1e-8:
            activation /= norm
        return activation.astype(np.float32, copy=False)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return features @ self.mbon + self.action_bias

    def act(self, features: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q_values(features)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def imitate(self, features: np.ndarray, action: int, *, weight: float = 1.0) -> float:
        """One weighted softmax step from a privileged teacher demonstration."""
        logits = self.q_values(features)
        logits = logits - float(logits.max())
        probs = np.exp(logits).astype(np.float32)
        probs /= float(probs.sum())
        loss = -float(np.log(max(float(probs[action]), 1e-8)))
        grad = -probs
        grad[action] += 1.0
        step = self.learning_rate * float(weight)
        active = np.flatnonzero(features)
        self.mbon[active] += step * features[active, None] * grad[None, :]
        self.action_bias += step * 0.05 * grad
        self.updates += 1
        return loss

    def learn_td(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ) -> float:
        current = float(self.q_values(features)[action])
        bootstrap = 0.0 if done else self.gamma * float(self.q_values(next_features).max())
        td_error = float(np.clip(reward + bootstrap - current, -10.0, 10.0))
        active = np.flatnonzero(features)
        # TD fine-tuning is intentionally conservative so a handful of
        # terminal penalties cannot erase the balanced visual imitation prior.
        td_step = self.learning_rate * 0.1
        self.mbon[active, action] += td_step * td_error * features[active]
        self.updates += 1
        return td_error

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "format_version": self.format_version,
            "input_dim": self.input_dim,
            "n_actions": self.n_actions,
            "n_kenyon": self.n_kenyon,
            "fan_in": self.fan_in,
            "active_fraction": self.active_fraction,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "updates": self.updates,
            "metadata": metadata or {},
        }
        np.savez_compressed(
            path,
            config=np.array(json.dumps(config, ensure_ascii=False)),
            kc_indices=self.kc_indices,
            kc_weights=self.kc_weights,
            mbon=self.mbon,
            action_bias=self.action_bias,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple["MushroomBodyAgent", dict]:
        with np.load(Path(path), allow_pickle=False) as data:
            config = json.loads(str(data["config"].item()))
            agent = cls(
                config["input_dim"],
                config["n_actions"],
                n_kenyon=config["n_kenyon"],
                fan_in=config["fan_in"],
                active_fraction=config["active_fraction"],
                gamma=config["gamma"],
                learning_rate=config["learning_rate"],
                seed=config["seed"],
            )
            agent.kc_indices = data["kc_indices"].copy()
            agent.kc_weights = data["kc_weights"].copy()
            agent.mbon = data["mbon"].copy()
            agent.action_bias = data["action_bias"].copy()
            agent.updates = int(config.get("updates", 0))
        return agent, dict(config.get("metadata", {}))
