"""Headless pixel environments for all six games.

The wrappers do not start a webcam and do not expose game state to the agent.
Internal fields are read only for rewards, termination and privileged teachers.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def _game_module(game_name: str):
    folder = ROOT / "games" / game_name
    path = folder / "game.py"
    if not path.exists():
        raise KeyError(f"Unknown game {game_name!r}")
    # Block Blast keeps its Piece class in a sibling module and imports it as
    # ``pieces``.  The path addition is local to loading and avoids changing
    # the original student game code.
    folder_text = str(folder)
    if folder_text not in sys.path:
        sys.path.insert(0, folder_text)
    module_name = f"fly_ai_game_{game_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PixelGameEnv:
    """Small Gym-like API without requiring Gymnasium."""

    game_name = "base"
    width = 640
    height = 480
    n_actions = 2
    action_names = ("noop", "action")

    def __init__(self, *, seed: int = 0, max_steps: int = 1500) -> None:
        pygame.init()
        pygame.font.init()
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.steps = 0
        self.screen = pygame.Surface((self.width, self.height))
        self.game: Any = None
        self._seed_all(self.seed)

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed = int(seed)
        self._seed_all(self.seed)
        self.steps = 0
        self._build_game()
        return self.render(), self.info()

    def _build_game(self) -> None:
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError

    def _validate_action(self, action: int) -> int:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} outside [0, {self.n_actions})")
        return action

    def _advance(self, dt: float, frames: int) -> None:
        for _ in range(frames):
            self.game.update(dt)

    def _draw(self) -> None:
        self.game.draw()

    def render(self) -> np.ndarray:
        self._draw()
        # surfarray uses W x H x C; the public contract is H x W x C.
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()

    def info(self) -> dict:
        return {"game": self.game_name, "step": self.steps}

    def teacher_action(self) -> int:
        raise NotImplementedError(f"No privileged teacher for {self.game_name}")

    def teacher_sample_valid(self) -> bool:
        """Whether the current frame is an informative decision point."""
        return True

    @property
    def has_teacher(self) -> bool:
        return type(self).teacher_action is not PixelGameEnv.teacher_action

    def close(self) -> None:
        pass


class KnifeHitEnv(PixelGameEnv):
    game_name = "knife_hit"
    width, height = 560, 760
    n_actions = 2
    action_names = ("wait", "throw")

    def _build_game(self) -> None:
        module = _game_module(self.game_name)
        self.module = module
        self.game = module.KnifeHitGame(self.screen, self.width, self.height)

    def step(self, action: int):
        action = self._validate_action(action)
        before_score = self.game.score
        before_level = self.game.level
        before_apples = self.game.apples_total
        before_phase = self.game.phase
        accepted = bool(
            action == 1
            and self.game.phase == self.module.PLAY
            and self.game.flying is None
            and self.game.knives_left > 0
        )
        if action == 1:
            self.game.throw()
        self._advance(1.0 / 60.0, 4)
        self.steps += 1

        reward = float(self.game.score - before_score)
        reward += 0.25 * float(self.game.apples_total - before_apples)
        reward += 2.0 * max(0, self.game.level - before_level)
        if before_phase != self.module.FAIL and self.game.phase in (self.module.FAIL, self.module.GAME_OVER):
            reward -= 2.0
        if action == 1 and not accepted:
            reward -= 0.02
        reward -= 0.001
        terminated = self.game.phase == self.module.GAME_OVER
        truncated = self.steps >= self.max_steps
        return self.render(), reward, terminated, truncated, self.info()

    def teacher_action(self) -> int:
        g, m = self.game, self.module
        if g.phase != m.PLAY or g.flying is not None or g.knives_left <= 0:
            return 0
        # Predict the impact angle over the short flight using the same angular
        # velocity function as the simulator.  This privileged teacher is used
        # only to create demonstrations; the trained policy sees pixels only.
        distance = g.start_tip_y - (g.center[1] + m.LOG_R - m.EMBED)
        remaining = max(0.0, distance / m.KNIFE_SPEED)
        rotation = float(g.rotation)
        level_t = float(g.level_t)
        dt = 1.0 / 240.0
        while remaining > 0:
            h = min(dt, remaining)
            rotation += g.cfg["direction"] * m.angular_speed(
                g.cfg["pattern"], g.cfg["speed"], level_t
            ) * h
            level_t += h
            remaining -= h
        local = (90.0 - rotation) % 360.0
        safe_gap = m.HIT_ANGLE + 3.0
        safe = all(m.angle_diff(local, stuck) >= safe_gap for stuck in g.stuck)
        return int(safe)

    def teacher_sample_valid(self) -> bool:
        g, m = self.game, self.module
        return bool(g.phase == m.PLAY and g.flying is None and g.knives_left > 0)

    def info(self) -> dict:
        return {
            **super().info(),
            "score": int(self.game.score),
            "level": int(self.game.level),
            "phase": str(self.game.phase),
            "knives_left": int(self.game.knives_left),
        }


class RaceEnv(PixelGameEnv):
    game_name = "race"
    width, height = 900, 760
    n_actions = 4
    action_names = ("lane_1", "lane_2", "lane_3", "lane_4")

    def _build_game(self) -> None:
        module = _game_module(self.game_name)
        self.module = module
        self.game = module.RaceGame(self.screen, self.width, self.height)

    def step(self, action: int):
        action = self._validate_action(action)
        before_score = float(self.game.score)
        before_phase = self.game.phase
        self.game.set_target(action / (self.n_actions - 1))
        self._advance(1.0 / 60.0, 6)
        self.steps += 1
        reward = (float(self.game.score) - before_score) / 25.0
        if before_phase != self.module.CRASH and self.game.phase in (self.module.CRASH, self.module.GAME_OVER):
            reward -= 5.0
        terminated = self.game.phase == self.module.GAME_OVER
        truncated = self.steps >= self.max_steps
        return self.render(), reward, terminated, truncated, self.info()

    def teacher_action(self) -> int:
        g, m = self.game, self.module
        if g.phase == m.COUNTDOWN_PHASE:
            return min(range(4), key=lambda lane: abs(m.LANE_CENTERS[lane] - g.player_x))
        scores = []
        for lane in range(4):
            clearance = float("inf")
            for vehicle in g.traffic:
                if vehicle.lane != lane:
                    continue
                rel = abs(vehicle.y - g.player_y) - (vehicle.length + m.PLAYER_L) / 2
                clearance = min(clearance, rel)
            # A small preference for the safer right-hand lanes prevents the
            # x2 oncoming reward from teaching suicidal driving.
            scores.append(clearance + (8.0 if lane >= 2 else 0.0))
        return int(np.argmax(scores))

    def info(self) -> dict:
        return {
            **super().info(),
            "score": float(self.game.score),
            "phase": str(self.game.phase),
            "seconds": float(self.game.time),
        }


class BlockBlastEnv(PixelGameEnv):
    game_name = "block_blast"
    width, height = 760, 760
    n_actions = 193

    @property
    def action_names(self):
        return ("wait",) + tuple(
            f"slot_{slot}_row_{row}_col_{col}"
            for slot in range(3) for row in range(8) for col in range(8)
        )

    def _build_game(self) -> None:
        module = _game_module(self.game_name)
        self.module = module
        self.game = module.BlockBlastGame(self.screen, self.width, self.height)
        self.cursor = (0, 0)

    def step(self, action: int):
        action = self._validate_action(action)
        before = int(self.game.score)
        placed = False
        if action > 0:
            idx = action - 1
            slot, rem = divmod(idx, 64)
            row, col = divmod(rem, 8)
            piece = self.game.tray[slot]
            if piece is not None:
                start = self.game.tray_slot_center(slot)
                x = self.game.grid_origin[0] + col * self.game.cell + piece.width * self.game.cell / 2
                y = self.game.grid_origin[1] + row * self.game.cell + piece.height * self.game.cell / 2
                self.cursor = (x, y)
                self.game.on_press(start)
                self.game.on_drag(self.cursor)
                placed = bool(self.game.preview_valid)
                self.game.on_release(self.cursor)
        self.game.update()
        self.steps += 1
        delta = int(self.game.score) - before
        reward = delta / 8.0 + (0.05 if placed else (-0.03 if action else -0.002))
        terminated = bool(self.game.game_over)
        truncated = self.steps >= self.max_steps
        return self.render(), float(reward), terminated, truncated, self.info()

    def _draw(self) -> None:
        self.game.draw(self.cursor)

    def info(self) -> dict:
        return {**super().info(), "score": int(self.game.score), "game_over": bool(self.game.game_over)}


class DartsEnv(PixelGameEnv):
    game_name = "darts"
    width, height = 1100, 760
    n_actions = 10
    action_names = ("wait", "bull", "top", "upper_right", "right", "lower_right",
                    "bottom", "lower_left", "left", "upper_left")

    def _build_game(self) -> None:
        module = _game_module(self.game_name)
        self.module = module
        self.game = module.DartsGame(self.screen, self.width, self.height)
        cx, cy = module.BOARD_CENTER
        radius = module.R * 0.62
        self.targets = [(cx, cy)] + [
            (cx + np.cos(angle) * radius, cy + np.sin(angle) * radius)
            for angle in np.linspace(-np.pi / 2, 3 * np.pi / 2, 8, endpoint=False)
        ]

    def step(self, action: int):
        action = self._validate_action(action)
        before = int(self.game.total)
        if action > 0 and self.game.phase == self.module.AIM:
            pos = self.targets[action - 1]
            self.game.on_press(pos)
            self.game.on_release(pos, self.module.ThrowMotion())
        elif action > 0 and self.game.phase == self.module.THROW:
            pos = self.game.aim or self.targets[action - 1]
            self.game.on_press(pos)
            self.game.on_release(pos, self.module.ThrowMotion(vy=-self.module.IDEAL_POWER))
        self._advance(1.0 / 60.0, 6)
        self.steps += 1
        reward = (int(self.game.total) - before) / 20.0 - 0.001
        terminated = self.game.phase == self.module.GAME_OVER
        truncated = self.steps >= self.max_steps
        return self.render(), float(reward), terminated, truncated, self.info()

    def _draw(self) -> None:
        self.game.draw((0, 0), False, None)

    def info(self) -> dict:
        return {**super().info(), "score": int(self.game.total), "round": int(self.game.round), "phase": str(self.game.phase)}


class SpiderEnv(PixelGameEnv):
    game_name = "spider"
    width, height = 1100, 800
    n_actions = 6
    action_names = ("wait", "left", "right", "take_or_drop", "deal", "undo")

    def _build_game(self) -> None:
        module = _game_module(self.game_name)
        self.module = module
        self.game = module.SpiderGame(self.screen, self.width, self.height)
        self.game.reset(suits=1)

    def step(self, action: int):
        action = self._validate_action(action)
        before_completed = len(self.game.completed)
        before_score = int(self.game.score)
        before_face_up = sum(card.face_up for col in self.game.columns for card in col)
        before_moves = int(self.game.moves)
        if action == 1:
            self.game.move_selection(-1)
        elif action == 2:
            self.game.move_selection(1)
        elif action == 3:
            self.game.action()
        elif action == 4:
            self.game.deal()
        elif action == 5:
            self.game.undo()
        self.game.update(0.1)
        self.steps += 1
        completed = len(self.game.completed) - before_completed
        face_up = sum(card.face_up for col in self.game.columns for card in col) - before_face_up
        moved = int(self.game.moves) != before_moves
        reward = completed * 20.0 + face_up * 0.3 + (0.02 if moved else -0.01)
        reward += max(-2, int(self.game.score) - before_score) * 0.01
        if self.game.won:
            reward += 100.0
        terminated = bool(self.game.won)
        truncated = self.steps >= self.max_steps
        return self.render(), float(reward), terminated, truncated, self.info()

    def info(self) -> dict:
        return {
            **super().info(),
            "score": int(self.game.score),
            "moves": int(self.game.moves),
            "completed": len(self.game.completed),
            "won": bool(self.game.won),
        }


class SplinterEnv(PixelGameEnv):
    game_name = "splinter"
    width, height = 1000, 720
    n_actions = 18
    action_names = tuple(
        f"{'pinch' if pinch else 'open'}_{direction}"
        for pinch in (0, 1)
        for direction in ("stay", "up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left")
    )
    _directions = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))

    def _build_game(self) -> None:
        module = _game_module(self.game_name)
        self.module = module
        self.game = module.SplinterGame(self.screen, self.width, self.height)

    def step(self, action: int):
        action = self._validate_action(action)
        before_removed = sum(s.removed for s in self.game.splinters)
        before_pain = float(self.game.pain)
        before_level = int(self.game.level)
        before_phase = self.game.phase
        pinching, direction = divmod(action, 9)
        dx, dy = self._directions[direction]
        x = min(self.width - 1, max(0, self.game.cursor[0] + dx * 18))
        y = min(self.height - 1, max(0, self.game.cursor[1] + dy * 18))
        pos = (x, y)
        if pinching and not self.game.pinching:
            self.game.on_press(pos)
        elif pinching:
            self.game.on_drag(pos)
        elif self.game.pinching:
            self.game.on_release(pos)
        else:
            self.game.on_hover(pos)
        self.game.update(0.05)
        self.steps += 1
        level_advanced = self.game.level > before_level
        # next_level() replaces the splinter list. Counts from different levels
        # are not comparable and used to turn the level-completion bonus negative.
        removed = 0 if level_advanced else max(
            0, sum(s.removed for s in self.game.splinters) - before_removed
        )
        reward = removed * 8.0 - max(0.0, self.game.pain - before_pain) * 0.08 - 0.002
        reward += max(0, self.game.level - before_level) * 15.0
        if before_phase != self.module.FAILED and self.game.phase == self.module.FAILED:
            reward -= 8.0
        terminated = self.game.phase == self.module.FAILED
        truncated = self.steps >= self.max_steps
        return self.render(), float(reward), terminated, truncated, self.info()

    def info(self) -> dict:
        return {
            **super().info(),
            "level": int(self.game.level),
            "pain": float(self.game.pain),
            "removed": int(sum(s.removed for s in self.game.splinters)),
            "phase": str(self.game.phase),
        }


from .flight3d import Flight3DEnv
from .neon_drive import NeonDriveEnv


ENV_REGISTRY = {
    "flight_3d": Flight3DEnv,
    "neon_drive": NeonDriveEnv,
    "knife_hit": KnifeHitEnv,
    "race": RaceEnv,
    "block_blast": BlockBlastEnv,
    "darts": DartsEnv,
    "spider": SpiderEnv,
    "splinter": SplinterEnv,
}


def make_env(name: str, **kwargs) -> PixelGameEnv:
    try:
        env_class = ENV_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown game {name!r}; choose from {', '.join(ENV_REGISTRY)}") from exc
    return env_class(**kwargs)
