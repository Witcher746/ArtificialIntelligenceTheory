"""Interactive endless viewer for the virtual fly and all game adapters."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .brain import MushroomBodyAgent
from .envs import ENV_REGISTRY, make_env
from .fly3d import FlyRenderer3D
from .flight3d import FlightCVController
from .gpu_brain import GpuBrain, PROFILE_ORDER
from .neon_drive import NeonDriveCVController
from .neural_viz import NeuralMapRenderer
from .vision import make_eye


ROOT = Path(__file__).resolve().parents[1]
GAME_ORDER = ("flight_3d", "neon_drive", "race", "knife_hit", "block_blast", "darts", "spider", "splinter")
GAME_LABELS = {
    "flight_3d": "3D-полёт",
    "neon_drive": "Neon Drive",
    "race": "Гонка",
    "knife_hit": "Knife Hit",
    "block_blast": "Block Blast",
    "darts": "Дартс",
    "spider": "Паук",
    "splinter": "Занозы",
}


def facet_grid(values: np.ndarray) -> np.ndarray:
    """Lay 721 biological-style facets onto a 38 x 19 display grid."""
    grid = np.zeros(38 * 19, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    grid[: min(flat.size, grid.size)] = flat[: grid.size]
    return grid.reshape(19, 38)


class LiveSession:
    """One game session that automatically restarts after every episode."""

    def __init__(self, game: str = "flight_3d", seed: int = 80_000) -> None:
        self.base_seed = int(seed)
        self.episode = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = 0
        self.last_observation: np.ndarray | None = None
        self.last_features: np.ndarray | None = None
        self.last_q: np.ndarray | None = None
        self.motor_state = np.zeros(7, dtype=np.float32)
        self.activity = 0.0
        self.done_since: float | None = None
        self.rng = np.random.default_rng(seed)
        self.switch(game)

    def switch(self, game: str) -> None:
        if game not in ENV_REGISTRY:
            raise KeyError(game)
        if hasattr(self, "env"):
            self.env.close()
        self.game = game
        self.env = make_env(game, seed=self.base_seed, max_steps=1800)
        self.eye = make_eye(game)
        self.agent: MushroomBodyAgent | None = None
        self.cv_controller: FlightCVController | None = None
        self.drive_controller: NeonDriveCVController | None = None
        self.neural_drive = None
        self.policy_is_cv = False
        self.policy_label = "СЛУЧАЙНЫЙ BASELINE"
        checkpoint = ROOT / "artifacts" / "fly_ai" / f"{game}_fly_agent.npz"
        if game == "flight_3d":
            self.cv_controller = FlightCVController()
            self.policy_label = "RGB CV AUTOPILOT"
            self.policy_is_cv = True
        elif game == "neon_drive":
            compact_checkpoint = ROOT / "artifacts" / "fly_ai" / "neon_drive_dqn_best.pt"
            full_checkpoint = ROOT / "artifacts" / "fly_ai" / "neon_night" / "best.pt"
            neural_checkpoint = compact_checkpoint if compact_checkpoint.exists() else full_checkpoint
            if neural_checkpoint.exists():
                try:
                    from .neon_dqn import NeonDQNPolicy
                    self.neural_drive = NeonDQNPolicy(neural_checkpoint)
                    self.policy_label = "LEARNED PIXEL DQN"
                except (OSError, RuntimeError, KeyError, ValueError):
                    self.drive_controller = NeonDriveCVController()
                    self.policy_label = "RGB NEON DRIVER"
            else:
                self.drive_controller = NeonDriveCVController()
                self.policy_label = "RGB NEON DRIVER"
            self.policy_is_cv = True
        elif checkpoint.exists():
            candidate, metadata = MushroomBodyAgent.load(checkpoint)
            if (metadata.get("game") == game and candidate.input_dim == self.eye.output_dim
                    and candidate.n_actions == self.env.n_actions):
                self.agent = candidate
                self.policy_label = "ОБУЧЕННАЯ CV POLICY"
                self.policy_is_cv = True
        elif self.env.has_teacher:
            self.policy_label = "ДЕМО-ЭКСПЕРТ (НЕ CV)"
        self.episode = 0
        self._reset()

    def _reset(self) -> None:
        seed = self.base_seed + self.episode
        self.frame, self.info = self.env.reset(seed=seed)
        self.eye.reset()
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = 0
        self.last_observation = None
        self.last_features = None
        self.last_q = None
        self.motor_state = np.zeros(7, dtype=np.float32)
        self.activity = 0.0
        self.done_since = None
        if self.neural_drive is not None:
            self.neural_drive.reset()

    def restart(self) -> None:
        self.episode += 1
        self._reset()

    def _baseline_action(self) -> int:
        if self.neural_drive is not None:
            return int(self.neural_drive.act(self.frame))
        if self.cv_controller is not None:
            return int(self.cv_controller.act(self.frame))
        if self.drive_controller is not None:
            return int(self.drive_controller.act(self.frame))
        if self.env.has_teacher:
            return int(self.env.teacher_action())
        if self.game == "block_blast":
            return 0 if self.rng.random() < 0.12 else int(self.rng.integers(1, self.env.n_actions))
        if self.game == "darts":
            return 0 if self.rng.random() < 0.20 else int(self.rng.integers(1, self.env.n_actions))
        if self.game == "spider":
            return int(self.rng.choice(self.env.n_actions, p=(0.05, 0.20, 0.20, 0.35, 0.15, 0.05)))
        if self.game == "splinter":
            return int(self.rng.integers(self.env.n_actions))
        return int(self.rng.integers(self.env.n_actions))

    def advance(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.done_since is not None:
            if now - self.done_since >= 1.25:
                self.restart()
            return

        observation = self.eye.encode(self.frame)
        self.last_observation = observation
        motion = observation[721:1442] if observation.size >= 1442 else observation
        self.activity = float(np.clip(np.mean(np.abs(motion)) * 5.0, 0.0, 1.0))
        if self.agent is not None:
            self.last_features = self.agent.features(observation)
            self.last_q = self.agent.q_values(self.last_features)
            action = self.agent.act(self.last_features)
        else:
            action = self._baseline_action()
            self.last_features = self._synthetic_kenyon(observation)
            self.last_q = np.zeros(self.env.n_actions, dtype=np.float32)
            if self.cv_controller is not None:
                confidence = self.cv_controller.confidence
            elif self.neural_drive is not None:
                confidence = self.neural_drive.confidence
            elif self.drive_controller is not None:
                confidence = self.drive_controller.confidence
            else:
                confidence = 0.45
            self.last_q[action] = 0.4 + confidence
        self.motor_state = self._motor_for_action(int(action))
        self.frame, reward, terminated, truncated, self.info = self.env.step(action)
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        if terminated or truncated:
            self.done_since = now

    @property
    def action_name(self) -> str:
        names = self.env.action_names
        return str(names[self.last_action]) if 0 <= self.last_action < len(names) else str(self.last_action)

    @staticmethod
    def _synthetic_kenyon(observation: np.ndarray) -> np.ndarray:
        chunks = np.array_split(np.abs(observation.astype(np.float32)), 32)
        pooled = np.asarray([float(chunk.mean()) for chunk in chunks], dtype=np.float32)
        rows = np.arange(1, 97, dtype=np.float32)[:, None]
        cols = np.arange(1, 33, dtype=np.float32)[None, :]
        basis = np.sin(rows * cols * 0.173) + np.cos(rows * cols * 0.071)
        activation = np.maximum(0.0, basis @ pooled)
        threshold = np.partition(activation, -15)[-15]
        activation[activation < threshold] = 0.0
        maximum = float(activation.max())
        return activation / maximum if maximum > 1e-8 else activation

    def _motor_for_action(self, action: int) -> np.ndarray:
        name = str(self.env.action_names[action]).lower()
        left = -1.0 if "left" in name else (1.0 if "right" in name else 0.0)
        vertical = 1.0 if "up" in name else (-1.0 if "down" in name else 0.0)
        effort = 0.55 + 0.45 * self.activity
        if self.game == "flight_3d":
            legs = (0.15, 0.10, 0.15)
            wings = (float(np.clip(effort - left * 0.18, 0.0, 1.0)),
                     float(np.clip(effort + left * 0.18, 0.0, 1.0)))
        else:
            pulse = 1.0 if any(token in name for token in ("throw", "pinch", "take", "deal")) else effort
            wings = (effort, effort)
            legs = (pulse, 0.35 + abs(left) * 0.55, 0.25 + abs(vertical) * 0.65)
        return np.asarray((*wings, *legs, left, vertical), dtype=np.float32)

    def brain_activity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sensory = self.last_observation if self.last_observation is not None else np.zeros(1443, np.float32)
        kenyon = self.last_features if self.last_features is not None else np.zeros(96, np.float32)
        mbon = self.last_q if self.last_q is not None else np.zeros(self.env.n_actions, np.float32)
        return sensory, kenyon, mbon, self.motor_state


class FlyStudio:
    background = (7, 11, 17)
    panel = (15, 22, 31)
    panel_alt = (20, 31, 43)
    text = (226, 235, 242)
    muted = (135, 153, 168)
    cyan = (68, 201, 218)
    amber = (245, 181, 67)

    def __init__(self, start_game: str = "flight_3d", gpu_profile: str = "rtx") -> None:
        pygame.init()
        pygame.font.init()
        display = pygame.display.Info()
        width = min(1440, max(1000, display.current_w - 80))
        height = min(880, max(680, display.current_h - 120))
        self.windowed_size = (width, height)
        self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        pygame.display.set_caption("Виртуальная муха — CV Studio")
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small = pygame.font.SysFont("segoeui", 14)
        self.title_font = pygame.font.SysFont("segoeui", 26, bold=True)
        self.big = pygame.font.SysFont("segoeui", 34, bold=True)
        self.clock = pygame.time.Clock()
        self.session = LiveSession(start_game)
        self.fly = FlyRenderer3D()
        self.brain = NeuralMapRenderer()
        self.gpu_brain = GpuBrain(gpu_profile)
        self.running = True
        self.paused = False
        self.fullscreen = False
        self.speed_index = 1
        self.speeds = (0.5, 1.0, 2.0, 4.0)
        self.accumulator = 0.0
        self.decision_hz = 15.0
        self.game_buttons: dict[str, pygame.Rect] = {}
        self.control_buttons: dict[str, pygame.Rect] = {}
        self.gpu_button = pygame.Rect(0, 0, 1, 1)
        self.fly_rect = pygame.Rect(0, 0, 1, 1)
        self.fly_yaw = 0.0
        self.fly_pitch = 0.0
        self.dragging_fly = False
        self.last_mouse = (0, 0)

    def _text(self, value: object, color=None, font=None) -> pygame.Surface:
        return (font or self.font).render(str(value), True, color or self.text)

    def _button(self, rect: pygame.Rect, label: str, active: bool = False) -> None:
        hover = rect.collidepoint(pygame.mouse.get_pos())
        color = self.cyan if active else ((36, 56, 70) if hover else self.panel_alt)
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, (61, 82, 96), rect, 1, border_radius=10)
        text = self._text(label, (8, 20, 27) if active else self.text, self.small)
        self.screen.blit(text, text.get_rect(center=rect.center))

    @staticmethod
    def _fit(source: pygame.Surface, target: pygame.Rect) -> tuple[pygame.Surface, pygame.Rect]:
        ratio = min(target.width / source.get_width(), target.height / source.get_height())
        size = (max(1, int(source.get_width() * ratio)), max(1, int(source.get_height() * ratio)))
        scaled = pygame.transform.smoothscale(source, size)
        return scaled, scaled.get_rect(center=target.center)

    def _layout(self) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
        width, height = self.screen.get_size()
        top = pygame.Rect(16, 14, width - 32, 54)
        content_top = 78
        content_bottom = height - 48
        sidebar_w = 220 if width >= 1180 else 190
        fly_w = 350 if width >= 1280 else 290
        sidebar = pygame.Rect(16, content_top, sidebar_w, content_bottom - content_top)
        fly = pygame.Rect(width - 16 - fly_w, content_top, fly_w, content_bottom - content_top)
        game = pygame.Rect(sidebar.right + 12, content_top, fly.left - sidebar.right - 24, content_bottom - content_top)
        return top, sidebar, game, fly

    def _draw_sidebar(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, self.panel, rect, border_radius=16)
        x, y = rect.x + 14, rect.y + 14
        self.screen.blit(self._text("РЕЖИМ ИГРЫ", self.muted, self.small), (x, y))
        y += 30
        self.game_buttons.clear()
        button_height = 34
        button_step = 38
        for index, game in enumerate(GAME_ORDER):
            button = pygame.Rect(x, y, rect.width - 28, button_height)
            self.game_buttons[game] = button
            self._button(button, f"{index + 1}. {GAME_LABELS[game]}", game == self.session.game)
            y += button_step

        y += 8
        self.screen.blit(self._text("УПРАВЛЕНИЕ", self.muted, self.small), (x, y))
        y += 28
        pause = pygame.Rect(x, y, rect.width - 28, 38)
        restart = pygame.Rect(x, y + 45, rect.width - 28, 38)
        minus = pygame.Rect(x, y + 98, 42, 36)
        plus = pygame.Rect(rect.right - 56, y + 98, 42, 36)
        self.control_buttons = {"pause": pause, "restart": restart, "minus": minus, "plus": plus}
        self._button(pause, "Продолжить" if self.paused else "Пауза")
        self._button(restart, "Новый эпизод")
        self._button(minus, "−")
        self._button(plus, "+")
        speed = self._text(f"Скорость {self.speeds[self.speed_index]:g}×", self.text, self.small)
        self.screen.blit(speed, speed.get_rect(center=(rect.centerx, minus.centery)))

        help_y = y + 144
        self.screen.blit(self._text("1–8 · Space пауза · G GPU", self.muted, self.small), (x, help_y))
        if rect.bottom - help_y >= 44:
            self.screen.blit(self._text("R рестарт · F11 экран · Esc выход", self.muted, self.small),
                             (x, help_y + 20))

    def _draw_game(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, self.panel, rect, border_radius=16)
        viewport = rect.inflate(-18, -74)
        viewport.y += 42
        viewport.height -= 16
        frame_surface = pygame.surfarray.make_surface(self.session.frame.transpose(1, 0, 2))
        scaled, destination = self._fit(frame_surface, viewport)
        self.screen.blit(scaled, destination)
        pygame.draw.rect(self.screen, (60, 78, 91), destination, 1, border_radius=4)
        name = self._text(GAME_LABELS[self.session.game], self.text, self.title_font)
        self.screen.blit(name, (rect.x + 14, rect.y + 7))

        if self.session.done_since is not None:
            shade = pygame.Surface(destination.size, pygame.SRCALPHA)
            shade.fill((2, 6, 10, 160))
            self.screen.blit(shade, destination)
            message = self._text("ЭПИЗОД ЗАВЕРШЁН", self.amber, self.big)
            self.screen.blit(message, message.get_rect(center=destination.center))
            sub = self._text("автоматический перезапуск…", self.text, self.font)
            self.screen.blit(sub, sub.get_rect(center=(destination.centerx, destination.centery + 43)))
        elif self.paused:
            message = self._text("ПАУЗА", self.amber, self.big)
            self.screen.blit(message, message.get_rect(center=destination.center))

        footer = pygame.Rect(rect.x + 14, rect.bottom - 29, rect.width - 28, 22)
        score = self.session.info.get("score", self.session.info.get("level", 0))
        status = f"Эпизод {self.session.episode + 1}   Счёт {float(score):.1f}   Награда {self.session.total_reward:.2f}"
        self.screen.blit(self._text(status, self.muted, self.small), footer.topleft)

    def _draw_sensor(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, self.panel_alt, rect, border_radius=12)
        self.screen.blit(self._text("COMPOUND EYE · 721 FACETS", self.muted, self.small), (rect.x + 10, rect.y + 8))
        observation = self.session.last_observation
        if observation is None or observation.size < 1442:
            return
        values = facet_grid(observation[:721])
        motion = facet_grid(observation[721:1442])
        gap = 5
        grid_w = (rect.width - 25) // 2
        cell_w = max(1, grid_w // 38)
        cell_h = max(1, (rect.height - 38) // 19)
        for grid_index, grid in enumerate((values, motion)):
            origin_x = rect.x + 10 + grid_index * (grid_w + gap)
            origin_y = rect.y + 30
            for row in range(19):
                for col in range(38):
                    value = float(grid[row, col])
                    if grid_index == 0:
                        level = int(np.clip((value + 1.0) * 105.0, 0, 220))
                        color = (20, level, min(255, level + 35))
                    elif value >= 0:
                        level = int(np.clip(value * 255.0, 0, 255))
                        color = (18 + level * 9 // 10, 24 + level // 3, 24)
                    else:
                        level = int(np.clip(-value * 255.0, 0, 255))
                        color = (20, 28 + level // 3, 35 + level * 4 // 5)
                    pygame.draw.rect(self.screen, color,
                                     (origin_x + col * cell_w, origin_y + row * cell_h,
                                      max(1, cell_w - 1), max(1, cell_h - 1)))

    def _draw_fly_panel(self, rect: pygame.Rect, now: float) -> None:
        pygame.draw.rect(self.screen, self.panel, rect, border_radius=16)
        fly_height = max(215, int(rect.height * 0.40))
        self.fly_rect = pygame.Rect(rect.x + 8, rect.y + 8, rect.width - 16, fly_height)
        self.fly.draw(self.screen, self.fly_rect, now, self.session.activity,
                      self.session.last_action, self.fly_yaw, self.fly_pitch, self.paused,
                      self.session.motor_state)
        title = self._text("3D FLY · LIVE", self.text, self.font)
        self.screen.blit(title, (self.fly_rect.x + 12, self.fly_rect.y + 10))
        mode_color = self.cyan if self.session.policy_is_cv else self.amber
        mode = self._text(self.session.policy_label, mode_color, self.small)
        self.screen.blit(mode, (self.fly_rect.x + 12, self.fly_rect.y + 34))
        action = self._text(f"Действие: {self.session.action_name[:28]}", self.text, self.small)
        self.screen.blit(action, (self.fly_rect.x + 12, self.fly_rect.bottom - 28))
        brain_height = max(145, int(rect.height * 0.28))
        brain_rect = pygame.Rect(rect.x + 8, self.fly_rect.bottom + 8, rect.width - 16, brain_height)
        sensory, kenyon, mbon, motor = self.session.brain_activity()
        gpu = self.gpu_brain.snapshot()
        if gpu.active and gpu.activity.size:
            kenyon = gpu.activity
            gpu_label = f"GPU FIELD · {gpu.neuron_states / 1000:.0f}K STATES · {gpu.cycles_per_second:.0f} HZ"
        elif not gpu.available:
            gpu_label = "BRAIN MAP · GPU НЕДОСТУПЕН"
        else:
            gpu_label = "BRAIN MAP · GPU ВЫКЛЮЧЕН"
        self.brain.draw(self.screen, brain_rect, sensory, kenyon, mbon, motor, gpu_label)
        sensor = pygame.Rect(rect.x + 8, brain_rect.bottom + 8, rect.width - 16,
                             rect.bottom - brain_rect.bottom - 16)
        self._draw_sensor(sensor)

    def draw(self) -> None:
        self.screen.fill(self.background)
        now = time.monotonic()
        top, sidebar, game, fly = self._layout()
        pygame.draw.rect(self.screen, self.panel, top, border_radius=14)
        self.screen.blit(self._text("ВИРТУАЛЬНАЯ МУХА", self.text, self.title_font), (top.x + 16, top.y + 10))
        gpu = self.gpu_brain.snapshot()
        if not gpu.available:
            gpu_text = "GPU ERROR · нажми G"
        elif gpu.profile == "off":
            gpu_text = "GPU OFF · нажми G"
        else:
            count = f"{gpu.neuron_states / 1000:.0f}K"
            stats = []
            if gpu.gpu_utilization is not None:
                stats.append(f"{gpu.gpu_utilization}%")
            if gpu.temperature_c is not None:
                stats.append(f"{gpu.temperature_c}°C")
            suffix = " · " + " · ".join(stats) if stats else " · запуск…"
            gpu_text = f"GPU {gpu.profile.upper()} · {count} STATES{suffix}"
        self.gpu_button = pygame.Rect(top.right - 330, top.y + 8, 314, 38)
        self._button(self.gpu_button, gpu_text, gpu.active)
        self._draw_sidebar(sidebar)
        self._draw_game(game)
        self._draw_fly_panel(fly, now)
        note = "G: OFF/ECO/RTX/MAX · GPU-поле получает CV-сигнал; игровая policy остаётся проверяемой отдельно."
        self.screen.blit(self._text(note, self.muted, self.small), (18, self.screen.get_height() - 32))
        pygame.display.flip()

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)

    def _click(self, pos: tuple[int, int]) -> None:
        if self.gpu_button.collidepoint(pos):
            self.gpu_brain.cycle_profile()
            return
        for game, rect in self.game_buttons.items():
            if rect.collidepoint(pos):
                self.session.switch(game)
                self.accumulator = 0.0
                return
        for name, rect in self.control_buttons.items():
            if not rect.collidepoint(pos):
                continue
            if name == "pause":
                self.paused = not self.paused
            elif name == "restart":
                self.session.restart()
            elif name == "minus":
                self.speed_index = max(0, self.speed_index - 1)
            elif name == "plus":
                self.speed_index = min(len(self.speeds) - 1, self.speed_index + 1)
            return

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.windowed_size = (max(900, event.w), max(620, event.h))
                self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.session.restart()
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_g:
                    self.gpu_brain.cycle_profile()
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    self.session.switch(GAME_ORDER[event.key - pygame.K_1])
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_index = max(0, self.speed_index - 1)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed_index = min(len(self.speeds) - 1, self.speed_index + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._click(event.pos)
                if self.fly_rect.collidepoint(event.pos):
                    self.dragging_fly = True
                    self.last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging_fly = False
            elif event.type == pygame.MOUSEMOTION and self.dragging_fly:
                dx = event.pos[0] - self.last_mouse[0]
                dy = event.pos[1] - self.last_mouse[1]
                self.fly_yaw += dx * 0.012
                self.fly_pitch = float(np.clip(self.fly_pitch + dy * 0.008, -0.9, 0.9))
                self.last_mouse = event.pos

    def run(self) -> None:
        while self.running:
            dt = min(0.1, self.clock.tick(60) / 1000.0)
            self._events()
            if not self.paused:
                self.accumulator += dt * self.speeds[self.speed_index]
                interval = 1.0 / self.decision_hz
                steps = 0
                while self.accumulator >= interval and steps < 4:
                    self.session.advance()
                    self.gpu_brain.set_input(self.session.last_observation)
                    self.accumulator -= interval
                    steps += 1
            self.draw()
        self.session.env.close()
        self.gpu_brain.stop()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=GAME_ORDER, default="flight_3d")
    parser.add_argument("--gpu", choices=PROFILE_ORDER, default="rtx",
                        help="CUDA neural-field load profile")
    parser.add_argument("--screenshot", type=Path,
                        help="render one dashboard frame to PNG and exit")
    args = parser.parse_args()
    studio = FlyStudio(args.game, args.gpu)
    if args.screenshot:
        studio.session.advance()
        studio.draw()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(studio.screen, args.screenshot)
        studio.session.env.close()
        studio.gpu_brain.stop()
        pygame.quit()
        return
    studio.run()


if __name__ == "__main__":
    main()
