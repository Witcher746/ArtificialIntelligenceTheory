"""Endless pseudo-3D flight course and a pixel-only classical-CV autopilot."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pygame


RING_COLOR = (62, 220, 239)
RING_GLOW = (112, 244, 250)
OBSTACLE_COLOR = (229, 77, 57)


@dataclass
class Ring:
    x: float
    y: float
    z: float
    radius: float = 2.45
    passed: bool = False


@dataclass
class Obstacle:
    x: float
    y: float
    z: float
    radius: float
    spin: float


class FlightCVController:
    """Steer toward cyan hoops and away from red obstacles using RGB only."""

    def __init__(self) -> None:
        self.target = (0.0, 0.0)
        self.confidence = 0.0
        self.danger = 0.0

    def act(self, frame: np.ndarray) -> int:
        pixels = np.asarray(frame, dtype=np.uint8)
        height, width = pixels.shape[:2]
        red = pixels[..., 0].astype(np.int16)
        green = pixels[..., 1].astype(np.int16)
        blue = pixels[..., 2].astype(np.int16)

        ring_mask = (green > 160) & (blue > 175) & (red < 145) & ((blue - red) > 55)
        ys, xs = np.nonzero(ring_mask)
        if len(xs) >= 8:
            # The nearest hoop contributes the most pixels, so a regular centroid
            # naturally prioritises it without reading simulator coordinates.
            target_x = float(xs.mean())
            target_y = float(ys.mean())
            self.confidence = min(1.0, len(xs) / 550.0)
        else:
            target_x, target_y = width * 0.5, height * 0.48
            self.confidence = 0.0

        danger_mask = (red > 155) & (red > green * 1.45) & (red > blue * 1.35)
        danger_y, danger_x = np.nonzero(danger_mask)
        central = ((danger_x > width * 0.32) & (danger_x < width * 0.68)
                   & (danger_y > height * 0.25) & (danger_y < height * 0.74))
        central_count = int(np.count_nonzero(central))
        # Filled obstacle silhouettes grow rapidly as they approach. Ignore
        # small distant red objects that sit behind the current cyan hoop.
        self.danger = min(1.0, central_count / 15_000.0)
        if central_count > 6_000:
            obstacle_x = float(danger_x[central].mean())
            obstacle_y = float(danger_y[central].mean())
            target_x += -math.copysign(width * 0.26, obstacle_x - width * 0.5)
            target_y += -math.copysign(height * 0.20, obstacle_y - height * 0.48)

        dx = target_x - width * 0.5
        dy = target_y - height * 0.48
        self.target = (float(np.clip(dx / (width * 0.5), -1.0, 1.0)),
                       float(np.clip(dy / (height * 0.5), -1.0, 1.0)))
        horizontal = -1 if dx < -12 else (1 if dx > 12 else 0)
        # Screen y grows downward, whereas world z grows upward.
        vertical = 1 if dy < -10 else (-1 if dy > 10 else 0)
        action_by_vector = {
            (0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, 1): 3, (0, -1): 4,
            (-1, 1): 5, (1, 1): 6, (-1, -1): 7, (1, -1): 8,
        }
        return action_by_vector[(horizontal, vertical)]


class Flight3DEnv:
    """Gym-like endless flight environment rendered to an RGB Pygame surface."""

    game_name = "flight_3d"
    width, height = 1100, 760
    n_actions = 9
    action_names = (
        "hold", "left", "right", "up", "down",
        "up_left", "up_right", "down_left", "down_right",
    )
    action_vectors = np.asarray(
        ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1),
         (-1, 1), (1, 1), (-1, -1), (1, -1)),
        dtype=np.float32,
    )

    def __init__(self, *, seed: int = 0, max_steps: int = 4000) -> None:
        pygame.init()
        pygame.font.init()
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.screen = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont("segoeui", 22, bold=True)
        self.small = pygame.font.SysFont("segoeui", 16)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.reset(seed)

    @property
    def has_teacher(self) -> bool:
        return False

    def teacher_action(self) -> int:
        raise NotImplementedError("Flight uses the RGB controller")

    def close(self) -> None:
        pass

    def _generate_course(self) -> None:
        while self.next_ring_x < self.position[0] + 250.0:
            center_y = float(self.rng.uniform(-5.4, 5.4))
            center_z = float(self.rng.uniform(2.2, 8.0))
            self.rings.append(Ring(self.next_ring_x, center_y, center_z))
            gap_start = self.next_ring_x + 8.0
            for _ in range(int(self.rng.integers(1, 3))):
                obstacle_x = float(self.rng.uniform(gap_start, self.next_ring_x + 21.0))
                obstacle_y = float(self.rng.uniform(-7.2, 7.2))
                obstacle_z = float(self.rng.uniform(1.4, 8.5))
                # Keep a navigable corridor around the next hoop.
                if math.hypot(obstacle_y - center_y, obstacle_z - center_z) < 3.8:
                    obstacle_y = float(np.clip(obstacle_y + math.copysign(4.2, obstacle_y or 1.0), -8.2, 8.2))
                self.obstacles.append(Obstacle(obstacle_x, obstacle_y, obstacle_z,
                                               float(self.rng.uniform(0.65, 1.15)),
                                               float(self.rng.uniform(-1.0, 1.0))))
            self.next_ring_x += float(self.rng.uniform(25.0, 32.0))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed = int(seed)
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.steps = 0
        self.position = np.asarray((0.0, 0.0, 4.0), dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.speed = 10.5
        self.score = 0
        self.missed = 0
        self.phase = "FLY"
        self.rings: list[Ring] = []
        self.obstacles: list[Obstacle] = []
        self.next_ring_x = 18.0
        self.world_time = 0.0
        self.last_action = 0
        self._generate_course()
        return self.render(), self.info()

    def _project(self, point: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float] | None:
        camera_x = float(self.position[0] - 8.0)
        depth = float(point[0] - camera_x)
        if depth <= 0.8:
            return None
        focal = self.width * 0.78
        x = self.width * 0.5 + (float(point[1]) - float(self.position[1])) * focal / depth
        y = self.height * 0.48 - (float(point[2]) - float(self.position[2])) * focal / depth
        return x, y, depth

    def _draw_world(self) -> None:
        # Sky gradient.
        for y in range(0, self.height, 8):
            t = y / self.height
            color = (int(10 + 20 * t), int(25 + 30 * t), int(45 + 32 * t))
            pygame.draw.rect(self.screen, color, (0, y, self.width, 8))

        grid_color = (44, 76, 81)
        camera_x = float(self.position[0] - 8.0)
        for lateral in np.linspace(-12.0, 12.0, 13):
            points = []
            for forward in np.linspace(camera_x + 1.0, camera_x + 145.0, 34):
                projected = self._project((forward, lateral, 0.0))
                if projected is not None:
                    points.append((projected[0], projected[1]))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, grid_color, False, points, 1)
        start = math.floor(camera_x / 8.0) * 8.0
        for forward in np.arange(start, camera_x + 150.0, 8.0):
            left = self._project((forward, -12.0, 0.0))
            right = self._project((forward, 12.0, 0.0))
            if left and right:
                pygame.draw.line(self.screen, grid_color, left[:2], right[:2], 1)

        items: list[tuple[float, str, object]] = []
        target_ring = min((ring for ring in self.rings if not ring.passed),
                          key=lambda ring: ring.x, default=None)
        for ring in self.rings:
            projected = self._project((ring.x, ring.y, ring.z))
            if projected and projected[2] < 155.0:
                items.append((projected[2], "ring", ring))
        for obstacle in self.obstacles:
            projected = self._project((obstacle.x, obstacle.y, obstacle.z))
            if projected and projected[2] < 155.0:
                items.append((projected[2], "obstacle", obstacle))

        for _, kind, item in sorted(items, reverse=True, key=lambda row: row[0]):
            if kind == "ring":
                ring = item
                points = []
                for angle in np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False):
                    point = (ring.x, ring.y + math.cos(angle) * ring.radius,
                             ring.z + math.sin(angle) * ring.radius)
                    projected = self._project(point)
                    if projected:
                        points.append((projected[0], projected[1]))
                if len(points) >= 3:
                    center = self._project((ring.x, ring.y, ring.z))
                    width = max(2, min(10, int(70.0 / max(center[2], 1.0)))) if center else 2
                    is_target = ring is target_ring
                    pygame.draw.lines(self.screen, (24, 89, 99), True, points, width + 4)
                    pygame.draw.lines(self.screen, RING_COLOR if is_target else (35, 112, 132),
                                      True, points, width)
                    if is_target and width >= 4:
                        pygame.draw.lines(self.screen, RING_GLOW, True, points, 2)
            else:
                obstacle = item
                projected = self._project((obstacle.x, obstacle.y, obstacle.z))
                if not projected:
                    continue
                x, y, depth = projected
                radius = max(2, int(self.width * 0.78 * obstacle.radius / depth))
                if radius > 1:
                    pygame.draw.circle(self.screen, (80, 24, 24), (int(x), int(y)), radius + 5)
                    pygame.draw.circle(self.screen, OBSTACLE_COLOR, (int(x), int(y)), radius)
                    highlight = (int(x - radius * 0.30), int(y - radius * 0.28))
                    pygame.draw.circle(self.screen, (255, 145, 91), highlight, max(2, radius // 4))

        # Player fly/drone in a third-person chase view.
        cx, cy = int(self.width * 0.5), int(self.height * 0.48)
        bank = float(np.clip(self.velocity[0] * 2.2, -18.0, 18.0))
        pygame.draw.ellipse(self.screen, (34, 23, 16), (cx - 25, cy - 10, 50, 20))
        pygame.draw.ellipse(self.screen, (169, 113, 39), (cx - 18, cy - 8, 36, 16))
        wing = int(20 + abs(math.sin(self.world_time * 22.0)) * 13)
        pygame.draw.polygon(self.screen, (153, 225, 236),
                            ((cx - 4, cy), (cx - 42, cy - wing + int(bank)), (cx - 12, cy + 2)))
        pygame.draw.polygon(self.screen, (153, 225, 236),
                            ((cx + 4, cy), (cx + 42, cy - wing - int(bank)), (cx + 12, cy + 2)))
        pygame.draw.circle(self.screen, (242, 182, 54), (cx + 17, cy), 7)
        pygame.draw.line(self.screen, (238, 238, 222), (cx - 7, cy + 14), (cx - 16, cy + 27), 2)
        pygame.draw.line(self.screen, (238, 238, 222), (cx + 7, cy + 14), (cx + 16, cy + 27), 2)

        hud = pygame.Surface((230, 106), pygame.SRCALPHA)
        hud.fill((5, 12, 18, 175))
        hud.blit(self.font.render(f"RINGS  {self.score}", True, (225, 238, 242)), (12, 8))
        hud.blit(self.small.render(f"SPEED  {self.speed:4.1f} m/s", True, (116, 226, 237)), (12, 42))
        hud.blit(self.small.render(f"ALT    {self.position[2]:4.1f} m", True, (116, 226, 237)), (12, 66))
        hud.blit(self.small.render(f"MISSED {self.missed}", True, (246, 177, 75)), (12, 88))
        self.screen.blit(hud, (18, 18))

        if self.phase == "CRASH":
            shade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            shade.fill((85, 4, 7, 115))
            self.screen.blit(shade, (0, 0))
            label = self.font.render("COLLISION", True, (255, 214, 201))
            self.screen.blit(label, label.get_rect(center=(self.width // 2, self.height // 2 - 60)))

    def render(self) -> np.ndarray:
        self._draw_world()
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside flight action space")
        self.last_action = action
        reward = 0.0
        vector = self.action_vectors[action]
        for _ in range(4):
            dt = 1.0 / 60.0
            self.world_time += dt
            self.velocity += vector * np.asarray((8.5, 7.0), dtype=np.float32) * dt
            self.velocity *= math.exp(-2.7 * dt)
            self.velocity = np.clip(self.velocity, (-4.5, -3.8), (4.5, 3.8))
            self.position[0] += self.speed * dt
            self.position[1] += self.velocity[0] * dt
            self.position[2] += self.velocity[1] * dt
            self.position[2] = float(np.clip(self.position[2], 0.7, 10.0))
            self.speed = min(14.0, 10.5 + self.score * 0.12)

            for ring in self.rings:
                if not ring.passed and self.position[0] >= ring.x:
                    ring.passed = True
                    distance = math.hypot(float(self.position[1]) - ring.y,
                                          float(self.position[2]) - ring.z)
                    if distance <= ring.radius - 0.42:
                        self.score += 1
                        reward += 5.0
                    else:
                        self.missed += 1
                        reward -= 0.35

            for obstacle in self.obstacles:
                dx = float(self.position[0]) - obstacle.x
                if abs(dx) > obstacle.radius + 0.55:
                    continue
                distance = math.sqrt(dx * dx
                                     + (float(self.position[1]) - obstacle.y) ** 2
                                     + (float(self.position[2]) - obstacle.z) ** 2)
                if distance < obstacle.radius + 0.48:
                    self.phase = "CRASH"
                    reward -= 10.0
                    break
            if self.phase == "CRASH":
                break
            if abs(float(self.position[1])) > 10.5:
                self.phase = "CRASH"
                reward -= 10.0
                break
            reward += 0.002

        self.steps += 1
        self.rings = [ring for ring in self.rings if ring.x > self.position[0] - 12.0]
        self.obstacles = [item for item in self.obstacles if item.x > self.position[0] - 12.0]
        self._generate_course()
        terminated = self.phase == "CRASH"
        truncated = self.steps >= self.max_steps
        return self.render(), float(reward), terminated, truncated, self.info()

    def info(self) -> dict:
        return {
            "game": self.game_name,
            "step": int(self.steps),
            "score": int(self.score),
            "rings": int(self.score),
            "missed": int(self.missed),
            "speed": float(self.speed),
            "altitude": float(self.position[2]),
            "phase": self.phase,
        }
