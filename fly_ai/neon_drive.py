"""Retro-futuristic pseudo-3D driving mini-simulator with an RGB driver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame


CYAN = (39, 222, 244)
MAGENTA = (245, 58, 184)


@dataclass
class TrafficCar:
    distance: float
    lane: float
    speed_mps: float
    color: tuple[int, int, int]
    near_miss_awarded: bool = False


class NeonDriveCVController:
    """Follow neon road edges and avoid traffic using the rendered RGB frame."""

    def __init__(self) -> None:
        self.target_offset = 0.0
        self.confidence = 0.0
        self.danger = 0.0

    def act(self, frame: np.ndarray) -> int:
        pixels = np.asarray(frame, dtype=np.uint8)
        height, width = pixels.shape[:2]
        red = pixels[..., 0].astype(np.int16)
        green = pixels[..., 1].astype(np.int16)
        blue = pixels[..., 2].astype(np.int16)
        y0, y1 = int(height * 0.53), int(height * 0.88)
        band = np.zeros((height, width), dtype=bool)
        band[y0:y1] = True

        left_edge = band & (green > 165) & (blue > 180) & (red < 125)
        right_edge = band & (red > 175) & (blue > 145) & (green < 135)
        _, left_x = np.nonzero(left_edge)
        _, right_x = np.nonzero(right_edge)
        if left_x.size and right_x.size:
            road_center = (float(np.median(left_x)) + float(np.median(right_x))) * 0.5
            self.confidence = min(1.0, (left_x.size + right_x.size) / 1_500.0)
        else:
            road_center = width * 0.5
            self.confidence = 0.0

        traffic_mask = band & (red > 165) & (green > 55) & (green < 175) & (blue < 145)
        traffic_y, traffic_x = np.nonzero(traffic_mask)
        danger_zone = ((traffic_x > width * 0.32) & (traffic_x < width * 0.68)
                       & (traffic_y > height * 0.40) & (traffic_y < height * 0.86))
        danger_count = int(np.count_nonzero(danger_zone))
        self.danger = min(1.0, danger_count / 2_000.0)
        brake = danger_count > 650
        if danger_count > 150:
            obstacle_x = float(traffic_x[danger_zone].mean())
            road_center += -math.copysign(width * 0.22, obstacle_x - width * 0.5)

        offset = road_center - width * 0.5
        self.target_offset = float(np.clip(offset / (width * 0.5), -1.0, 1.0))
        if brake:
            return 4
        if offset < -20:
            return 5
        if offset > 20:
            return 6
        return 3


class NeonDriveEnv:
    game_name = "neon_drive"
    width, height = 1100, 760
    n_actions = 7
    action_names = ("coast", "left", "right", "throttle", "brake", "throttle_left", "throttle_right")

    def __init__(self, *, seed: int = 0, max_steps: int = 4000) -> None:
        pygame.init()
        pygame.font.init()
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.screen = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont("consolas", 23, bold=True)
        self.small = pygame.font.SysFont("consolas", 16)
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    @property
    def has_teacher(self) -> bool:
        return False

    def teacher_action(self) -> int:
        raise NotImplementedError("Neon Drive uses the RGB driver")

    def close(self) -> None:
        pass

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.steps = 0
        self.distance = 0.0
        self.speed_kmh = 72.0
        self.lateral = 0.0
        self.lateral_velocity = 0.0
        self.checkpoints = 0
        self.overtakes = 0
        self.near_misses = 0
        self.weaves = 0
        self.last_lane = 1
        self.last_weave_step = -100
        self.next_checkpoint = 210.0
        self.phase = "DRIVE"
        self.crash_reason = ""
        self.world_time = 0.0
        colors = ((238, 82, 54), (248, 154, 46), (220, 91, 113))
        self.traffic = [
            TrafficCar(float(self.rng.uniform(85, 230)), float(self.rng.choice((-0.62, 0.0, 0.62))),
                       float(self.rng.uniform(18, 37)), colors[index % len(colors)])
            for index in range(7)
        ]
        return self.render(), self.info()

    def _curve(self, distance: float) -> float:
        return math.sin(distance / 78.0) * 0.82 + math.sin(distance / 31.0) * 0.31

    def _road_row(self, depth: float) -> tuple[float, float, float, float]:
        far = 150.0
        progress = float(np.clip(1.0 - depth / far, 0.0, 1.0))
        y = self.height * 0.30 + progress ** 1.68 * self.height * 0.67
        half = 24.0 + progress ** 1.55 * self.width * 0.46
        curve = self._curve(self.distance + depth)
        center = self.width * 0.5 + curve * (24.0 + 105.0 * progress)
        center -= self.lateral * half * 0.74
        return center, y, half, progress

    def _draw_background(self) -> None:
        horizon = int(self.height * 0.30)
        for y in range(0, horizon, 6):
            t = y / max(1, horizon)
            pygame.draw.rect(self.screen, (int(5 + t * 10), int(8 + t * 8), int(25 + t * 20)),
                             (0, y, self.width, 6))
        self.screen.fill((4, 8, 20), (0, horizon, self.width, self.height - horizon))
        for index in range(45):
            x = int((index * 193 + self.seed * 17) % self.width)
            y = int((index * 71 + self.seed * 11) % max(1, horizon - 18))
            brightness = 90 + (index * 31) % 150
            pygame.draw.circle(self.screen, (brightness, brightness, min(255, brightness + 40)), (x, y), 1)
        pygame.draw.circle(self.screen, (126, 75, 182), (self.width - 135, 92), 54)
        pygame.draw.circle(self.screen, (185, 105, 221), (self.width - 150, 80), 27)

    def _draw_city(self) -> None:
        horizon = int(self.height * 0.30)
        segment = int(self.distance // 18)
        for side in (-1, 1):
            for index in range(13):
                key = segment + index
                depth = 18.0 + index * 10.5 - (self.distance % 18.0) * 0.35
                center, y, half, progress = self._road_row(depth)
                width = max(4, int(8 + progress * 34))
                height = max(10, int((28 + ((key * 37) % 70)) * progress))
                x = int(center + side * (half + 35 + ((key * 23) % 55) * progress))
                rect = pygame.Rect(x - width // 2, int(y - height), width, height)
                color = (17 + (key * 7) % 18, 19, 40 + (key * 11) % 35)
                pygame.draw.rect(self.screen, color, rect)
                window_color = CYAN if (key + side) % 2 else MAGENTA
                if width >= 10:
                    for wy in range(rect.y + 5, rect.bottom - 4, max(8, width // 3)):
                        pygame.draw.line(self.screen, window_color, (rect.x + 3, wy), (rect.right - 3, wy), 1)
        pygame.draw.line(self.screen, (76, 42, 109), (0, horizon), (self.width, horizon), 2)

    def _draw_road(self) -> None:
        depths = np.linspace(150.0, 1.8, 52)
        rows = [self._road_row(float(depth)) for depth in depths]
        for index in range(len(rows) - 1):
            c0, y0, h0, p0 = rows[index]
            c1, y1, h1, p1 = rows[index + 1]
            stripe = ((int(self.distance / 6.0) + index) % 2) == 0
            road = (19, 23, 38) if stripe else (23, 27, 44)
            pygame.draw.polygon(self.screen, road,
                                ((c0 - h0, y0), (c0 + h0, y0), (c1 + h1, y1), (c1 - h1, y1)))
            shoulder = 7 + int(p1 * 6)
            pygame.draw.line(self.screen, CYAN, (c0 - h0, y0), (c1 - h1, y1), shoulder)
            pygame.draw.line(self.screen, MAGENTA, (c0 + h0, y0), (c1 + h1, y1), shoulder)
            if (index + int(self.distance / 5.0)) % 5 < 3:
                for lane in (-1 / 3, 1 / 3):
                    pygame.draw.line(self.screen, (167, 175, 196),
                                     (c0 + h0 * lane, y0), (c1 + h1 * lane, y1),
                                     max(1, int(1 + p1 * 3)))

    def _draw_checkpoint(self) -> None:
        depth = self.next_checkpoint - self.distance
        if not 1.0 < depth < 150.0:
            return
        center, y, half, progress = self._road_row(depth)
        arch_h = max(9, int(20 + progress * 220))
        arch_w = max(16, int(half * 1.55))
        thickness = max(2, int(2 + progress * 8))
        pygame.draw.line(self.screen, (117, 246, 125), (center - arch_w, y), (center - arch_w, y - arch_h), thickness)
        pygame.draw.line(self.screen, (117, 246, 125), (center + arch_w, y), (center + arch_w, y - arch_h), thickness)
        pygame.draw.line(self.screen, (117, 246, 125), (center - arch_w, y - arch_h),
                         (center + arch_w, y - arch_h), thickness)

    def _draw_traffic(self) -> None:
        visible = []
        for car in self.traffic:
            depth = car.distance - self.distance
            if 1.2 < depth < 150.0:
                visible.append((depth, car))
        for depth, car in sorted(visible, reverse=True):
            center, y, half, progress = self._road_row(depth)
            x = center + car.lane * half * 0.67
            width = max(4, int(8 + progress * 58))
            height = max(3, int(width * 0.48))
            rect = pygame.Rect(int(x - width / 2), int(y - height), width, height)
            pygame.draw.ellipse(self.screen, (40, 11, 18), rect.inflate(8, 5))
            pygame.draw.rect(self.screen, car.color, rect, border_radius=max(1, width // 7))
            pygame.draw.rect(self.screen, (71, 201, 231),
                             (rect.x + width // 4, rect.y + 2, width // 2, max(1, height // 3)),
                             border_radius=2)
            pygame.draw.circle(self.screen, (255, 214, 92), (rect.x + 3, rect.bottom - 2), max(1, width // 12))
            pygame.draw.circle(self.screen, (255, 214, 92), (rect.right - 3, rect.bottom - 2), max(1, width // 12))

    def _draw_player(self) -> None:
        cx, cy = self.width // 2, int(self.height * 0.87)
        bank = int(np.clip(self.lateral_velocity * 12.0, -18, 18))
        trail = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(trail, (36, 222, 246, 85), ((cx - 43, cy + 8), (cx - 98, cy + 68), (cx - 16, cy + 20)))
        pygame.draw.polygon(trail, (246, 52, 188, 85), ((cx + 43, cy + 8), (cx + 98, cy + 68), (cx + 16, cy + 20)))
        self.screen.blit(trail, (0, 0))
        body = [(cx - 62, cy + 18 + bank), (cx - 39, cy - 23 + bank),
                (cx, cy - 38), (cx + 39, cy - 23 - bank), (cx + 62, cy + 18 - bank),
                (cx + 28, cy + 34), (cx - 28, cy + 34)]
        pygame.draw.polygon(self.screen, (50, 55, 77), body)
        pygame.draw.polygon(self.screen, (130, 231, 241), body, 2)
        pygame.draw.polygon(self.screen, (20, 27, 46),
                            ((cx - 26, cy - 19), (cx, cy - 32), (cx + 26, cy - 19), (cx + 18, cy + 3), (cx - 18, cy + 3)))
        pygame.draw.circle(self.screen, CYAN, (cx - 36, cy + 16), 6)
        pygame.draw.circle(self.screen, MAGENTA, (cx + 36, cy + 16), 6)

    def _draw_hud(self) -> None:
        panel = pygame.Surface((262, 122), pygame.SRCALPHA)
        panel.fill((4, 9, 18, 190))
        panel.blit(self.font.render(f"{self.speed_kmh:03.0f} KM/H", True, CYAN), (12, 8))
        panel.blit(self.small.render(f"DIST  {self.distance:07.1f} M", True, (224, 231, 242)), (12, 46))
        panel.blit(self.small.render(f"PORTALS  {self.checkpoints}", True, (118, 238, 128)), (12, 70))
        panel.blit(self.small.render("AI  RGB NEON DRIVER", True, MAGENTA), (12, 94))
        self.screen.blit(panel, (18, 18))
        if self.phase == "CRASH":
            shade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            shade.fill((78, 5, 28, 135))
            self.screen.blit(shade, (0, 0))
            label = self.font.render("HOVER CAR LOST", True, (255, 216, 230))
            self.screen.blit(label, label.get_rect(center=(self.width // 2, self.height // 2)))

    def render(self) -> np.ndarray:
        self._draw_background()
        self._draw_city()
        self._draw_road()
        self._draw_checkpoint()
        self._draw_traffic()
        self._draw_player()
        self._draw_hud()
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside Neon Drive action space")
        reward = 0.0
        steer = -1.0 if action in (1, 5) else (1.0 if action in (2, 6) else 0.0)
        throttle = action in (3, 5, 6)
        braking = action == 4
        for _ in range(4):
            dt = 1.0 / 60.0
            self.world_time += dt
            if throttle:
                self.speed_kmh += 32.0 * dt
            elif braking:
                self.speed_kmh -= 92.0 * dt
            else:
                self.speed_kmh -= 10.0 * dt
            self.speed_kmh = float(np.clip(self.speed_kmh, 28.0, 190.0))
            self.lateral_velocity += steer * 2.35 * dt
            self.lateral_velocity *= math.exp(-4.0 * dt)
            self.lateral += self.lateral_velocity * dt * (0.50 + self.speed_kmh / 190.0)
            self.distance += self.speed_kmh / 3.6 * dt
            for car in self.traffic:
                car.distance += car.speed_mps * dt
                if car.distance < self.distance - 9.0:
                    self.overtakes += 1
                    reward += 1.2
                    car.distance = self.distance + float(self.rng.uniform(110, 245))
                    car.lane = float(self.rng.choice((-0.62, 0.0, 0.62)))
                    car.speed_mps = float(self.rng.uniform(17, 39))
                    car.near_miss_awarded = False
            if self.distance >= self.next_checkpoint:
                self.checkpoints += 1
                self.next_checkpoint += 250.0
                reward += 10.0
            for car in self.traffic:
                gap = car.distance - self.distance
                lateral_gap = abs(self.lateral - car.lane)
                if -0.7 < gap < 3.0 and lateral_gap < 0.22:
                    self.phase = "CRASH"
                    self.crash_reason = "traffic"
                    reward -= 10.0
                    break
                if (-0.7 < gap < 3.0 and 0.22 <= lateral_gap < 0.52
                        and not car.near_miss_awarded):
                    car.near_miss_awarded = True
                    self.near_misses += 1
                    reward += 0.6
            if abs(self.lateral) > 1.65:
                self.phase = "CRASH"
                self.crash_reason = "offroad"
                reward -= 10.0
            if self.phase == "CRASH":
                break
            reward += self.speed_kmh / 190.0 * 0.004

        current_lane = int(np.argmin(np.abs(np.asarray((-0.62, 0.0, 0.62)) - self.lateral)))
        if (current_lane != self.last_lane and self.speed_kmh >= 90.0
                and self.steps - self.last_weave_step >= 12):
            self.weaves += 1
            self.last_weave_step = self.steps
            reward += 0.15
        self.last_lane = current_lane
        self.steps += 1
        terminated = self.phase == "CRASH"
        truncated = self.steps >= self.max_steps
        return self.render(), float(reward), terminated, truncated, self.info()

    def info(self) -> dict:
        return {
            "game": self.game_name,
            "step": int(self.steps),
            "score": float(self.checkpoints * 100 + self.distance / 10.0),
            "checkpoints": int(self.checkpoints),
            "overtakes": int(self.overtakes),
            "near_misses": int(self.near_misses),
            "weaves": int(self.weaves),
            "distance": float(self.distance),
            "speed": float(self.speed_kmh),
            "phase": self.phase,
            "crash_reason": self.crash_reason,
        }
