"""Compact live neural and motor activity map for Fly Studio."""

from __future__ import annotations

import math

import numpy as np
import pygame


def _normalise(values: np.ndarray, size: int) -> np.ndarray:
    source = np.nan_to_num(
        np.asarray(values, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if source.size == 0:
        return np.zeros(size, dtype=np.float32)
    chunks = np.array_split(np.abs(source), size)
    pooled = np.asarray([float(chunk.max()) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    maximum = float(pooled.max())
    return pooled / maximum if maximum > 1e-8 else pooled


class NeuralMapRenderer:
    """Show sensory, Kenyon, MBON and motor nodes with real live values."""

    def __init__(self) -> None:
        self.font: pygame.font.Font | None = None
        self.small: pygame.font.Font | None = None

    def _fonts(self) -> None:
        if self.font is None:
            self.font = pygame.font.SysFont("segoeui", 14, bold=True)
            self.small = pygame.font.SysFont("consolas", 11)

    @staticmethod
    def _node_color(value: float, family: str) -> tuple[int, int, int]:
        value = float(np.clip(value, 0.0, 1.0))
        base = {
            "eye": (50, 206, 224),
            "kc": (245, 180, 70),
            "mbon": (239, 92, 126),
            "motor": (132, 218, 108),
        }[family]
        return tuple(int(24 + (channel - 24) * value) for channel in base)

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        sensory: np.ndarray,
        kenyon: np.ndarray,
        mbon: np.ndarray,
        motor: np.ndarray,
        gpu_label: str = "",
    ) -> None:
        self._fonts()
        pygame.draw.rect(surface, (20, 31, 43), rect, border_radius=12)
        title_text = gpu_label or "BRAIN MAP · LIVE NEURONS"
        title = self.font.render(title_text, True, (136, 154, 169))
        surface.blit(title, (rect.x + 10, rect.y + 7))
        gpu_field_active = gpu_label.startswith("GPU FIELD")

        eye = _normalise(sensory, 10)
        raw_kenyon = np.nan_to_num(
            np.asarray(kenyon, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        kc = _normalise(raw_kenyon, 36)
        out = _normalise(mbon, min(9, max(2, np.asarray(mbon).size)))
        motors = _normalise(motor, 7)
        top = rect.y + 34
        bottom = rect.bottom - 28
        height = max(20, bottom - top)
        eye_x = rect.x + 22
        kc_center = (rect.x + int(rect.width * 0.40), top + height // 2)
        mbon_x = rect.x + int(rect.width * 0.70)
        motor_x = rect.right - 18

        eye_pos = [(eye_x, top + int((index + 0.5) * height / len(eye))) for index in range(len(eye))]
        kc_pos = []
        for index in range(len(kc)):
            lobe = -1 if index < len(kc) // 2 else 1
            local = index % (len(kc) // 2)
            angle = 2 * math.pi * local / (len(kc) // 2)
            x = kc_center[0] + lobe * 18 + int(math.cos(angle) * (18 + 8 * (local % 3) / 2))
            y = kc_center[1] + int(math.sin(angle) * height * 0.37)
            kc_pos.append((x, y))
        mbon_pos = [(mbon_x, top + int((index + 0.5) * height / len(out))) for index in range(len(out))]
        motor_pos = [(motor_x, top + int((index + 0.5) * height / len(motors))) for index in range(len(motors))]

        # Brain outline and sparse, activity-weighted connections.
        lobe_rect = pygame.Rect(kc_center[0] - 47, top + 2, 94, max(20, height - 4))
        pygame.draw.ellipse(surface, (55, 68, 80), lobe_rect, 1)
        for index, point in enumerate(kc_pos):
            source = eye_pos[(index * 7) % len(eye_pos)]
            if kc[index] > 0.08:
                color = (27, int(55 + kc[index] * 75), int(64 + kc[index] * 90))
                pygame.draw.line(surface, color, source, point, 1)
            target = mbon_pos[(index * 5) % len(mbon_pos)]
            if kc[index] > 0.25:
                pygame.draw.line(surface, (84, 63, 44), point, target, 1)
        for index, point in enumerate(mbon_pos):
            target = motor_pos[index % len(motor_pos)]
            pygame.draw.line(surface, (63, 48, 58), point, target, 1)

        for value, point in zip(eye, eye_pos):
            pygame.draw.circle(surface, self._node_color(float(value), "eye"), point, 3)
        for index, (value, point) in enumerate(zip(kc, kc_pos)):
            radius = 2 + int(value > 0.72)
            pygame.draw.circle(surface, self._node_color(float(value), "kc"), point, radius)
            if value > 0.88:
                pygame.draw.circle(surface, (255, 222, 146), point, radius + 2, 1)
        for value, point in zip(out, mbon_pos):
            pygame.draw.circle(surface, self._node_color(float(value), "mbon"), point, 4)
        for value, point in zip(motors, motor_pos):
            pygame.draw.circle(surface, self._node_color(float(value), "motor"), point, 4)

        labels = (("EYE", eye_x - 11), ("GPU" if gpu_field_active else "KC", kc_center[0] - 8),
                  ("MBON", mbon_x - 15), ("MOTOR", motor_x - 27))
        for label, x in labels:
            surface.blit(self.small.render(label, True, (121, 140, 155)), (x, rect.bottom - 20))
        top_ids = np.argsort(np.abs(raw_kenyon))[-3:][::-1] if raw_kenyon.size else np.zeros(3, dtype=int)
        raw_max = float(np.max(np.abs(raw_kenyon))) if raw_kenyon.size else 0.0
        active = ("TOP GPU  " if gpu_field_active else "TOP KC  ") + "  ".join(
            f"{int(index):04d}:{(abs(float(raw_kenyon[index])) / raw_max if raw_max > 1e-8 else 0.0):.2f}"
            for index in top_ids
        )
        surface.blit(self.small.render(active, True, (221, 184, 102)), (rect.x + 10, rect.bottom - 34))
