"""Small dependency-free software 3D renderer for the virtual fly."""

from __future__ import annotations

import math

import numpy as np
import pygame


def _rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float32)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)), dtype=np.float32)
    rz = np.array(((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    return rx @ ry @ rz


def _sphere_mesh(rings: int = 8, segments: int = 14) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    vertices = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append((math.cos(phi), math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta)))
    faces = []
    for ring in range(rings):
        for segment in range(segments):
            nxt = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + nxt
            c = (ring + 1) * segments + nxt
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    return np.asarray(vertices, dtype=np.float32), faces


def _shade(color: tuple[int, int, int], value: float) -> tuple[int, int, int]:
    value = min(1.25, max(0.22, value))
    return tuple(min(255, max(0, int(channel * value))) for channel in color)


class FlyRenderer3D:
    """Render a rotating mesh fly and animate it from agent activity."""

    def __init__(self) -> None:
        self.vertices, self.faces = _sphere_mesh()

    @staticmethod
    def _project(points: np.ndarray, center: tuple[float, float], focal: float) -> np.ndarray:
        depth = np.maximum(2.0, 7.5 - points[:, 2])
        return np.column_stack((center[0] + points[:, 0] * focal / depth,
                                center[1] - points[:, 1] * focal / depth))

    def _ellipsoid(
        self,
        surface: pygame.Surface,
        center_3d: tuple[float, float, float],
        scale_3d: tuple[float, float, float],
        color: tuple[int, int, int],
        matrix: np.ndarray,
        center_2d: tuple[float, float],
        focal: float,
    ) -> None:
        world = self.vertices * np.asarray(scale_3d, dtype=np.float32) + np.asarray(center_3d, dtype=np.float32)
        rotated = world @ matrix.T
        screen = self._project(rotated, center_2d, focal)
        light = np.asarray((-0.35, 0.65, 0.70), dtype=np.float32)
        ordered = sorted(self.faces, key=lambda face: float(rotated[list(face), 2].mean()))
        for face in ordered:
            indices = list(face)
            points = rotated[indices]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            length = float(np.linalg.norm(normal))
            if length < 1e-8:
                continue
            normal /= length
            intensity = 0.42 + 0.58 * abs(float(np.dot(normal, light)))
            polygon = [(int(screen[index, 0]), int(screen[index, 1])) for index in indices]
            pygame.draw.polygon(surface, _shade(color, intensity), polygon)
            pygame.draw.polygon(surface, _shade(color, intensity * 0.58), polygon, 1)

    def _polyline(
        self,
        surface: pygame.Surface,
        points: list[tuple[float, float, float]],
        matrix: np.ndarray,
        center: tuple[float, float],
        focal: float,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        transformed = np.asarray(points, dtype=np.float32) @ matrix.T
        projected = self._project(transformed, center, focal)
        pygame.draw.lines(surface, color, False, projected.astype(int).tolist(), width)

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        time_s: float,
        activity: float,
        action: int,
        yaw_offset: float = 0.0,
        pitch_offset: float = 0.0,
        paused: bool = False,
        motor_state: np.ndarray | None = None,
    ) -> None:
        pygame.draw.rect(surface, (10, 16, 24), rect, border_radius=18)
        inner = rect.inflate(-14, -14)
        pygame.draw.rect(surface, (15, 25, 37), inner, border_radius=14)
        old_clip = surface.get_clip()
        surface.set_clip(inner)

        center = (float(inner.centerx), float(inner.centery + inner.height * 0.07))
        focal = float(min(inner.width, inner.height) * 1.28)
        yaw = time_s * 0.28 + yaw_offset
        pitch = -0.12 + pitch_offset
        motor = np.zeros(7, dtype=np.float32) if motor_state is None else np.resize(
            np.asarray(motor_state, dtype=np.float32), 7
        )
        matrix = _rotation(yaw, pitch, float(np.clip(motor[5], -1.0, 1.0)) * 0.22)
        activity = float(np.clip(activity, 0.0, 1.0))

        # Perspective floor grid makes the software-rendered volume legible.
        grid_color = (25, 48, 63)
        for z in np.linspace(-3.0, 3.0, 7):
            self._polyline(surface, [(-3.2, -1.35, float(z)), (3.2, -1.35, float(z))],
                           matrix, center, focal, grid_color, 1)
        for x in np.linspace(-3.0, 3.0, 7):
            self._polyline(surface, [(float(x), -1.35, -3.0), (float(x), -1.35, 3.0)],
                           matrix, center, focal, grid_color, 1)

        wing_overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for wing_index, side in enumerate((-1.0, 1.0)):
            drive = float(np.clip(motor[wing_index], 0.0, 1.0))
            flap = 0.0 if paused else math.sin(time_s * (18.0 + drive * 22.0)) * (0.28 + drive * 0.62)
            wing = np.asarray([
                (-0.15, 0.35, side * 0.30),
                (-1.75, 0.55 + flap * side, side * 1.85),
                (-2.30, 0.08 + flap * side, side * 1.35),
                (-0.45, 0.05, side * 0.28),
            ], dtype=np.float32) @ matrix.T
            projected = self._project(wing, center, focal).astype(int).tolist()
            pygame.draw.polygon(wing_overlay, (132, 218, 242, 82), projected)
            pygame.draw.lines(wing_overlay, (190, 239, 250, 175), True, projected, 2)
            root, tip_a, tip_b, _ = projected
            pygame.draw.line(wing_overlay, (135, 210, 232, 100), root, tip_a, 1)
            pygame.draw.line(wing_overlay, (135, 210, 232, 100), root, tip_b, 1)
        surface.blit(wing_overlay, (0, 0))

        # Six articulated legs.
        for side_index, side in enumerate((-1.0, 1.0)):
            for leg_index, (x, reach) in enumerate(((0.45, 1.15), (-0.10, 1.45), (-0.65, 1.75))):
                drive = float(np.clip(motor[2 + leg_index], 0.0, 1.0))
                stride = 0.0 if paused else math.sin(time_s * (5.0 + drive * 5.0)
                                                       + leg_index * 1.8 + side_index * math.pi) * drive
                self._polyline(surface,
                               [(x, -0.28, side * 0.38),
                                (x - 0.10 + stride * 0.12, -1.00 + abs(stride) * 0.16, side * 0.95),
                                (x - 0.48 + stride * 0.38, -1.34 + abs(stride) * 0.23, side * reach)],
                               matrix, center, focal, (72, 53, 35), 3)

        # Abdomen, thorax and head.
        self._ellipsoid(surface, (-1.05, 0.0, 0.0), (1.42, 0.62, 0.62),
                        (157, 105, 36), matrix, center, focal)
        self._ellipsoid(surface, (0.15, 0.04, 0.0), (0.86, 0.74, 0.73),
                        (88, 64, 37), matrix, center, focal)
        self._ellipsoid(surface, (1.02, -0.02, 0.0), (0.65, 0.59, 0.60),
                        (73, 54, 35), matrix, center, focal)

        # Compound eyes have separate faceted meshes.
        eye_glow = int(40 + 115 * activity)
        for side in (-1.0, 1.0):
            self._ellipsoid(surface, (1.31, 0.03, side * 0.43), (0.40, 0.43, 0.24),
                            (155 + eye_glow // 3, 34 + eye_glow // 8, 30), matrix, center, focal)

        # Abdomen bands and antennae.
        for x in (-1.65, -1.15, -0.65):
            ring = [(x, math.cos(a) * 0.64, math.sin(a) * 0.64) for a in np.linspace(0, 2 * math.pi, 22)]
            transformed = np.asarray(ring, dtype=np.float32) @ matrix.T
            pygame.draw.lines(surface, (59, 43, 27), True,
                              self._project(transformed, center, focal).astype(int).tolist(), 3)
        for side in (-1.0, 1.0):
            self._polyline(surface, [(1.38, 0.25, side * 0.23), (1.88, 0.65, side * 0.42),
                                     (2.25, 0.72, side * 0.55)],
                           matrix, center, focal, (106, 81, 50), 2)

        # Pulsing mushroom-body / MBON nodes inside the thorax.
        action_colors = ((77, 208, 225), (250, 191, 75), (247, 118, 142), (145, 220, 103))
        pulse_color = action_colors[action % len(action_colors)]
        neural_points = np.asarray([
            (0.72, 0.18, -0.15), (0.45, 0.28, 0.18), (0.12, 0.18, -0.24),
            (-0.05, 0.05, 0.22), (0.25, -0.10, -0.20), (0.55, -0.08, 0.17),
        ], dtype=np.float32) @ matrix.T
        neural_screen = self._project(neural_points, center, focal).astype(int)
        for index, point in enumerate(neural_screen):
            phase = 0.5 + 0.5 * math.sin(time_s * 8.0 - index * 0.8)
            radius = max(2, int(2 + activity * 5 * phase))
            pygame.draw.circle(surface, pulse_color, point.tolist(), radius)

        surface.set_clip(old_clip)
