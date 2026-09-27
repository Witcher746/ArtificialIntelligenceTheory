"""Low-resolution, motion-sensitive vision inspired by a compound eye.

The policy deliberately receives no game objects or coordinates.  A rendered
RGB frame is sampled into 721 luminance facets and 721 temporal-contrast
facets.  The number 721 mirrors the per-eye ommatidia count used by the
``fly-brain`` reference project; this is an engineering analogy, not a
biological simulation.
"""

from __future__ import annotations

import numpy as np


class CompoundEye:
    """Convert arbitrary RGB frames to a fixed-size pixel observation."""

    facets_x = 38
    facets_y = 19
    facets = 721

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        # luminance + temporal contrast + bias
        return self.facets * 2 + 1

    def reset(self) -> None:
        self._previous = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Return a float32 vector computed exclusively from an RGB frame."""
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 RGB frame, got {frame.shape!r}")

        height, width = frame.shape[:2]
        ys = np.linspace(0, height - 1, self.facets_y).round().astype(np.intp)
        xs = np.linspace(0, width - 1, self.facets_x).round().astype(np.intp)
        sampled = frame[np.ix_(ys, xs)].astype(np.float32, copy=False)
        luminance = (
            sampled[..., 0] * 0.2126
            + sampled[..., 1] * 0.7152
            + sampled[..., 2] * 0.0722
        ).reshape(-1)[: self.facets]
        luminance /= 255.0

        # Local contrast is more robust than absolute screen brightness.
        contrast = luminance - float(luminance.mean())
        scale = float(luminance.std()) + 1e-4
        contrast = np.clip(contrast / (3.0 * scale), -1.0, 1.0)

        if self._previous is None:
            motion = np.zeros_like(luminance)
        else:
            motion = np.clip((luminance - self._previous) * 4.0, -1.0, 1.0)
        self._previous = luminance.copy()

        return np.concatenate(
            [contrast.astype(np.float32), motion.astype(np.float32), np.ones(1, np.float32)]
        )


class KnifeCompoundEye(CompoundEye):
    """Compound eye with a high-resolution annular fovea around the log.

    Knife Hit needs angular precision that 721 full-screen facets cannot
    provide.  The extra samples still come only from RGB pixels and are
    analogous to allocating more ommatidia to a behaviorally important area.
    """

    angular_facets = 96

    def __init__(self) -> None:
        super().__init__()
        self._previous_ring: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        return super().output_dim + self.angular_facets * 2 + 8

    def reset(self) -> None:
        super().reset()
        self._previous_ring = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        base = super().encode(frame)
        height, width = frame.shape[:2]
        cx, cy = width * 0.5, height * (250.0 / 760.0)
        scale = min(width / 560.0, height / 760.0)
        radii = np.linspace(94.0 * scale, 188.0 * scale, 28, dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, self.angular_facets, endpoint=False, dtype=np.float32)
        xs = np.rint(cx + np.cos(angles)[:, None] * radii[None, :]).astype(np.intp)
        ys = np.rint(cy + np.sin(angles)[:, None] * radii[None, :]).astype(np.intp)
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
        pixels = frame[ys, xs].astype(np.float32)
        blade_colors = np.array([[214.0, 220.0, 230.0], [150.0, 158.0, 172.0]], dtype=np.float32)
        color_distance = np.sqrt(np.sum((pixels[:, :, None, :] - blade_colors[None, None, :, :]) ** 2, axis=3))
        # Exact renderer colours make this a stable classical-CV blade mask;
        # antialiased edges receive a softer score.
        blade_score = np.clip(1.0 - color_distance.min(axis=2) / 115.0, 0.0, 1.0)
        ring = np.max(blade_score, axis=1).astype(np.float32)
        shift = 0
        if self._previous_ring is None:
            ring_motion = np.zeros_like(ring)
        else:
            ring_motion = np.clip((ring - self._previous_ring) * 4.0, -1.0, 1.0)
            shifts = range(-4, 5)
            correlations = [float(np.dot(ring, np.roll(self._previous_ring, s))) for s in shifts]
            shift = list(shifts)[int(np.argmax(correlations))]
        self._previous_ring = ring.copy()

        bottom = self.angular_facets // 4
        predicted = np.roll(ring, int(round(shift * 2.4)))
        around_bottom = [(bottom + offset) % self.angular_facets for offset in range(-4, 5)]
        close_bottom = [(bottom + offset) % self.angular_facets for offset in range(-2, 3)]
        occupied_now = np.flatnonzero(ring > 0.55)
        occupied_future = np.flatnonzero(predicted > 0.55)
        def clearance(indices: np.ndarray) -> float:
            if not len(indices):
                return 1.0
            distance = np.abs(indices - bottom)
            distance = np.minimum(distance, self.angular_facets - distance)
            return float(np.min(distance) / (self.angular_facets / 2))
        y0 = int(cy + 188.0 * scale)
        y1 = min(height, int(height * 0.75))
        x0, x1 = max(0, int(cx - 10 * scale)), min(width, int(cx + 11 * scale))
        corridor = frame[y0:y1, x0:x1].astype(np.float32)
        if corridor.size:
            distance = np.sqrt(
                np.sum((corridor[:, :, None, :] - blade_colors[None, None, :, :]) ** 2, axis=3)
            )
            flight_signal = float(np.max(np.clip(1.0 - distance.min(axis=2) / 115.0, 0.0, 1.0)))
        else:
            flight_signal = 0.0
        cv_summary = np.array(
            [
                float(np.max(ring[around_bottom])),
                float(np.max(ring[close_bottom])),
                float(np.max(predicted[around_bottom])),
                float(np.max(predicted[close_bottom])),
                clearance(occupied_now),
                clearance(occupied_future),
                shift / 4.0,
                flight_signal,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, ring, ring_motion, cv_summary]).astype(np.float32, copy=False)


def make_eye(game: str) -> CompoundEye:
    return KnifeCompoundEye() if game == "knife_hit" else CompoundEye()
