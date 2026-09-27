"""Pixel-only DQN model and frame encoder for Neon Drive."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


INPUT_HEIGHT = 64
INPUT_WIDTH = 96
INPUT_CHANNELS = 2


class NeonFrameEncoder:
    """Turn an RGB frame into luminance + motion uint8 channels."""

    def __init__(self, height: int = INPUT_HEIGHT, width: int = INPUT_WIDTH) -> None:
        self.height = int(height)
        self.width = int(width)
        self.previous: np.ndarray | None = None
        self._indices: tuple[np.ndarray, np.ndarray] | None = None
        self._source_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        self.previous = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        pixels = np.asarray(frame, dtype=np.uint8)
        source_shape = pixels.shape[:2]
        if self._indices is None or self._source_shape != source_shape:
            rows = np.linspace(0, source_shape[0] - 1, self.height, dtype=np.int32)
            cols = np.linspace(0, source_shape[1] - 1, self.width, dtype=np.int32)
            self._indices = rows, cols
            self._source_shape = source_shape
        rows, cols = self._indices
        small = pixels[rows[:, None], cols[None, :]]
        gray = ((small[..., 0].astype(np.uint16) * 77
                 + small[..., 1].astype(np.uint16) * 150
                 + small[..., 2].astype(np.uint16) * 29) >> 8).astype(np.uint8)
        if self.previous is None:
            motion = np.full_like(gray, 128)
        else:
            difference = gray.astype(np.int16) - self.previous.astype(np.int16)
            motion = np.clip(difference + 128, 0, 255).astype(np.uint8)
        self.previous = gray.copy()
        return np.stack((gray, motion), axis=0)


class NeonDQN(nn.Module):
    """Compact dueling convolutional Q-network."""

    def __init__(self, n_actions: int = 7) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.encoder = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened = self.encoder(torch.zeros(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(flattened, 512), nn.ReLU(inplace=True))
        self.value = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.advantage = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, self.n_actions)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.float().div(255.0)
        hidden = self.trunk(self.encoder(values))
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class NeonDQNPolicy:
    """Inference adapter used by Fly Studio after a trained checkpoint exists."""

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = NeonDQN(int(payload.get("n_actions", 7))).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.frame_encoder = NeonFrameEncoder()
        self.confidence = 0.0

    def reset(self) -> None:
        self.frame_encoder.reset()

    @torch.inference_mode()
    def act(self, frame: np.ndarray) -> int:
        state = self.frame_encoder.encode(frame)
        tensor = torch.from_numpy(state[None]).to(self.device, non_blocking=True)
        q_values = self.model(tensor)[0]
        probabilities = torch.softmax(q_values, dim=0)
        self.confidence = float(probabilities.max().item())
        return int(torch.argmax(q_values).item())
