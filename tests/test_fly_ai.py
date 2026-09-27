import os
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from fly_ai.brain import MushroomBodyAgent
from fly_ai.envs import ENV_REGISTRY, make_env
from fly_ai.fly3d import FlyRenderer3D
from fly_ai.flight3d import Flight3DEnv, FlightCVController
from fly_ai.gpu_brain import GpuBrain, PROFILES
from fly_ai.neon_drive import NeonDriveCVController, NeonDriveEnv
from fly_ai.neon_dqn import NeonDQN, NeonFrameEncoder
from fly_ai.neural_viz import _normalise
from fly_ai.studio import GAME_ORDER, LiveSession, facet_grid
from fly_ai.vision import CompoundEye


class FlyAiTests(unittest.TestCase):
    def test_compound_eye_contract(self):
        eye = CompoundEye()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        first = eye.encode(frame)
        frame[20:50, 30:70] = 255
        second = eye.encode(frame)
        self.assertEqual(first.shape, (1443,))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(second).all())
        self.assertGreater(float(np.abs(second[721:1442]).sum()), 0.0)

    def test_agent_roundtrip(self):
        agent = MushroomBodyAgent(17, 3, n_kenyon=32, fan_in=5, seed=7)
        observation = np.linspace(-1, 1, 17, dtype=np.float32)
        features = agent.features(observation)
        agent.imitate(features, 2)
        before = agent.q_values(features)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "agent.npz"
            agent.save(path, {"game": "test"})
            restored, metadata = MushroomBodyAgent.load(path)
        np.testing.assert_allclose(before, restored.q_values(restored.features(observation)))
        self.assertEqual(metadata["game"], "test")

    def test_all_games_have_headless_pixel_step(self):
        for name in ENV_REGISTRY:
            with self.subTest(game=name):
                env = make_env(name, seed=123, max_steps=2)
                frame, info = env.reset()
                self.assertEqual(frame.shape, (env.height, env.width, 3))
                self.assertEqual(frame.dtype, np.uint8)
                next_frame, reward, terminated, truncated, next_info = env.step(0)
                self.assertEqual(next_frame.shape, frame.shape)
                self.assertTrue(np.isfinite(reward))
                self.assertIn("game", info)
                self.assertIn("step", next_info)
                self.assertFalse(terminated)
                self.assertFalse(truncated)

    def test_seeded_rollout_is_repeatable(self):
        env = make_env("race", seed=321, max_steps=50)
        hashes = []
        for _ in range(2):
            frame, _ = env.reset(seed=321)
            digest = hashlib.sha256(frame.tobytes())
            for action in [2, 2, 3, 3, 2, 1, 2] * 4:
                frame, _, terminated, truncated, _ = env.step(action)
                digest.update(frame.tobytes())
                if terminated or truncated:
                    break
            hashes.append(digest.hexdigest())
        self.assertEqual(hashes[0], hashes[1])

    def test_splinter_level_transition_keeps_completion_bonus(self):
        env = make_env("splinter", seed=123, max_steps=2)
        env.reset()
        env.game.splinters = [SimpleNamespace(removed=True) for _ in range(4)]
        old_level = env.game.level

        def advance_level(_dt):
            env.game.level = old_level + 1
            env.game.splinters = [SimpleNamespace(removed=False) for _ in range(5)]

        env.game.update = advance_level
        env.render = lambda: np.zeros((env.height, env.width, 3), dtype=np.uint8)
        _, reward, _, _, _ = env.step(0)
        self.assertGreater(reward, 14.0)

    def test_headless_path_does_not_load_camera_stack(self):
        self.assertNotIn("mediapipe", sys.modules)
        self.assertNotIn("cv2", sys.modules)

    def test_software_3d_fly_renders(self):
        import pygame

        surface = pygame.Surface((480, 360))
        FlyRenderer3D().draw(surface, surface.get_rect(), 1.0, 0.7, 2)
        pixels = pygame.surfarray.array3d(surface)
        self.assertGreater(float(pixels.std()), 10.0)
        self.assertGreater(int(np.count_nonzero(pixels)), 20_000)

    def test_721_facets_fit_visual_grid(self):
        values = np.arange(721, dtype=np.float32)
        grid = facet_grid(values)
        self.assertEqual(grid.shape, (19, 38))
        self.assertEqual(float(grid.reshape(-1)[720]), 720.0)
        self.assertEqual(float(grid.reshape(-1)[721]), 0.0)

    def test_gpu_profiles_and_switching_without_starting_cuda(self):
        brain = GpuBrain("off", autostart=False)
        self.assertEqual(brain.snapshot().neuron_states, 0)
        self.assertEqual(brain.cycle_profile(), "eco")
        self.assertEqual(brain.snapshot().neuron_states, PROFILES["eco"].neuron_states)
        brain.set_profile("rtx")
        self.assertEqual(brain.snapshot().neuron_states, 331_776)
        brain.set_profile("max")
        self.assertEqual(brain.snapshot().neuron_states, 512_000)

    def test_studio_switches_games_and_restarts_forever(self):
        session = LiveSession("race", seed=90000)
        self.assertIsNotNone(session.agent)
        for game in GAME_ORDER:
            with self.subTest(game=game):
                session.switch(game)
                session.advance(now=1.0)
                self.assertEqual(session.frame.shape, (session.env.height, session.env.width, 3))
                self.assertTrue(np.isfinite(session.last_reward))
        before = session.episode
        session.done_since = 0.0
        session.advance(now=2.0)
        self.assertEqual(session.episode, before + 1)

    def test_rgb_flight_autopilot_passes_hoops(self):
        env = Flight3DEnv(seed=123, max_steps=120)
        controller = FlightCVController()
        frame, _ = env.reset(123)
        terminated = False
        for _ in range(120):
            frame, _, terminated, truncated, info = env.step(controller.act(frame))
            if terminated or truncated:
                break
        self.assertFalse(terminated)
        self.assertGreaterEqual(info["rings"], 3)
        self.assertEqual(info["missed"], 0)

    def test_rgb_neon_driver_reaches_checkpoint(self):
        env = NeonDriveEnv(seed=321, max_steps=140)
        driver = NeonDriveCVController()
        frame, _ = env.reset(321)
        terminated = False
        for _ in range(140):
            frame, _, terminated, truncated, info = env.step(driver.act(frame))
            if terminated or truncated:
                break
        self.assertFalse(terminated)
        self.assertGreaterEqual(info["checkpoints"], 1)
        self.assertGreater(info["distance"], 250.0)

    def test_neon_dqn_pixel_contract(self):
        import torch

        encoder = NeonFrameEncoder()
        first = encoder.encode(np.zeros((760, 1100, 3), dtype=np.uint8))
        second_frame = np.zeros((760, 1100, 3), dtype=np.uint8)
        second_frame[300:500, 400:700] = 255
        second = encoder.encode(second_frame)
        self.assertEqual(first.shape, (2, 64, 96))
        self.assertEqual(first.dtype, np.uint8)
        self.assertGreater(int(np.abs(second[1].astype(np.int16) - 128).sum()), 0)
        q_values = NeonDQN(7)(torch.from_numpy(second[None]))
        self.assertEqual(tuple(q_values.shape), (1, 7))

    def test_neural_visualizer_sanitises_non_finite_activity(self):
        values = _normalise(np.asarray([0.2, np.nan, np.inf, -np.inf], dtype=np.float32), 4)
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
