"""Watch a saved fly policy play in a regular Pygame window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.pop("SDL_VIDEODRIVER", None)

import pygame

from .brain import MushroomBodyAgent
from .envs import make_env
from .vision import make_eye


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=0,
                        help="number of episodes; 0 means play until the window is closed")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    agent, metadata = MushroomBodyAgent.load(args.model)
    game = metadata.get("game")
    if not game:
        raise SystemExit("Checkpoint has no game metadata")
    env = make_env(game, seed=80_000, max_steps=1800)
    eye = make_eye(game)
    if eye.output_dim != agent.input_dim or env.n_actions != agent.n_actions:
        raise SystemExit("Checkpoint schema does not match the selected environment")

    pygame.display.quit()
    pygame.display.init()
    window = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption(f"Virtual fly: {game}")
    clock = pygame.time.Clock()
    running = True
    episode = 0
    while running and (args.episodes <= 0 or episode < args.episodes):
        frame, _ = env.reset(seed=80_000 + episode)
        eye.reset()
        for _ in range(env.max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            if not running:
                break
            action = agent.act(agent.features(eye.encode(frame)))
            frame, _, terminated, truncated, _ = env.step(action)
            window.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)
            if terminated or truncated:
                break
        episode += 1
    pygame.quit()


if __name__ == "__main__":
    main()
