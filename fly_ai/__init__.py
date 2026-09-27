"""Pixel-only virtual fly agent for the Pygame CV games."""

from .brain import MushroomBodyAgent
from .envs import ENV_REGISTRY, make_env
from .vision import CompoundEye, KnifeCompoundEye, make_eye

__all__ = [
    "CompoundEye", "KnifeCompoundEye", "make_eye", "MushroomBodyAgent", "ENV_REGISTRY", "make_env"
]
