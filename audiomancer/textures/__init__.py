"""Textures — ready-to-use evolving ambient presets.

Combines synthesis + modulation + effects into self-contained texture generators.

Usage:
    from audiomancer.textures import generate, REGISTRY
    signal = generate("deep_space", duration_sec=300, seed=42)
"""

from audiomancer.textures._presets import (
    breathing_pad,
    crystal_shimmer,
    deep_space,
    earth_hum,
    ethereal_wash,
    evolving_drone,
    noise_wash,
    ocean_bed,
    singing_bowl,
)
from audiomancer.textures._registry import REGISTRY, generate, list_textures

__all__ = [
    "evolving_drone", "breathing_pad", "deep_space", "ocean_bed",
    "crystal_shimmer", "earth_hum", "ethereal_wash", "singing_bowl",
    "noise_wash", "REGISTRY", "generate", "list_textures",
]
