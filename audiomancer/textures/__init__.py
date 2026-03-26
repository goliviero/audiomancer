"""Textures — ready-to-use evolving ambient presets.

Combines synthesis + modulation + effects into self-contained texture generators.

Usage:
    from audiomancer.textures import generate, REGISTRY
    signal = generate("deep_space", duration_sec=300, seed=42)
"""

from audiomancer.textures._presets import (
    evolving_drone, breathing_pad, deep_space, ocean_bed,
    crystal_shimmer, earth_hum, ethereal_wash, singing_bowl, noise_wash,
)
from audiomancer.textures._registry import REGISTRY, generate, list_textures

__all__ = [
    "evolving_drone", "breathing_pad", "deep_space", "ocean_bed",
    "crystal_shimmer", "earth_hum", "ethereal_wash", "singing_bowl",
    "noise_wash", "REGISTRY", "generate", "list_textures",
]
