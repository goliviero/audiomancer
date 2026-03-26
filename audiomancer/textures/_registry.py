"""Registry and dispatcher for texture presets."""

import numpy as np

from audiomancer import SAMPLE_RATE
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

REGISTRY = {
    "evolving_drone": {
        "fn": evolving_drone,
        "description": "Rich harmonic drone with drifting amplitude and filter",
        "role": "foundation",
        "tonal": True,
    },
    "breathing_pad": {
        "fn": breathing_pad,
        "description": "Chord pad with slow inhale/exhale movement",
        "role": "foundation",
        "tonal": True,
    },
    "deep_space": {
        "fn": deep_space,
        "description": "Dark vast texture — sculpted brown noise",
        "role": "foundation",
        "tonal": False,
    },
    "ocean_bed": {
        "fn": ocean_bed,
        "description": "Underwater ambient — pink noise with filter waves",
        "role": "bed",
        "tonal": False,
    },
    "crystal_shimmer": {
        "fn": crystal_shimmer,
        "description": "High-frequency cluster — sparkly, ethereal",
        "role": "overlay",
        "tonal": True,
    },
    "earth_hum": {
        "fn": earth_hum,
        "description": "Sub-bass drone + brown noise — primal, grounding",
        "role": "foundation",
        "tonal": True,
    },
    "ethereal_wash": {
        "fn": ethereal_wash,
        "description": "Saw pad drenched in reverb — dreamy, floating",
        "role": "foundation",
        "tonal": True,
    },
    "singing_bowl": {
        "fn": singing_bowl,
        "description": "Inharmonic resonance — metallic, meditative",
        "role": "accent",
        "tonal": True,
    },
    "noise_wash": {
        "fn": noise_wash,
        "description": "Colored noise with evolving filter — ambient bed",
        "role": "bed",
        "tonal": False,
    },
}


def generate(name: str, duration_sec: float = 300.0,
             seed: int | None = None,
             sample_rate: int = SAMPLE_RATE,
             **kwargs) -> np.ndarray:
    """Generate a texture by name.

    Args:
        name: Texture name (see REGISTRY keys).
        duration_sec: Duration in seconds.
        seed: Random seed for reproducibility.
        sample_rate: Sample rate.
        **kwargs: Additional arguments passed to the texture function.

    Returns:
        Stereo audio signal.

    Raises:
        ValueError: If texture name is not found.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise ValueError(f"Unknown texture: {name!r}. Available: {available}")
    return entry["fn"](duration_sec=duration_sec, seed=seed,
                       sample_rate=sample_rate, **kwargs)


def list_textures() -> list[dict]:
    """List all available textures with metadata."""
    return [
        {"name": name, **{k: v for k, v in info.items() if k != "fn"}}
        for name, info in REGISTRY.items()
    ]
