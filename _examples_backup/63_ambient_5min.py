"""Example 63 — 5-Minute Ambient Texture.

The flagship generative example: a complete 5-minute ambient piece
generated from a single function call. Uses D minor pentatonic with
3 layers of evolving pads and sparse melodic elements.

Change the seed to generate an entirely different piece.
"""

from fractal.constants import SAMPLE_RATE
from fractal.generative import ambient_texture
from fractal.export import export_wav

# Generate 5 minutes of ambient texture
print("Generating 5-minute ambient texture (this may take a moment)...")
texture = ambient_texture(
    key="D3",
    scale_type="pentatonic_minor",
    duration_sec=300,  # 5 minutes
    layers=3,
    amplitude=0.5,
    seed=42,
)

export_wav(texture, "outputs/audio/63_ambient_5min.wav")
print("Exported: outputs/audio/63_ambient_5min.wav")
print(f"  Duration: {len(texture) / SAMPLE_RATE:.0f}s ({len(texture) / SAMPLE_RATE / 60:.1f} min)")
print("  Key: D minor pentatonic")
print("  Layers: drone pad + 3 melodic layers")
print("  Seed: 42 (change for different results)")
