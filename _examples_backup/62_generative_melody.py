"""Example 62 — Generative Melody.

A randomly generated melody over A minor pentatonic, rendered with
the pluck preset. Change the seed to get a different melody.
"""

from fractal.music_theory import scale_hz
from fractal.generative import phrase_generator
from fractal.effects import Reverb, NormalizePeak, EffectChain
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

# A minor pentatonic across 2 octaves
notes = scale_hz("A3", "pentatonic_minor", octaves=2)

# Generate a 4-bar phrase at 100 BPM
melody = phrase_generator(
    scale_notes=notes,
    preset="pluck",
    tempo_bpm=100,
    measures=4,
    density=0.4,
    amplitude=0.5,
    seed=42,
)

# Add reverb
fx = EffectChain([
    Reverb(decay=0.4, mix=0.25),
    NormalizePeak(target_db=-3.0),
])
melody = fx.process(melody, SAMPLE_RATE)

export_wav(melody, "outputs/audio/62_generative_melody.wav")
print("Exported: outputs/audio/62_generative_melody.wav")
print("  Scale: A minor pentatonic (2 octaves)")
print("  Preset: pluck")
print("  Tempo: 100 BPM, 4 bars, density=0.4, seed=42")
