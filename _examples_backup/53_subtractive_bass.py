"""Example 53 — Subtractive Bass.

Classic analog-style bass: sawtooth oscillator through a low-pass filter
with a filter envelope that sweeps from open to closed. The bread and
butter of every analog synth.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.synth import subtractive
from fractal.envelopes import ADSR
from fractal.effects import Distortion, NormalizePeak, EffectChain
from fractal.export import export_wav
from fractal.music_theory import note_to_hz
from fractal.signal import concat, silence

DURATION = 0.8  # per note

# Filter envelope: fast attack, medium decay -- classic plucky bass
n = int(SAMPLE_RATE * DURATION)
filter_env = np.exp(-np.linspace(0, 6, n))

# Amplitude envelope
adsr = ADSR(attack=0.005, decay=0.15, sustain=0.6, release=0.2)

# Play a bass line: E1 - G1 - A1 - B1
notes = ["E2", "G2", "A2", "B2"]
parts = []

for note in notes:
    freq = note_to_hz(note)
    sig = subtractive(
        oscillator="saw",
        frequency=freq,
        duration_sec=DURATION,
        cutoff_hz=3000,
        resonance=0.3,
        envelope=adsr,
        filter_envelope=filter_env,
        amplitude=0.7,
    )
    parts.append(sig)
    parts.append(silence(0.05))  # small gap between notes

bass_line = concat(*parts)

# Add subtle warmth
fx = EffectChain([
    Distortion(drive=1.5, mix=0.2),
    NormalizePeak(target_db=-3.0),
])
bass_line = fx.process(bass_line, SAMPLE_RATE)

export_wav(bass_line, "outputs/audio/53_subtractive_bass.wav")
print("Exported: outputs/audio/53_subtractive_bass.wav")
