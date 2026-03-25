"""Example 54 — Supersaw Pad.

7-voice unison sawtooth with wide detune, filtered and reverbed.
The classic trance/EDM supersaw pad sound. Play a Dm7 chord.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.synth import unison, subtractive
from fractal.generators import sawtooth
from fractal.envelopes import ADSR, SmoothFade
from fractal.effects import Reverb, LowPassFilter, NormalizePeak, EffectChain
from fractal.export import export_wav
from fractal.music_theory import chord_hz
from fractal.signal import mix_signals

DURATION = 8.0

# Dm7 chord: D4 F4 A4 C5
chord_freqs = chord_hz("D4", "min7")

# Build each chord note as a 7-voice unison saw
voices = []
for freq in chord_freqs:
    voice = unison(
        generator_fn=sawtooth,
        frequency=freq,
        duration_sec=DURATION,
        voices=7,
        detune_cents=25.0,
        amplitude=0.3,
    )
    voices.append(voice)

# Mix all chord notes
pad = mix_signals(voices)

# Slow fade in/out for pad character
fade = SmoothFade(fade_in=2.0, fade_out=2.0)
pad = fade.apply(pad)

# Filter and reverb
fx = EffectChain([
    LowPassFilter(cutoff_hz=4000),
    Reverb(decay=0.6, mix=0.4),
    NormalizePeak(target_db=-3.0),
])
pad = fx.process(pad, SAMPLE_RATE)

export_wav(pad, "outputs/audio/54_supersaw_pad.wav")
print("Exported: outputs/audio/54_supersaw_pad.wav")
