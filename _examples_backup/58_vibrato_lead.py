"""Example 58 — Vibrato Lead.

A sustained lead tone with vibrato applied. Demonstrates how vibrato
adds life and expression to a static tone. Three intensities compared.
"""

from fractal.generators import sine
from fractal.modulation import apply_vibrato
from fractal.envelopes import SmoothFade
from fractal.effects import Reverb, NormalizePeak, EffectChain
from fractal.export import export_wav
from fractal.music_theory import note_to_hz
from fractal.constants import SAMPLE_RATE

DURATION = 4.0

freq = note_to_hz("A4")
tone = sine(freq, DURATION, amplitude=0.6)

# Subtle vibrato (classical style)
subtle = apply_vibrato(tone, rate=5.0, depth_cents=10.0)

# Medium vibrato (expressive)
medium = apply_vibrato(tone, rate=5.5, depth_cents=30.0)

# Wide vibrato (dramatic)
wide = apply_vibrato(tone, rate=6.0, depth_cents=80.0)

# Add fade and reverb
fade = SmoothFade(fade_in=0.3, fade_out=0.5)
fx = EffectChain([
    Reverb(decay=0.3, mix=0.2),
    NormalizePeak(target_db=-3.0),
])

for name, sig in [("subtle", subtle), ("medium", medium), ("wide", wide)]:
    sig = fade.apply(sig)
    sig = fx.process(sig, SAMPLE_RATE)
    path = f"outputs/audio/58_vibrato_{name}.wav"
    export_wav(sig, path)
    print(f"Exported: {path}")
