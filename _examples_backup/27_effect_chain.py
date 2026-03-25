"""Example 27 — Effect chain: multiple effects applied in series.

Shows the EffectChain class: HPF -> EQ -> Reverb -> Normalize.
Applied to a rich drone, this simulates a basic mastering chain.

Run: python examples/27_effect_chain.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.effects import EffectChain, HighPassFilter, EQ, Reverb, NormalizePeak
from fractal.envelopes import Swell, SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/27_effect_chain.wav")

DURATION = 15.0

# Build a raw drone
fund  = sine(110.0, DURATION, amplitude=0.4)
fifth = sine(165.0, DURATION, amplitude=0.25)
noise = pink_noise(DURATION, amplitude=0.15)
raw = mix_signals([fund, fifth, noise])

# Apply envelope
raw = Swell(rise_time=5.0, curve_type="cosine").apply(raw)
raw = SmoothFade(fade_out=4.0).apply(raw)

# Mastering chain
chain = EffectChain([
    HighPassFilter(cutoff_hz=40),          # remove subsonic rumble
    EQ(bands=[
        {"type": "low_shelf", "freq": 100, "gain_db": 2.0},
        {"type": "peak", "freq": 3000, "gain_db": 1.5},
        {"type": "high_shelf", "freq": 10000, "gain_db": -2.0},
    ]),
    Reverb(decay=0.3, mix=0.15, room_size="medium"),
    NormalizePeak(target_db=-1.0),
])

mastered = chain.process(raw)
mastered = mono_to_stereo(mastered)

export_wav(mastered, OUTPUT)
print(f"[OK] {OUTPUT} -- 15s drone through mastering chain: HPF+EQ+Reverb+Normalize")
