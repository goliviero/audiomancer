"""Example 17 — Tremolo effect on a sine wave.

LFO-based volume modulation at 4 Hz creates a pulsing, classic tremolo.
Three variations concatenated: sine LFO, triangle LFO, square LFO.

Run: python examples/17_tremolo_sine.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak, concat, silence
from fractal.envelopes import Tremolo, FadeInOut
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/17_tremolo_sine.wav")

FREQ = 220.0
DURATION = 4.0
RATE = 4.0
DEPTH = 0.7

base = sine(FREQ, DURATION, amplitude=0.5)

# Three LFO shapes
sine_trem = Tremolo(rate=RATE, depth=DEPTH, shape="sine").apply(base)
tri_trem  = Tremolo(rate=RATE, depth=DEPTH, shape="triangle").apply(base)
sq_trem   = Tremolo(rate=RATE, depth=DEPTH, shape="square").apply(base)

# Each section with quick fade in/out
fade = FadeInOut(fade_in=0.2, fade_out=0.2)
sine_trem = fade.apply(sine_trem)
tri_trem  = fade.apply(tri_trem)
sq_trem   = fade.apply(sq_trem)

gap = silence(0.5)
result = concat(sine_trem, gap, tri_trem, gap, sq_trem)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- tremolo at {RATE}Hz: sine -> triangle -> square LFO")
