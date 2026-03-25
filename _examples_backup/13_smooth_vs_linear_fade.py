"""Example 13 — Smooth (cosine) vs linear fade comparison.

Generates two versions of the same drone:
  A) Linear fade out (abrupt cutoff feel)
  B) Smooth cosine fade out (S-curve, more natural)

They're concatenated with a short gap so you can hear the difference.

Run: python examples/13_smooth_vs_linear_fade.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak, concat, silence
from fractal.envelopes import FadeInOut, SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/13_smooth_vs_linear_fade.wav")

DURATION = 5.0
FADE = 3.0

drone = sine(110.0, DURATION, amplitude=0.5)

# Version A: linear fade out
linear = FadeInOut(fade_out=FADE).apply(drone)

# Version B: smooth cosine fade out
smooth = SmoothFade(fade_out=FADE).apply(drone)

# Concatenate: linear -- gap -- smooth
result = concat(linear, silence(1.0), smooth)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- linear fade (5s) then smooth fade (5s), 1s gap between")
