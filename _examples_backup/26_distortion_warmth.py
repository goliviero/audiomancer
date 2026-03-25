"""Example 26 — Soft distortion: from clean to warm to fuzzy.

Three versions of the same sine wave:
  1. Clean (no distortion)
  2. Warm (drive=2, subtle saturation)
  3. Fuzz (drive=8, heavy clipping)

Run: python examples/26_distortion_warmth.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak, concat, silence
from fractal.effects import Distortion
from fractal.envelopes import SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/26_distortion_warmth.wav")

DURATION = 3.0
base = sine(220.0, DURATION, amplitude=0.5)

clean = base.copy()
warm  = Distortion(drive=2.0, mix=1.0).process(base)
fuzz  = Distortion(drive=8.0, mix=1.0).process(base)

fade = SmoothFade(fade_in=0.2, fade_out=0.2)
gap = silence(0.5)

result = concat(fade.apply(clean), gap, fade.apply(warm), gap, fade.apply(fuzz))
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- distortion: clean -> warm (drive=2) -> fuzz (drive=8)")
