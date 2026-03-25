"""Example 23 — Stereo width comparison: narrow, normal, wide.

Takes a binaural signal and demonstrates 3 width settings:
  1. width=0.3 (narrow, almost mono)
  2. width=1.0 (original)
  3. width=2.0 (hyper-wide)

Run: python examples/23_stereo_widening.py
"""

from pathlib import Path

from fractal.generators import binaural
from fractal.signal import normalize_peak, concat, silence
from fractal.effects import StereoWidth
from fractal.envelopes import SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/23_stereo_widening.wav")

DURATION = 5.0
sig = binaural(200.0, 10.0, DURATION, amplitude=0.4)

narrow = StereoWidth(width=0.3).process(sig)
normal = sig.copy()
wide   = StereoWidth(width=2.0).process(sig)

fade = SmoothFade(fade_in=0.5, fade_out=0.5)
gap = silence(0.5, stereo=True)

result = concat(fade.apply(narrow), gap, fade.apply(normal), gap, fade.apply(wide))
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- stereo width: narrow (0.3) -> normal (1.0) -> wide (2.0)")
