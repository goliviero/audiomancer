"""Example 22 — EQ-sculpted drone: shape a flat noise into a warm pad.

Takes pink noise and carves it with a 3-band EQ:
  - Boost sub-bass (80 Hz) for warmth
  - Boost presence (2.5 kHz) for shimmer
  - Cut highs (8 kHz) to tame harshness

Run: python examples/22_eq_sculpted_drone.py
"""

from pathlib import Path

from fractal.generators import pink_noise
from fractal.signal import normalize_peak, mono_to_stereo
from fractal.effects import EQ
from fractal.envelopes import SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/22_eq_sculpted_drone.wav")

noise = pink_noise(15.0, amplitude=0.5)

eq = EQ(bands=[
    {"type": "low_shelf", "freq": 80, "gain_db": 4.0},
    {"type": "peak", "freq": 2500, "gain_db": 3.0},
    {"type": "high_shelf", "freq": 8000, "gain_db": -4.0},
])
shaped = eq.process(noise)
shaped = SmoothFade(fade_in=3.0, fade_out=3.0).apply(shaped)
shaped = normalize_peak(shaped, target_db=-3.0)
shaped = mono_to_stereo(shaped)

export_wav(shaped, OUTPUT)
print(f"[OK] {OUTPUT} -- 15s pink noise sculpted with 3-band EQ")
