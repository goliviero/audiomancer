"""Example 11 — Linear fade in/out on a drone.

Compares Phase 1 (abrupt start) vs Phase 2 (proper fades).
Generates a 55Hz drone with 3s fade in + 3s fade out.

Run: python examples/11_fade_in_out.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak, mono_to_stereo
from fractal.envelopes import FadeInOut
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/11_fade_in_out.wav")

drone = sine(55.0, 10.0, amplitude=0.5)
drone = FadeInOut(fade_in=3.0, fade_out=3.0).apply(drone)
drone = normalize_peak(drone, target_db=-3.0)
drone = mono_to_stereo(drone)

export_wav(drone, OUTPUT)
print(f"[OK] {OUTPUT} -- 10s drone at 55Hz, 3s fade in + 3s fade out")
