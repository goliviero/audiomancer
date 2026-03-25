"""Example 14 — Exponential fade in: slow start, fast finish.

Exponential curves feel more natural than linear because human loudness
perception is logarithmic. This example shows a 5s exp fade in on pink noise.

Run: python examples/14_exponential_fade.py
"""

from pathlib import Path

from fractal.generators import pink_noise
from fractal.signal import normalize_peak, mono_to_stereo
from fractal.envelopes import ExponentialFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/14_exponential_fade.wav")

noise = pink_noise(10.0, amplitude=0.6)
noise = ExponentialFade(fade_in=5.0, fade_out=3.0, steepness=6.0).apply(noise)
noise = normalize_peak(noise, target_db=-3.0)
noise = mono_to_stereo(noise)

export_wav(noise, OUTPUT)
print(f"[OK] {OUTPUT} -- 10s pink noise, exponential fade in (5s) + out (3s)")
