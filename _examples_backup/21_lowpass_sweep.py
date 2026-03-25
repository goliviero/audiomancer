"""Example 21 — Low-pass filter sweep on white noise.

Generates white noise then applies increasingly open LPF cutoffs.
Classic synth technique: sounds like a filter opening up.

Run: python examples/21_lowpass_sweep.py
"""

from pathlib import Path

from fractal.generators import white_noise
from fractal.signal import normalize_peak, concat, mono_to_stereo
from fractal.effects import LowPassFilter
from fractal.envelopes import FadeInOut
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/21_lowpass_sweep.wav")

DURATION = 3.0

cutoffs = [200, 500, 1000, 2000, 5000, 10000]
sections = []

for cutoff in cutoffs:
    noise = white_noise(DURATION, amplitude=0.5)
    filtered = LowPassFilter(cutoff_hz=cutoff).process(noise)
    filtered = FadeInOut(fade_in=0.1, fade_out=0.1).apply(filtered)
    sections.append(filtered)

result = concat(*sections)
result = normalize_peak(result, target_db=-3.0)
result = mono_to_stereo(result)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- LPF sweep: {cutoffs} Hz, {DURATION}s each")
