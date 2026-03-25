"""Example 01 — Simple sine wave at 440 Hz (A4).

The simplest possible Fractal script.
Generates a pure tone, normalizes it, exports to WAV.

Run: python examples/01_sine_440hz.py
Out: outputs/audio/01_sine_440hz.wav
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/01_sine_440hz.wav")

sig = sine(frequency=440.0, duration_sec=5.0, amplitude=0.5)
sig = normalize_peak(sig, target_db=-3.0)
export_wav(sig, OUTPUT, sample_rate=SAMPLE_RATE)

print(f"[OK] {OUTPUT} — 5s pure A4 tone, 440 Hz")
