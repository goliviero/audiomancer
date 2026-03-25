"""Example 03 — Alpha binaural beat (10 Hz).

Carrier: 200 Hz. Beat: 10 Hz (alpha range = relaxation, focus).
Left: 200 Hz. Right: 210 Hz. Brain perceives 10 Hz beat.

Use with headphones only — binaural effect requires separate L/R channels.

Run: python examples/03_binaural_alpha.py
Out: outputs/audio/03_binaural_alpha.wav
"""

from pathlib import Path

from fractal.generators import binaural
from fractal.signal import normalize_peak
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/03_binaural_alpha.wav")

# 30 seconds — enough to test the effect without waiting too long
sig = binaural(carrier_hz=200.0, beat_hz=10.0, duration_sec=30.0, amplitude=0.4)
sig = normalize_peak(sig, target_db=-6.0)  # keep it soft for headphone use

export_wav(sig, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — 30s alpha binaural beat (200Hz carrier, 10Hz beat)")
print("     Use with headphones!")
