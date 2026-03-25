"""Example 08 — Schumann resonance binaural beat (7.83 Hz).

The Schumann resonance is Earth's electromagnetic resonance frequency.
7.83 Hz falls in the theta/alpha border — associated with deep relaxation.
Carrier: 136.1 Hz (OM frequency / Earth year tone).

Run: python examples/08_schumann_resonance.py
Out: outputs/audio/08_schumann_resonance.wav
"""

from pathlib import Path

import numpy as np

from fractal.generators import binaural, sine
from fractal.signal import mix_signals, normalize_peak
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/08_schumann_resonance.wav")

DURATION = 60.0   # 1 minute

# Binaural beat at 7.83 Hz — the Schumann resonance
beat = binaural(carrier_hz=136.1, beat_hz=7.83, duration_sec=DURATION, amplitude=0.35)

# Add a soft 136.1 Hz mono tone as a subtle carrier reinforcement
carrier = sine(136.1, DURATION, amplitude=0.15)
# Convert mono to stereo manually
carrier_stereo = np.column_stack([carrier, carrier])

result = mix_signals([beat, carrier_stereo], volumes_db=[0.0, -6.0])
result = normalize_peak(result, target_db=-6.0)

export_wav(result, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — 60s Schumann 7.83Hz binaural + 136.1Hz carrier")
print("     Use with headphones!")
