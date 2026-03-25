"""Example 19 — Binaural beat with proper envelopes (Phase 1 + 2 combined).

This is the "fixed" version of example 03: same alpha binaural beat,
but now with smooth fade in (5s), smooth fade out (5s), and a gentle swell.
Night and day difference from the abrupt Phase 1 version.

Run: python examples/19_binaural_with_envelope.py
"""

from pathlib import Path

from fractal.generators import binaural
from fractal.signal import normalize_peak
from fractal.envelopes import SmoothFade, Swell
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/19_binaural_with_envelope.wav")

DURATION = 60.0

sig = binaural(carrier_hz=200.0, beat_hz=10.0, duration_sec=DURATION, amplitude=0.4)

# Swell in over 8 seconds
sig = Swell(rise_time=8.0, curve_type="cosine").apply(sig)
# Smooth fade out over 8 seconds
sig = SmoothFade(fade_out=8.0).apply(sig)

sig = normalize_peak(sig, target_db=-6.0)

export_wav(sig, OUTPUT)
print(f"[OK] {OUTPUT} -- 60s alpha binaural, 8s cosine swell in + 8s smooth fade out")
print("     Compare with 03_binaural_alpha.wav to hear the difference!")
