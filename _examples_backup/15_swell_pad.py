"""Example 15 — Ambient pad with swell envelope.

A rich harmonic pad slowly rises over 8 seconds, then holds.
Classic ambient technique for building tension or atmosphere.

Run: python examples/15_swell_pad.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.envelopes import Swell, SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/15_swell_pad.wav")

DURATION = 20.0

# Rich pad: fundamental + detuned layers for warmth
fund  = sine(110.0, DURATION, amplitude=0.4)
oct   = sine(220.0, DURATION, amplitude=0.25)
fifth = sine(165.0, DURATION, amplitude=0.2)
# Slight detune for chorus-like width
detune = sine(110.7, DURATION, amplitude=0.15)

pad = mix_signals([fund, oct, fifth, detune], volumes_db=[0.0, -4.0, -6.0, -8.0])

# Swell in over 8 seconds
pad = Swell(rise_time=8.0, peak=1.0, curve_type="cosine").apply(pad)
# Smooth fade out at the end
pad = SmoothFade(fade_out=5.0).apply(pad)

pad = normalize_peak(pad, target_db=-3.0)
pad = mono_to_stereo(pad)

export_wav(pad, OUTPUT)
print(f"[OK] {OUTPUT} -- 20s ambient pad, 8s cosine swell + 5s smooth fade out")
