"""Example 16 — Rhythmic gate on a noise texture.

Chops a continuous noise bed into a rhythmic pattern using the Gate
envelope. Classic sidechain / trance gate effect.

Run: python examples/16_gate_rhythm.py
"""

from pathlib import Path

from fractal.generators import pink_noise, sine
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.envelopes import Gate, FadeInOut
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/16_gate_rhythm.wav")

DURATION = 10.0

# Base: pink noise + sub tone
noise = pink_noise(DURATION, amplitude=0.5)
sub   = sine(60.0, DURATION, amplitude=0.3)
bed   = mix_signals([noise, sub])

# Gate: 1/8 note at 120 BPM = 0.25s per beat, 1/8 = 0.125s
# On for 0.1s, off for 0.15s -> rhythmic chop
gate = Gate(on_time=0.1, off_time=0.15, smooth_ms=5.0)
gated = gate.apply(bed)

# Wrap with fade in/out
gated = FadeInOut(fade_in=1.0, fade_out=2.0).apply(gated)
gated = normalize_peak(gated, target_db=-3.0)
gated = mono_to_stereo(gated)

export_wav(gated, OUTPUT)
print(f"[OK] {OUTPUT} -- 10s gated noise+sub, 0.1s on / 0.15s off, 5ms smooth")
