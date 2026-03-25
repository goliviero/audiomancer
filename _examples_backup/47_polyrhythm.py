"""Example 47 -- Polyrhythm: 3 against 4.

Two layers:
  - Layer A: 4 evenly spaced hits per cycle (kick)
  - Layer B: 3 evenly spaced hits per cycle (high tone)

Creates the classic 3:4 polyrhythm. Looped 4 times.

Run: python examples/47_polyrhythm.py
"""

from pathlib import Path

from fractal.generators import sine, white_noise
from fractal.envelopes import ADSR
from fractal.sequencer import Pattern
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/47_polyrhythm.wav")

CYCLE_SEC = 2.0  # one full cycle

# Sounds
low = sine(80.0, 0.12, amplitude=0.5)
low = ADSR(attack=0.005, decay=0.06, sustain=0.1, release=0.04).apply(low)

high = sine(600.0, 0.08, amplitude=0.35)
high = ADSR(attack=0.003, decay=0.03, sustain=0.1, release=0.03).apply(high)

# Pattern: 4 low hits + 3 high hits in the same cycle
poly = Pattern(duration_sec=CYCLE_SEC)

# 4 evenly spaced
for i in range(4):
    poly.add_clip(low, start_sec=i * CYCLE_SEC / 4, track_name="4-beat")

# 3 evenly spaced
for i in range(3):
    poly.add_clip(high, start_sec=i * CYCLE_SEC / 3, track_name="3-beat")

# Loop 4x
full = poly.repeat(4)
result = full.render()
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- 3:4 polyrhythm, 4 cycles")
