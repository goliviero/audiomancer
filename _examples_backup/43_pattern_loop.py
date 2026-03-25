"""Example 43 -- Pattern looped 4x: repeat a drum pattern.

Same kick+snare as example 42, but looped 4 times for a 4-bar section.

Run: python examples/43_pattern_loop.py
"""

from pathlib import Path

from fractal.generators import sine, white_noise
from fractal.envelopes import ADSR
from fractal.sequencer import Pattern
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/43_pattern_loop.wav")

BPM = 120
BEAT_SEC = 60.0 / BPM

# Kick + snare sounds
kick = sine(60.0, 0.15, amplitude=0.7)
kick = ADSR(attack=0.005, decay=0.08, sustain=0.1, release=0.05).apply(kick)

snare = white_noise(0.12, amplitude=0.4)
snare = ADSR(attack=0.002, decay=0.05, sustain=0.2, release=0.04).apply(snare)

# 1-bar pattern
bar = Pattern(duration_sec=BEAT_SEC * 4)
bar.add_clip(kick, start_sec=0.0)
bar.add_clip(snare, start_sec=BEAT_SEC)
bar.add_clip(kick, start_sec=BEAT_SEC * 2)
bar.add_clip(snare, start_sec=BEAT_SEC * 3)

# Loop 4x
four_bars = bar.repeat(4)
result = four_bars.render()
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- kick+snare pattern looped 4x ({BPM} BPM, 4 bars)")
