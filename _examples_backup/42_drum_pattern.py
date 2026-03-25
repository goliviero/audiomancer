"""Example 42 -- Basic drum pattern: kick + snare on beats.

Creates simple kick (60Hz sine burst) and snare (noise burst) sounds
and places them on a 4-beat pattern.

Kick on beats 0, 2. Snare on beats 1, 3.

Run: python examples/42_drum_pattern.py
"""

from pathlib import Path

from fractal.generators import sine, white_noise
from fractal.envelopes import ADSR, SmoothFade
from fractal.sequencer import Pattern
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/42_drum_pattern.wav")

BPM = 120
BEAT_SEC = 60.0 / BPM  # 0.5s per beat

# Kick: low sine burst
kick = sine(60.0, 0.15, amplitude=0.7)
kick = ADSR(attack=0.005, decay=0.08, sustain=0.1, release=0.05).apply(kick)

# Snare: noise burst
snare = white_noise(0.12, amplitude=0.4)
snare = ADSR(attack=0.002, decay=0.05, sustain=0.2, release=0.04).apply(snare)

# Build 1-bar pattern (4 beats)
bar = Pattern(duration_sec=BEAT_SEC * 4)
bar.add_clip(kick, start_sec=0.0, track_name="kick")
bar.add_clip(snare, start_sec=BEAT_SEC * 1, track_name="snare")
bar.add_clip(kick, start_sec=BEAT_SEC * 2, track_name="kick")
bar.add_clip(snare, start_sec=BEAT_SEC * 3, track_name="snare")

result = bar.render()
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- kick+snare, 1 bar at {BPM} BPM")
