"""Example 45 -- Multi-pattern arrangement: intro + verse + chorus.

Three distinct patterns placed sequentially on the timeline:
  - Intro: just kick (sparse)
  - Verse: kick + snare
  - Chorus: kick + snare + hi-hat (denser)

Run: python examples/45_arrangement.py
"""

from pathlib import Path

from fractal.generators import sine, white_noise
from fractal.envelopes import ADSR
from fractal.sequencer import Pattern, Sequencer
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/45_arrangement.wav")

BPM = 110
BEAT_SEC = 60.0 / BPM

# Sounds
kick = sine(55.0, 0.15, amplitude=0.6)
kick = ADSR(attack=0.005, decay=0.08, sustain=0.1, release=0.05).apply(kick)

snare = white_noise(0.1, amplitude=0.35)
snare = ADSR(attack=0.002, decay=0.04, sustain=0.15, release=0.03).apply(snare)

hihat = white_noise(0.04, amplitude=0.2)
hihat = ADSR(attack=0.001, decay=0.02, sustain=0.05, release=0.01).apply(hihat)

# Intro: kick only (2 bars)
intro = Pattern(duration_sec=BEAT_SEC * 8)
for beat in [0, 4]:
    intro.add_clip(kick, start_sec=beat * BEAT_SEC)

# Verse: kick + snare (2 bars)
verse = Pattern(duration_sec=BEAT_SEC * 8)
for beat in [0, 2, 4, 6]:
    verse.add_clip(kick, start_sec=beat * BEAT_SEC)
for beat in [1, 3, 5, 7]:
    verse.add_clip(snare, start_sec=beat * BEAT_SEC)

# Chorus: kick + snare + hi-hat on every beat (2 bars)
chorus = Pattern(duration_sec=BEAT_SEC * 8)
for beat in [0, 2, 4, 6]:
    chorus.add_clip(kick, start_sec=beat * BEAT_SEC)
for beat in [1, 3, 5, 7]:
    chorus.add_clip(snare, start_sec=beat * BEAT_SEC)
for beat in range(8):
    chorus.add_clip(hihat, start_sec=beat * BEAT_SEC)

# Arrange: intro -> verse -> chorus -> chorus
seq = Sequencer(tempo_bpm=BPM)
seq.add_pattern(intro, start_beat=0)
seq.add_pattern(verse, start_beat=8)
seq.add_pattern(chorus, start_beat=16)
seq.add_pattern(chorus, start_beat=24)

result = seq.render(tail_sec=0.5)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- arrangement: intro -> verse -> chorus -> chorus ({BPM} BPM)")
