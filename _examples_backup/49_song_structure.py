"""Example 49 -- Full song structure: 16 bars, A-B-A form.

A: sparse (kick only + bass drone)
B: full (kick + snare + hihat + bass)
Structure: A(4) - B(8) - A(4)

Run: python examples/49_song_structure.py
"""

from pathlib import Path

from fractal.generators import sine, white_noise
from fractal.envelopes import ADSR, SmoothFade
from fractal.sequencer import Pattern, Sequencer
from fractal.signal import normalize_peak, mix_signals, mono_to_stereo
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/49_song_structure.wav")

BPM = 100
BEAT_SEC = 60.0 / BPM

# --- Sounds ---
kick = sine(55.0, 0.15, amplitude=0.5)
kick = ADSR(attack=0.005, decay=0.08, sustain=0.1, release=0.05).apply(kick)

snare = white_noise(0.1, amplitude=0.3)
snare = ADSR(attack=0.002, decay=0.04, sustain=0.15, release=0.03).apply(snare)

hihat = white_noise(0.03, amplitude=0.15)
hihat = ADSR(attack=0.001, decay=0.015, sustain=0.05, release=0.01).apply(hihat)

# --- Patterns (1 bar = 4 beats) ---
# Section A: kick on 1 and 3
pat_a = Pattern(duration_sec=BEAT_SEC * 4)
pat_a.add_clip(kick, start_sec=0.0)
pat_a.add_clip(kick, start_sec=BEAT_SEC * 2)

# Section B: full kit
pat_b = Pattern(duration_sec=BEAT_SEC * 4)
pat_b.add_clip(kick, start_sec=0.0)
pat_b.add_clip(snare, start_sec=BEAT_SEC)
pat_b.add_clip(kick, start_sec=BEAT_SEC * 2)
pat_b.add_clip(snare, start_sec=BEAT_SEC * 3)
for i in range(4):
    pat_b.add_clip(hihat, start_sec=BEAT_SEC * i)

# --- Arrangement: A(4 bars) - B(8 bars) - A(4 bars) ---
seq = Sequencer(tempo_bpm=BPM)

# A section: 4 bars
a_section = pat_a.repeat(4)
seq.add_pattern(a_section, start_beat=0)

# B section: 8 bars
b_section = pat_b.repeat(8)
seq.add_pattern(b_section, start_beat=16)

# A section again: 4 bars
a_outro = pat_a.repeat(4)
seq.add_pattern(a_outro, start_beat=48)

drums = seq.render(tail_sec=0.5)
drums = mono_to_stereo(drums)

# --- Bass drone underneath ---
total_sec = seq.duration_sec + 0.5
bass = sine(55.0, total_sec, amplitude=0.1)
bass = SmoothFade(fade_in=2.0, fade_out=2.0).apply(bass)
bass = mono_to_stereo(bass)

result = mix_signals([drums, bass], volumes_db=[-3.0, -9.0])
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- ABA form: 16 bars at {BPM} BPM (A=sparse, B=full)")
