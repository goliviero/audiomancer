"""Example 51 -- Music theory: scales, chords, and progressions.

Demonstrates the music_theory module by generating:
  1. A C major scale as ascending tones
  2. An A minor pentatonic arpeggio
  3. A I-V-vi-IV chord progression (C major)

Run: python examples/51_music_theory_demo.py
"""

from pathlib import Path

from fractal.constants import SAMPLE_RATE
from fractal.generators import sine
from fractal.music_theory import (
    chord_hz,
    note_to_hz,
    progression_hz,
    scale_hz,
)
from fractal.envelopes import ADSR, SmoothFade
from fractal.effects import Reverb, NormalizePeak
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo, concat
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/51_music_theory_demo.wav")

# ---- Part 1: C major scale ----
NOTE_DUR = 0.4
c_major = scale_hz("C4", "major")
scale_tones = []
adsr = ADSR(attack=0.02, decay=0.05, sustain=0.6, release=0.1)
for freq in c_major:
    tone = sine(freq, NOTE_DUR, amplitude=0.4)
    tone = adsr.apply(tone)
    scale_tones.append(tone)
scale_section = concat(*scale_tones)

# ---- Part 2: A minor pentatonic arpeggio ----
penta = scale_hz("A3", "pentatonic_minor")
arp_tones = []
for freq in penta + penta[::-1]:  # Up then down
    tone = sine(freq, 0.3, amplitude=0.35)
    tone = ADSR(attack=0.01, decay=0.1, sustain=0.4, release=0.15).apply(tone)
    arp_tones.append(tone)
arp_section = concat(*arp_tones)

# ---- Part 3: I-V-vi-IV progression ----
prog = progression_hz("C4", "I_V_vi_IV")
chord_tones = []
for chord_freqs in prog:
    # Stack all notes in the chord
    layers = [sine(f, 2.0, amplitude=0.15) for f in chord_freqs]
    chord_signal = mix_signals(layers)
    chord_signal = SmoothFade(fade_in=0.3, fade_out=0.3).apply(chord_signal)
    chord_tones.append(chord_signal)
prog_section = concat(*chord_tones)

# ---- Combine all sections ----
from fractal.signal import silence
gap = silence(0.5)
full = concat(scale_section, gap, arp_section, gap, prog_section)
full = normalize_peak(full, target_db=-3.0)
full = mono_to_stereo(full)

export_wav(full, OUTPUT)
print(f"[OK] {OUTPUT}")
print("     C major scale -> A minor pentatonic arpeggio -> I-V-vi-IV progression")
print(f"     Duration: {full.shape[0] / SAMPLE_RATE:.1f}s")
