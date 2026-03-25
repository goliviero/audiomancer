"""Example 12 — ADSR-shaped keyboard notes.

Each note gets a classic synth ADSR envelope: quick attack, short decay,
sustained body, smooth release. Concatenated into a melody.

Run: python examples/12_adsr_keys.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak, concat
from fractal.envelopes import ADSR
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/12_adsr_keys.wav")

# C major scale
NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NOTE_DUR = 0.8

adsr = ADSR(attack=0.02, decay=0.1, sustain=0.6, release=0.3)

shaped_notes = []
for freq in NOTES:
    note = sine(freq, NOTE_DUR, amplitude=0.6)
    note = adsr.apply(note)
    shaped_notes.append(note)

melody = concat(*shaped_notes)
melody = normalize_peak(melody, target_db=-3.0)

export_wav(melody, OUTPUT)
print(f"[OK] {OUTPUT} -- C major scale, ADSR (A=20ms D=100ms S=0.6 R=300ms)")
