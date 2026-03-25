"""Example 05 — Pentatonic arpeggio: notes concatenated in sequence.

Demonstrates concat() + per-note generation.
Scale: A minor pentatonic — A3, C4, D4, E4, G4, A4

Run: python examples/05_pentatonic_arpeggio.py
Out: outputs/audio/05_pentatonic_arpeggio.wav
"""

from pathlib import Path

import numpy as np

from fractal.generators import sine
from fractal.signal import normalize_peak, concat
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/05_pentatonic_arpeggio.wav")

# A minor pentatonic: A3 C4 D4 E4 G4 A4
NOTE_FREQS = [220.00, 261.63, 293.66, 329.63, 392.00, 440.00]
NOTE_DUR   = 0.4   # seconds per note
AMPLITUDE  = 0.5

notes = []
for freq in NOTE_FREQS:
    note = sine(freq, NOTE_DUR, amplitude=AMPLITUDE)
    # Apply a quick linear fade-out to avoid clicks between notes
    fade = np.linspace(1.0, 0.0, len(note))
    note = note * fade
    notes.append(note)

# Play twice
sequence = concat(*notes, *notes)
sequence = normalize_peak(sequence, target_db=-3.0)

export_wav(sequence, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — A minor pentatonic arpeggio x2, {NOTE_DUR}s per note")
