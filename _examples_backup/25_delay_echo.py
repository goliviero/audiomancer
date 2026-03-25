"""Example 25 — Delay/echo effect on a melodic sequence.

A short pentatonic phrase with echo at 300ms, fading over 6 repeats.
Classic dub/ambient delay effect.

Run: python examples/25_delay_echo.py
"""

from pathlib import Path

import numpy as np

from fractal.generators import sine
from fractal.signal import normalize_peak, concat, mono_to_stereo
from fractal.effects import Delay
from fractal.envelopes import ADSR
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/25_delay_echo.wav")

FREQS = [440.0, 523.25, 659.25, 523.25, 440.0]
NOTE_DUR = 0.3
GAP = 0.2

adsr = ADSR(attack=0.01, decay=0.05, sustain=0.5, release=0.15)

notes = []
for freq in FREQS:
    note = sine(freq, NOTE_DUR + GAP, amplitude=0.5)
    note = adsr.apply(note)
    notes.append(note)

melody = concat(*notes)

# Add 300ms delay with moderate feedback
delayed = Delay(delay_ms=300, feedback=0.45, mix=0.35, n_echoes=6).process(melody)
delayed = normalize_peak(delayed, target_db=-3.0)
delayed = mono_to_stereo(delayed)

export_wav(delayed, OUTPUT)
print(f"[OK] {OUTPUT} -- 5-note melody with 300ms delay, 6 echoes")
