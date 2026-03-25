"""Example 02 — C Major chord: mix three sine waves.

Demonstrates mixing multiple generators at different volumes.
C4=261.63 Hz, E4=329.63 Hz, G4=392.00 Hz

Run: python examples/02_major_chord.py
Out: outputs/audio/02_major_chord.wav
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import mix_signals, normalize_peak
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/02_major_chord.wav")

DURATION = 4.0

# C Major triad (C4, E4, G4)
c4 = sine(261.63, DURATION, amplitude=0.5)
e4 = sine(329.63, DURATION, amplitude=0.4)
g4 = sine(392.00, DURATION, amplitude=0.4)

# Mix — root slightly louder
chord = mix_signals([c4, e4, g4], volumes_db=[0.0, -2.0, -2.0])
chord = normalize_peak(chord, target_db=-3.0)

export_wav(chord, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — C major chord (C4+E4+G4), 4s")
