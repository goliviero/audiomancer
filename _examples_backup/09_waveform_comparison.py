"""Example 09 — Waveform comparison: all 4 oscillators, same note.

Generates sine, square, sawtooth, and triangle at 220 Hz (A3),
then concatenates them with a short silence between each.
Good for hearing how waveform shape affects timbre.

Run: python examples/09_waveform_comparison.py
Out: outputs/audio/09_waveform_comparison.wav
"""

from pathlib import Path

from fractal.generators import sine, square, sawtooth, triangle
from fractal.signal import silence, normalize_peak, concat
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/09_waveform_comparison.wav")

FREQ      = 220.0   # A3
DURATION  = 2.0     # seconds per waveform
GAP       = 0.3     # silence between waveforms
AMP       = 0.5

gap = silence(GAP)

result = concat(
    sine(FREQ, DURATION, AMP),     gap,
    square(FREQ, DURATION, AMP),   gap,
    sawtooth(FREQ, DURATION, AMP), gap,
    triangle(FREQ, DURATION, AMP),
)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — sine -> square -> sawtooth -> triangle at {FREQ}Hz")
print("     2s each with 0.3s silence gap")
