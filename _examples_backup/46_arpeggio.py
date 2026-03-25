"""Example 46 -- Arpeggio via sequencer: notes placed on beats.

A C major arpeggio (C4-E4-G4-C5) placed one note per beat at 140 BPM.
Looped 2x for 8 notes total.

Run: python examples/46_arpeggio.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.envelopes import ADSR
from fractal.sequencer import Pattern, Sequencer
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/46_arpeggio.wav")

BPM = 140
BEAT_SEC = 60.0 / BPM

# C major arpeggio frequencies
FREQS = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5

adsr = ADSR(attack=0.01, decay=0.08, sustain=0.3, release=0.15)

# Build 1-bar pattern (4 beats, 1 note per beat)
arp = Pattern(duration_sec=BEAT_SEC * 4)
for i, freq in enumerate(FREQS):
    note = sine(freq, BEAT_SEC * 0.9, amplitude=0.4)
    note = adsr.apply(note)
    arp.add_clip(note, start_sec=i * BEAT_SEC)

# Loop 2x
full = arp.repeat(2)
result = full.render()
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- C major arpeggio, 8 notes at {BPM} BPM")
