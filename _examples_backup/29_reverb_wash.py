"""Example 29 — Reverb wash: 100% wet reverb on a note sequence.

By setting mix=1.0 (fully wet), the direct signal disappears and you
hear only the reverb reflections. Creates an ethereal, washy texture.

Run: python examples/29_reverb_wash.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import normalize_peak, concat, mono_to_stereo
from fractal.effects import Reverb, StereoWidth
from fractal.envelopes import ADSR, SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/29_reverb_wash.wav")

FREQS = [261.63, 329.63, 392.00, 329.63, 261.63, 220.00]
NOTE_DUR = 0.6

adsr = ADSR(attack=0.01, decay=0.1, sustain=0.4, release=0.3)
notes = []
for freq in FREQS:
    note = sine(freq, NOTE_DUR, amplitude=0.5)
    note = adsr.apply(note)
    notes.append(note)

melody = concat(*notes)

# 100% wet reverb with long decay
washed = Reverb(decay=0.7, mix=1.0, room_size="large").process(melody)
washed = SmoothFade(fade_in=0.5, fade_out=1.0).apply(washed)
washed = normalize_peak(washed, target_db=-3.0)
washed = mono_to_stereo(washed)
washed = StereoWidth(width=1.5).process(washed)

export_wav(washed, OUTPUT)
print(f"[OK] {OUTPUT} -- 6-note melody, 100%% wet reverb wash, large room")
