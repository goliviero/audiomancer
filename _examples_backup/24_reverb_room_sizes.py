"""Example 24 — Reverb room size comparison: small, medium, large.

A short piano-like tone (sine + harmonics + ADSR) through 3 room sizes.
Hear how space size changes the character of the reverb tail.

Run: python examples/24_reverb_room_sizes.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import mix_signals, normalize_peak, concat, silence, mono_to_stereo
from fractal.effects import Reverb
from fractal.envelopes import ADSR
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/24_reverb_room_sizes.wav")

# "Piano-ish" tone: fundamental + 2 harmonics with ADSR
def make_note(freq: float, dur: float) -> "np.ndarray":
    f1 = sine(freq, dur, amplitude=0.5)
    f2 = sine(freq * 2, dur, amplitude=0.2)
    f3 = sine(freq * 3, dur, amplitude=0.1)
    note = mix_signals([f1, f2, f3])
    return ADSR(attack=0.01, decay=0.2, sustain=0.3, release=0.5).apply(note)

note = make_note(440.0, 2.0)

small  = Reverb(decay=0.4, mix=0.4, room_size="small").process(note)
medium = Reverb(decay=0.4, mix=0.4, room_size="medium").process(note)
large  = Reverb(decay=0.4, mix=0.4, room_size="large").process(note)

gap = silence(1.0)
result = concat(small, gap, medium, gap, large)
result = normalize_peak(result, target_db=-3.0)
result = mono_to_stereo(result)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- reverb: small -> medium -> large room, same A440 note")
