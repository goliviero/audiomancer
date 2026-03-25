"""Example 37 -- Session one-liner export: build and export in minimal code.

Shows how concise a Fractal session can be — from signal to .wav in a few lines.

Run: python examples/37_session_export.py
"""

from pathlib import Path

from fractal.generators import sine, binaural
from fractal.envelopes import SmoothFade
from fractal.mixer import Session
from fractal.effects import NormalizePeak

OUTPUT_WAV = Path("outputs/audio/37_session_export.wav")
OUTPUT_FLAC = Path("outputs/audio/37_session_export.flac")

session = Session(master_effects=[NormalizePeak(target_db=-1.0)])

fade = SmoothFade(fade_in=3.0, fade_out=3.0)

session.add_track("drone", fade.apply(sine(110.0, 15.0, amplitude=0.4)), volume_db=-3.0)
session.add_track("binaural", fade.apply(binaural(200.0, 6.0, 15.0, amplitude=0.1)), volume_db=-18.0)

session.export(OUTPUT_WAV)
session.export(OUTPUT_FLAC)

print(f"[OK] {OUTPUT_WAV}")
print(f"[OK] {OUTPUT_FLAC}")
print("     Session.export() -- 2 tracks, WAV + FLAC in 5 lines")
