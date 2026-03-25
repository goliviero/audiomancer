"""Generate a 10-minute binaural beat at 432 Hz (solfeggio + alpha)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiomancer.binaural import binaural_layered
from audiomancer.utils import fade_in, fade_out, normalize, export_wav

signal = binaural_layered(carrier_hz=432.0, beat_hz=10.0,
                          duration_sec=600, pink_amount=0.15)
signal = fade_in(signal, 5.0)
signal = fade_out(signal, 10.0)
signal = normalize(signal, target_db=-1.0)
export_wav(signal, "output/binaural_alpha_432hz_10min.wav")
print("Done: output/binaural_alpha_432hz_10min.wav")
