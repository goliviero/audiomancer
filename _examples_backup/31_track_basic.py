"""Example 31 -- Basic track: one track with volume and panning.

Demonstrates the Track class: a sine wave rendered with volume
adjustment and stereo panning.

Run: python examples/31_track_basic.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.track import Track
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/31_track_basic.wav")

sig = sine(440.0, 3.0, amplitude=0.5)

track = Track(name="lead", signal=sig, volume_db=-3.0, pan=0.3)
rendered = track.render()
rendered = normalize_peak(rendered, target_db=-1.0)

export_wav(rendered, OUTPUT)
print(f"[OK] {OUTPUT} -- single track, -3dB, pan=0.3 right")
