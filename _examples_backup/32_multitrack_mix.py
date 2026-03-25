"""Example 32 -- Multi-track mix: 3 tracks at different volumes via Session.

Demonstrates Session with multiple tracks summed together.

Run: python examples/32_multitrack_mix.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise
from fractal.envelopes import SmoothFade
from fractal.mixer import Session
from fractal.effects import NormalizePeak

OUTPUT = Path("outputs/audio/32_multitrack_mix.wav")

session = Session(master_effects=[NormalizePeak(target_db=-1.0)])

# Track 1: fundamental drone
fund = sine(110.0, 10.0, amplitude=0.4)
fund = SmoothFade(fade_in=2.0, fade_out=2.0).apply(fund)
session.add_track("fundamental", fund, volume_db=-3.0)

# Track 2: fifth harmony
fifth = sine(165.0, 10.0, amplitude=0.3)
fifth = SmoothFade(fade_in=3.0, fade_out=2.0).apply(fifth)
session.add_track("fifth", fifth, volume_db=-6.0)

# Track 3: noise texture
noise = pink_noise(10.0, amplitude=0.2)
noise = SmoothFade(fade_in=4.0, fade_out=3.0).apply(noise)
session.add_track("noise", noise, volume_db=-15.0)

session.export(OUTPUT)
print(f"[OK] {OUTPUT} -- 3 tracks: fundamental (-3dB), fifth (-6dB), noise (-15dB)")
