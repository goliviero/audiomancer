"""Example 36 -- Bus effects: reverb on one bus, EQ on another.

Each bus applies its own effect chain to all tracks routed to it.

Run: python examples/36_bus_effects.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise
from fractal.envelopes import SmoothFade
from fractal.mixer import Session
from fractal.effects import Reverb, EQ, HighPassFilter, NormalizePeak

OUTPUT = Path("outputs/audio/36_bus_effects.wav")

DURATION = 10.0

session = Session(master_effects=[
    HighPassFilter(cutoff_hz=35),
    NormalizePeak(target_db=-1.0),
])

# Reverb bus — spacious, wet
session.add_bus("reverb", effects=[
    Reverb(decay=0.4, mix=0.3, room_size="large"),
])

# EQ bus — warm, sculpted
session.add_bus("warm", effects=[
    EQ(bands=[
        {"type": "low_shelf", "freq": 100, "gain_db": 3.0},
        {"type": "high_shelf", "freq": 6000, "gain_db": -2.0},
    ]),
])

fade = SmoothFade(fade_in=2.0, fade_out=2.0)

# Pad into reverb bus
pad = sine(220.0, DURATION, amplitude=0.3)
pad = fade.apply(pad)
session.add_track("pad", pad, bus="reverb", volume_db=-6.0)

# Bass into warm bus
bass = sine(65.0, DURATION, amplitude=0.3)
bass = fade.apply(bass)
session.add_track("bass", bass, bus="warm", volume_db=-6.0)

# Noise into reverb bus
noise = pink_noise(DURATION, amplitude=0.1)
noise = SmoothFade(fade_in=3.0, fade_out=3.0).apply(noise)
session.add_track("noise", noise, bus="reverb", volume_db=-15.0)

session.export(OUTPUT)
print(f"[OK] {OUTPUT} -- bus effects: reverb bus (pad+noise), warm bus (bass)")
