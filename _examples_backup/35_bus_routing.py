"""Example 35 -- Bus routing: group tracks into buses.

Two buses: 'drone' and 'texture'. Each bus has its own volume.
Both sum into the master bus.

Run: python examples/35_bus_routing.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise
from fractal.envelopes import SmoothFade, Swell
from fractal.mixer import Session
from fractal.effects import NormalizePeak

OUTPUT = Path("outputs/audio/35_bus_routing.wav")

DURATION = 12.0

session = Session(master_effects=[NormalizePeak(target_db=-1.0)])

# Drone bus
session.add_bus("drone", volume_db=-3.0)
sub = sine(55.0, DURATION, amplitude=0.3)
sub = Swell(rise_time=4.0).apply(sub)
sub = SmoothFade(fade_out=3.0).apply(sub)
session.add_track("sub", sub, bus="drone")

mid = sine(110.0, DURATION, amplitude=0.25)
mid = SmoothFade(fade_in=2.0, fade_out=3.0).apply(mid)
session.add_track("mid", mid, bus="drone", volume_db=-3.0)

# Texture bus
session.add_bus("texture", volume_db=-9.0)
noise = pink_noise(DURATION, amplitude=0.2)
noise = SmoothFade(fade_in=5.0, fade_out=4.0).apply(noise)
session.add_track("noise", noise, bus="texture")

shimmer = sine(880.0, DURATION, amplitude=0.08)
shimmer = SmoothFade(fade_in=6.0, fade_out=4.0).apply(shimmer)
session.add_track("shimmer", shimmer, bus="texture", pan=0.5)

session.export(OUTPUT)
print(f"[OK] {OUTPUT} -- 2 buses (drone, texture), 4 tracks")
