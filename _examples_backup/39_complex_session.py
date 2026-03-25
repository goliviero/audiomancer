"""Example 39 -- Complex session: 6 tracks, 2 buses, full routing.

A rich ambient session with sub, pad, fifth, noise, shimmer, and binaural.
Shows the full Track+Mixer workflow with bus routing and master chain.

Run: python examples/39_complex_session.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise, binaural
from fractal.envelopes import SmoothFade, Swell, ExponentialFade
from fractal.mixer import Session
from fractal.effects import (
    HighPassFilter, EQ, Reverb, StereoWidth, NormalizePeak,
)

OUTPUT = Path("outputs/audio/39_complex_session.wav")

DURATION = 30.0

session = Session(master_effects=[
    HighPassFilter(cutoff_hz=35),
    EQ(bands=[{"type": "peak", "freq": 150, "gain_db": 1.0}]),
    NormalizePeak(target_db=-1.0),
])

# --- Buses ---
session.add_bus("drone", volume_db=-3.0)
session.add_bus("atmosphere", volume_db=-6.0, effects=[
    Reverb(decay=0.35, mix=0.25, room_size="large"),
])

# --- Drone bus ---
sub = sine(55.0, DURATION, amplitude=0.2)
sub = ExponentialFade(fade_in=6.0, fade_out=8.0, steepness=3.0).apply(sub)
session.add_track("sub", sub, bus="drone", volume_db=-9.0)

mid = sine(110.0, DURATION, amplitude=0.3)
mid = Swell(rise_time=8.0, curve_type="cosine").apply(mid)
mid = SmoothFade(fade_out=6.0).apply(mid)
session.add_track("mid", mid, bus="drone", volume_db=-3.0)

fifth = sine(165.0, DURATION, amplitude=0.2)
fifth = SmoothFade(fade_in=4.0, fade_out=6.0).apply(fifth)
session.add_track("fifth", fifth, bus="drone", volume_db=-6.0)

# --- Atmosphere bus ---
noise = pink_noise(DURATION, amplitude=0.2)
noise = SmoothFade(fade_in=5.0, fade_out=5.0).apply(noise)
session.add_track("noise", noise, bus="atmosphere", volume_db=-12.0)

shimmer = sine(660.0, DURATION, amplitude=0.06)
shimmer = SmoothFade(fade_in=8.0, fade_out=8.0).apply(shimmer)
session.add_track("shimmer", shimmer, bus="atmosphere", volume_db=-15.0, pan=0.6,
                  effects=[StereoWidth(width=1.6)])

# --- Master bus ---
theta = binaural(200.0, 6.0, DURATION, amplitude=0.1)
theta = Swell(rise_time=10.0, curve_type="cosine").apply(theta)
theta = SmoothFade(fade_out=8.0).apply(theta)
session.add_track("binaural", theta, volume_db=-24.0)

session.export(OUTPUT)
print(f"[OK] {OUTPUT} -- complex session: 6 tracks, 2 buses, 30s ambient")
