"""Example 40 -- Ambient session v4: the Phase 4 flagship.

Same concept as v3 (example 30) but using Track + Session + Bus routing.
Cleaner code, proper signal flow, bus effects, master chain.

Layers:
  1. Sub drone (55Hz) -> drone bus
  2. Mid pad (110+165Hz) -> drone bus
  3. Noise bed -> atmosphere bus (with reverb)
  4. Delayed shimmer -> atmosphere bus (with reverb + stereo width)
  5. Binaural theta (6Hz) -> master

Run: python examples/40_ambient_session_v4.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise, binaural
from fractal.envelopes import (
    SmoothFade, Swell, ExponentialFade, Tremolo,
)
from fractal.mixer import Session
from fractal.effects import (
    HighPassFilter, EQ, Reverb, StereoWidth, Delay, NormalizePeak,
)
from fractal.signal import mix_signals
from fractal.export import export_flac

OUTPUT_WAV = Path("outputs/audio/40_ambient_session_v4.wav")
OUTPUT_FLAC = Path("outputs/audio/40_ambient_session_v4.flac")

DURATION = 60.0

# --- Build session ---
session = Session(master_effects=[
    HighPassFilter(cutoff_hz=35),
    EQ(bands=[
        {"type": "peak", "freq": 150, "gain_db": 1.0},
        {"type": "high_shelf", "freq": 8000, "gain_db": -3.0},
    ]),
    NormalizePeak(target_db=-3.0),
])

# Buses
session.add_bus("drone", volume_db=0.0, effects=[
    HighPassFilter(cutoff_hz=30),
    EQ(bands=[{"type": "low_shelf", "freq": 60, "gain_db": 2.0}]),
])

session.add_bus("atmosphere", volume_db=-3.0, effects=[
    Reverb(decay=0.35, mix=0.2, room_size="large"),
])

# --- Layer 1: Sub drone (gentle) ---
sub = sine(55.0, DURATION, amplitude=0.12)
sub = ExponentialFade(fade_in=8.0, fade_out=10.0, steepness=4.0).apply(sub)
session.add_track("sub", sub, bus="drone", volume_db=-15.0)

# --- Layer 2: Mid pad (softer harmonics) ---
mid_a = sine(110.0, DURATION, amplitude=0.2)
mid_b = sine(165.0, DURATION, amplitude=0.1)
mid = mix_signals([mid_a, mid_b], volumes_db=[0.0, -6.0])
mid = Swell(rise_time=12.0, curve_type="cosine").apply(mid)
mid = SmoothFade(fade_out=12.0).apply(mid)
session.add_track("mid_pad", mid, bus="drone", volume_db=-9.0,
                  effects=[EQ(bands=[
                      {"type": "peak", "freq": 2500, "gain_db": 1.0},
                      {"type": "high_shelf", "freq": 6000, "gain_db": -4.0},
                  ])])

# --- Layer 3: Noise bed (prominent — smoothest element) ---
noise = pink_noise(DURATION, amplitude=0.35)
noise = SmoothFade(fade_in=6.0, fade_out=8.0).apply(noise)
session.add_track("noise", noise, bus="atmosphere", volume_db=-6.0,
                  effects=[
                      EQ(bands=[
                          {"type": "low_shelf", "freq": 80, "gain_db": 2.0},
                          {"type": "high_shelf", "freq": 5000, "gain_db": -5.0},
                      ]),
                      StereoWidth(width=1.4),
                  ])

# --- Layer 4: Delayed shimmer (barely perceptible) ---
shimmer = sine(660.0, DURATION, amplitude=0.03)
shimmer = Tremolo(rate=0.15, depth=0.4, shape="sine").apply(shimmer)
shimmer = SmoothFade(fade_in=10.0, fade_out=10.0).apply(shimmer)
session.add_track("shimmer", shimmer, bus="atmosphere", volume_db=-21.0, pan=0.5,
                  effects=[
                      Delay(delay_ms=400, feedback=0.3, mix=0.4),
                      StereoWidth(width=1.8),
                  ])

# --- Layer 5: Binaural theta (very subtle) ---
theta = binaural(carrier_hz=200.0, beat_hz=6.0, duration_sec=DURATION, amplitude=0.06)
theta = Swell(rise_time=15.0, curve_type="cosine").apply(theta)
theta = SmoothFade(fade_out=12.0).apply(theta)
session.add_track("binaural", theta, volume_db=-27.0)

# --- Export ---
session.export(OUTPUT_WAV)

# Also export FLAC via render + export_flac
result = session.render()
export_flac(result, OUTPUT_FLAC)

print(f"[OK] {OUTPUT_WAV}")
print(f"[OK] {OUTPUT_FLAC}")
print("     60s ambient session v4 -- 5 layers, Session+Bus routing, master chain")
print("     Use with headphones!")
