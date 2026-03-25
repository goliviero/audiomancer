"""Example 30 — Full ambient session v3: Phase 1 + 2 + 3 combined.

The ultimate demo so far. Every layer gets generation, envelopes, AND effects.

Layers:
  1. Sub drone (55Hz) — HPF to remove DC, EQ warmth, smooth swell
  2. Mid pad (110+165Hz) — reverb for depth, EQ presence, swell
  3. Noise bed — EQ sculpted, stereo widened, smooth fades
  4. Binaural theta (6Hz) — gentle, subliminal
  5. Delayed shimmer — sine with echo + stereo width

Master chain: HPF + EQ + Normalize

Run: python examples/30_ambient_session_v3.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise, binaural
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.effects import (
    EffectChain, HighPassFilter, EQ, Reverb, StereoWidth,
    Delay, NormalizePeak,
)
from fractal.envelopes import Swell, SmoothFade, ExponentialFade, Tremolo
from fractal.export import export_wav, export_flac

OUTPUT_WAV  = Path("outputs/audio/30_ambient_session_v3.wav")
OUTPUT_FLAC = Path("outputs/audio/30_ambient_session_v3.flac")

DURATION = 60.0

# --- Layer 1: Sub drone ---
sub = sine(55.0, DURATION, amplitude=0.15)
sub = ExponentialFade(fade_in=8.0, fade_out=10.0, steepness=4.0).apply(sub)
sub = HighPassFilter(cutoff_hz=30).process(sub)
sub = EQ(bands=[{"type": "low_shelf", "freq": 60, "gain_db": 1.5}]).process(sub)
sub = mono_to_stereo(sub)

# --- Layer 2: Mid pad with reverb (softer harmonics) ---
mid_a = sine(110.0, DURATION, amplitude=0.2)
mid_b = sine(165.0, DURATION, amplitude=0.1)
mid = mix_signals([mid_a, mid_b], volumes_db=[0.0, -6.0])
mid = Swell(rise_time=12.0, curve_type="cosine").apply(mid)
mid = SmoothFade(fade_out=12.0).apply(mid)
mid = EQ(bands=[
    {"type": "peak", "freq": 2500, "gain_db": 1.0},
    {"type": "high_shelf", "freq": 6000, "gain_db": -4.0},
]).process(mid)
mid = Reverb(decay=0.35, mix=0.3, room_size="large").process(mid)
mid = mono_to_stereo(mid)

# --- Layer 3: Sculpted noise bed (prominent — smoothest element) ---
noise = pink_noise(DURATION, amplitude=0.35)
noise = SmoothFade(fade_in=6.0, fade_out=8.0).apply(noise)
noise = EQ(bands=[
    {"type": "low_shelf", "freq": 80, "gain_db": 2.0},
    {"type": "high_shelf", "freq": 5000, "gain_db": -5.0},
]).process(noise)
noise = mono_to_stereo(noise)
noise = StereoWidth(width=1.4).process(noise)

# --- Layer 4: Theta binaural (very subtle) ---
theta = binaural(carrier_hz=200.0, beat_hz=6.0, duration_sec=DURATION, amplitude=0.06)
theta = Swell(rise_time=15.0, curve_type="cosine").apply(theta)
theta = SmoothFade(fade_out=12.0).apply(theta)

# --- Layer 5: Delayed shimmer (barely perceptible) ---
shimmer = sine(660.0, DURATION, amplitude=0.03)
shimmer = Tremolo(rate=0.15, depth=0.4, shape="sine").apply(shimmer)
shimmer = SmoothFade(fade_in=10.0, fade_out=10.0).apply(shimmer)
shimmer = Delay(delay_ms=400, feedback=0.3, mix=0.4).process(shimmer)
shimmer = mono_to_stereo(shimmer)
shimmer = StereoWidth(width=1.8).process(shimmer)

# --- Final mix: noise-led, sines recessed ---
result = mix_signals(
    [sub, mid, noise, theta, shimmer],
    volumes_db=[-15.0, -9.0, -6.0, -27.0, -21.0],
)

# Master chain
master = EffectChain([
    HighPassFilter(cutoff_hz=35),
    EQ(bands=[
        {"type": "peak", "freq": 150, "gain_db": 1.0},
        {"type": "high_shelf", "freq": 8000, "gain_db": -3.0},
    ]),
    NormalizePeak(target_db=-3.0),
])
result = master.process(result)

export_wav(result, OUTPUT_WAV)
export_flac(result, OUTPUT_FLAC)

print(f"[OK] {OUTPUT_WAV}")
print(f"[OK] {OUTPUT_FLAC}")
print("     60s ambient session v3 -- 5 layers, envelopes + effects + master chain")
print("     Use with headphones!")
