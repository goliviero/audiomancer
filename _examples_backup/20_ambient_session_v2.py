"""Example 20 — Full ambient session v2: everything from Phase 1 + 2.

The same concept as example 10, but now with proper envelopes on every layer.
This is what a real Fractal session looks like with Phase 2.

Layers:
  1. Sub drone (55Hz) -- exponential swell
  2. Mid harmonics (110Hz + 165Hz) -- slow cosine swell, body of the mix
  3. Pink noise bed -- smooth fade in/out
  4. Binaural theta (6Hz) -- smooth swell + smooth fade out
  5. High shimmer (660Hz, very quiet) -- tremolo for texture

Run: python examples/20_ambient_session_v2.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise, binaural
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.envelopes import (
    Swell, SmoothFade, ExponentialFade, FadeInOut, Tremolo,
)
from fractal.export import export_wav, export_flac

OUTPUT_WAV  = Path("outputs/audio/20_ambient_session_v2.wav")
OUTPUT_FLAC = Path("outputs/audio/20_ambient_session_v2.flac")

DURATION = 45.0

# --- Layer 1: Sub drone (55 Hz) with exponential swell ---
sub = sine(55.0, DURATION, amplitude=0.15)
sub = ExponentialFade(fade_in=6.0, fade_out=8.0, steepness=4.0).apply(sub)
sub = mono_to_stereo(sub)

# --- Layer 2: Mid harmonics — body of the mix (softer) ---
mid_a = sine(110.0, DURATION, amplitude=0.2)
mid_b = sine(165.0, DURATION, amplitude=0.1)
mid = mix_signals([mid_a, mid_b], volumes_db=[0.0, -6.0])
mid = Swell(rise_time=10.0, curve_type="cosine").apply(mid)
mid = SmoothFade(fade_out=10.0).apply(mid)
mid = mono_to_stereo(mid)

# --- Layer 3: Pink noise bed — pushed up as the softest element ---
noise = pink_noise(DURATION, amplitude=0.35)
noise = SmoothFade(fade_in=4.0, fade_out=6.0).apply(noise)
noise = mono_to_stereo(noise)

# --- Layer 4: Theta binaural — subliminal (very quiet) ---
theta = binaural(carrier_hz=200.0, beat_hz=6.0, duration_sec=DURATION, amplitude=0.06)
theta = Swell(rise_time=12.0, curve_type="cosine").apply(theta)
theta = SmoothFade(fade_out=10.0).apply(theta)

# --- Layer 5: High shimmer with slow tremolo (barely audible) ---
shimmer = sine(660.0, DURATION, amplitude=0.03)
shimmer = Tremolo(rate=0.2, depth=0.5, shape="sine").apply(shimmer)
shimmer = SmoothFade(fade_in=8.0, fade_out=8.0).apply(shimmer)
shimmer = mono_to_stereo(shimmer)

# --- Final mix: noise leads for smoothness, sines support ---
result = mix_signals(
    [sub, mid, noise, theta, shimmer],
    volumes_db=[-12.0, -6.0, -6.0, -27.0, -21.0],
)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT_WAV)
export_flac(result, OUTPUT_FLAC)

print(f"[OK] {OUTPUT_WAV}")
print(f"[OK] {OUTPUT_FLAC}")
print("     45s ambient session v2 -- 5 layers, all enveloped")
print("     Compare with 10_ambient_layer_mix.wav to hear the Phase 2 difference!")
