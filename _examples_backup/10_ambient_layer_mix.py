"""Example 10 — Ambient layer mix: the closest thing to a session so far.

Layers 4 elements to create a short ambient pad:
  1. Sub drone — 55 Hz sine (warm low end, not subsonic)
  2. Mid drone — 110 Hz + 165 Hz sines (harmonic stack, the body of the mix)
  3. Pink noise bed — texture and air
  4. Theta binaural — 6 Hz beat for deep relaxation

This is the Fractal v0 "session": no envelopes yet, so it starts abruptly.
Phase 2 (envelopes) will fix that. Even so, this already sounds usable.

Run: python examples/10_ambient_layer_mix.py
Out: outputs/audio/10_ambient_layer_mix.wav
"""

from pathlib import Path

from fractal.generators import sine, pink_noise, binaural
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.export import export_wav, export_flac
from fractal.constants import SAMPLE_RATE

OUTPUT_WAV  = Path("outputs/audio/10_ambient_layer_mix.wav")
OUTPUT_FLAC = Path("outputs/audio/10_ambient_layer_mix.flac")

DURATION = 30.0

# Layer 1 — sub drone (55 Hz — audible low end, not subsonic rumble)
sub = sine(55.0, DURATION, amplitude=0.15)
sub_stereo = mono_to_stereo(sub)

# Layer 2 — mid drones: the body of the mix (lower amplitudes to soften)
mid_a = sine(110.0, DURATION, amplitude=0.2)
mid_b = sine(165.0, DURATION, amplitude=0.1)
mid   = mix_signals([mid_a, mid_b], volumes_db=[0.0, -6.0])
mid_stereo = mono_to_stereo(mid)

# Layer 3 — pink noise bed (pushed up — noise is the smoothest element)
noise = pink_noise(DURATION, amplitude=0.4)
noise_stereo = mono_to_stereo(noise)

# Layer 4 — theta binaural (6 Hz, very soft — felt not heard)
theta = binaural(carrier_hz=200.0, beat_hz=6.0, duration_sec=DURATION, amplitude=0.08)

# Mix: noise leads for softness, sines support, binaural is subliminal
result = mix_signals(
    [sub_stereo, mid_stereo, noise_stereo, theta],
    volumes_db=[-12.0, -6.0, -6.0, -27.0],
)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT_WAV, sample_rate=SAMPLE_RATE)
export_flac(result, OUTPUT_FLAC, sample_rate=SAMPLE_RATE)

print(f"[OK] {OUTPUT_WAV}")
print(f"[OK] {OUTPUT_FLAC}")
print("     30s ambient mix: sub + mid harmonics + pink noise + theta binaural")
print("     Use with headphones for binaural effect!")
