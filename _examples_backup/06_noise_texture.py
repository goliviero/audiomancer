"""Example 06 — Noise texture: pink noise bed + sine tone layer.

Demonstrates mixing a noise generator with a tonal element.
Useful as ambient background texture.

Run: python examples/06_noise_texture.py
Out: outputs/audio/06_noise_texture.wav
"""

from pathlib import Path

from fractal.generators import pink_noise, sine
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/06_noise_texture.wav")

DURATION = 15.0

noise = pink_noise(DURATION, amplitude=0.6)
tone  = sine(80.0, DURATION, amplitude=0.3)   # sub-bass hint

# Noise is the bed (-3dB), tone is underneath (-12dB)
texture = mix_signals([noise, tone], volumes_db=[-3.0, -12.0])
texture = normalize_peak(texture, target_db=-3.0)
texture = mono_to_stereo(texture)

export_wav(texture, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — 15s pink noise + 80Hz sub tone")
