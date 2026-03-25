"""Example 04 — Harmonic drone: fundamental + overtones.

Stack a fundamental with its natural harmonic series.
Root: 55 Hz (A1). Adds: 2f, 3f, 4f, 5f, 6f at decreasing volumes.
Creates a rich, organ-like sustained tone.

Run: python examples/04_harmonic_drone.py
Out: outputs/audio/04_harmonic_drone.wav
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/04_harmonic_drone.wav")

FUNDAMENTAL = 55.0   # A1
DURATION = 10.0

# Build harmonic series: f, 2f, 3f, 4f, 5f, 6f
harmonics = [FUNDAMENTAL * n for n in range(1, 7)]
volumes   = [0.0, -6.0, -10.0, -14.0, -18.0, -22.0]  # each overtone quieter

partials = [sine(f, DURATION, amplitude=0.5) for f in harmonics]

drone = mix_signals(partials, volumes_db=volumes)
drone = normalize_peak(drone, target_db=-3.0)
drone = mono_to_stereo(drone)   # widen to stereo for export

export_wav(drone, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — 10s harmonic drone, root={FUNDAMENTAL}Hz + 5 overtones")
