"""Example 07 — Stereo ping-pong: alternating L/R sine pulses.

Demonstrates manual stereo manipulation using numpy directly.
Each 'ping' is 0.5s on left, each 'pong' is 0.5s on right.
Frequencies alternate between two notes.

Run: python examples/07_stereo_ping_pong.py
Out: outputs/audio/07_stereo_ping_pong.wav
"""

from pathlib import Path

import numpy as np

from fractal.generators import sine
from fractal.export import export_wav
from fractal.signal import normalize_peak
from fractal.constants import SAMPLE_RATE

OUTPUT = Path("outputs/audio/07_stereo_ping_pong.wav")

PULSE_DUR  = 0.5      # seconds per ping/pong
N_PULSES   = 8        # 4 ping-pong pairs
FREQ_PING  = 440.0    # A4 — left channel
FREQ_PONG  = 523.25   # C5 — right channel

samples_per_pulse = int(SAMPLE_RATE * PULSE_DUR)
total_samples = N_PULSES * samples_per_pulse

left  = np.zeros(total_samples)
right = np.zeros(total_samples)

for i in range(N_PULSES):
    start = i * samples_per_pulse
    end   = start + samples_per_pulse
    # Fade out each pulse to avoid clicks
    fade  = np.linspace(1.0, 0.0, samples_per_pulse)
    if i % 2 == 0:
        left[start:end]  = sine(FREQ_PING, PULSE_DUR, amplitude=0.5) * fade
    else:
        right[start:end] = sine(FREQ_PONG, PULSE_DUR, amplitude=0.5) * fade

stereo = np.column_stack([left, right])
stereo = normalize_peak(stereo, target_db=-3.0)

export_wav(stereo, OUTPUT, sample_rate=SAMPLE_RATE)
print(f"[OK] {OUTPUT} — {N_PULSES} stereo ping-pong pulses (440Hz L <-> 523Hz R)")
