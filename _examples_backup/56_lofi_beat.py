"""Example 56 — Lo-fi Beat.

A lo-fi hip-hop style beat using the lo-fi drum kit with swing feel.
Kick on 1 and 3, snare on 2 and 4, hats on every 8th note with
slight swing timing.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.drums import drum_kit
from fractal.signal import pad_to_length
from fractal.effects import LowPassFilter, NormalizePeak, EffectChain
from fractal.export import export_wav

# Lo-fi kit
kit = drum_kit("lo-fi", amplitude=0.5)

bpm = 75
beat_sec = 60.0 / bpm
eighth_sec = beat_sec / 2
beat_samples = int(SAMPLE_RATE * beat_sec)
eighth_samples = int(SAMPLE_RATE * eighth_sec)

# Swing: offset every other 8th note by ~60% (instead of 50%)
swing = 0.6
swing_offset = int(SAMPLE_RATE * beat_sec * swing)
straight_offset = int(SAMPLE_RATE * beat_sec * (1.0 - swing))

# Build 4 bars
total_beats = 16
total_samples = int(SAMPLE_RATE * beat_sec * total_beats)
mix = np.zeros(total_samples, dtype=np.float64)


def place(signal, position_samples):
    """Place a signal at a position in the mix."""
    end = min(position_samples + len(signal), total_samples)
    length = end - position_samples
    if length > 0 and position_samples >= 0:
        mix[position_samples:end] += signal[:length]


# Place drums
for bar in range(4):
    bar_start = int(bar * 4 * beat_sec * SAMPLE_RATE)

    for beat in range(4):
        beat_pos = bar_start + int(beat * beat_sec * SAMPLE_RATE)

        # Kick on beats 1 and 3
        if beat in (0, 2):
            place(kit["kick"], beat_pos)

        # Snare on beats 2 and 4
        if beat in (1, 3):
            place(kit["snare"], beat_pos)

        # Hi-hat on every 8th note (with swing on off-beats)
        place(kit["hihat_closed"], beat_pos)
        # Swung 8th note
        offbeat_pos = beat_pos + swing_offset
        place(kit["hihat_closed"], offbeat_pos)

# Lo-fi processing: gentle low-pass + normalize
fx = EffectChain([
    LowPassFilter(cutoff_hz=6000),
    NormalizePeak(target_db=-3.0),
])
mix = fx.process(mix, SAMPLE_RATE)

export_wav(mix, "outputs/audio/56_lofi_beat.wav")
print("Exported: outputs/audio/56_lofi_beat.wav")
