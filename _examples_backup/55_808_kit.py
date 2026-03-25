"""Example 55 — 808 Drum Kit Showcase.

Generate every piece from the 808 kit and export each individually,
then play a simple kick-snare-hat pattern.
"""

from fractal.constants import SAMPLE_RATE
from fractal.drums import drum_kit, kick, snare, hihat
from fractal.signal import silence, concat
from fractal.effects import NormalizePeak
from fractal.export import export_wav

# Generate the full 808 kit
kit = drum_kit("808", amplitude=0.5)

# Export each piece
for name, sig in kit.items():
    export_wav(sig, f"outputs/audio/55_808_{name}.wav")
    print(f"  Exported: 55_808_{name}.wav")

# Build a simple 4-beat pattern: kick-hat-snare-hat
bpm = 90
beat_sec = 60.0 / bpm
gap = silence(beat_sec)

# Pad each hit to one beat length
from fractal.signal import pad_to_length
beat_samples = int(SAMPLE_RATE * beat_sec)

k = pad_to_length(kit["kick"], beat_samples)
s = pad_to_length(kit["snare"], beat_samples)
hh = pad_to_length(kit["hihat_closed"], beat_samples)

# One bar: kick - hat - snare - hat
bar = concat(k, hh, s, hh)

# Loop 4 bars
pattern = concat(bar, bar, bar, bar)

# Normalize
fx = NormalizePeak(target_db=-3.0)
pattern = fx.process(pattern, SAMPLE_RATE)

export_wav(pattern, "outputs/audio/55_808_pattern.wav")
print("Exported: outputs/audio/55_808_pattern.wav")
