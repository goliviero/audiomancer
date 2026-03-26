"""Create an ambient drone pad (55 Hz fundamental + airy pad)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiomancer.synth import drone, pad
from audiomancer.effects import reverb, lowpass
from audiomancer.layers import mix
from audiomancer.utils import fade_in, fade_out, normalize, export_wav

if __name__ == "__main__":
    # Deep drone at 55 Hz (A1)
    low_drone = drone(55.0, 60.0)
    low_drone = lowpass(low_drone, cutoff_hz=800)

    # Airy pad at 220 Hz (A3) with detuned voices
    airy_pad = pad(220.0, 60.0, voices=5, detune_cents=15)
    airy_pad = reverb(airy_pad, room_size=0.8, wet_level=0.4)
    airy_pad = lowpass(airy_pad, cutoff_hz=3000)

    # Mix and master
    signal = mix([low_drone, airy_pad], volumes_db=[0.0, -3.0])
    signal = fade_in(signal, 3.0)
    signal = fade_out(signal, 5.0)
    signal = normalize(signal, target_db=-1.0)
    export_wav(signal, "output/drone_pad_ambient.wav")
    print("Done: output/drone_pad_ambient.wav")
