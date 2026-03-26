"""Quick layering demo: drone + binaural + pad → 5 minutes.

Lighter version of 06_akasha_v003.py for quick tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiomancer.synth import drone, chord_pad
from audiomancer.binaural import binaural
from audiomancer.effects import reverb_hall, lowpass
from audiomancer.layers import layer, normalize_lufs
from audiomancer.utils import fade_in, fade_out, mono_to_stereo, export_wav

if __name__ == "__main__":
    DURATION = 300  # 5 minutes

    print("Generating layers (5 min preview)...")

    # Drone
    raw_drone = drone(136.1, DURATION, amplitude=0.7)
    raw_drone = lowpass(raw_drone, cutoff_hz=2000)
    drone_stereo = mono_to_stereo(raw_drone)
    drone_stereo = reverb_hall(drone_stereo)

    # Binaural
    bin_signal = binaural(200.0, 4.0, DURATION, amplitude=0.4)

    # Pad
    raw_pad = chord_pad([261.63, 329.63, 392.00], DURATION, amplitude=0.5)
    pad_stereo = mono_to_stereo(raw_pad)
    pad_stereo = reverb_hall(pad_stereo)

    # Mix
    master = layer(
        stems=[drone_stereo, bin_signal, pad_stereo],
        volumes=[0.5, 0.3, 0.25],
    )
    master = fade_in(master, 5.0)
    master = fade_out(master, 10.0)
    master = normalize_lufs(master, target_lufs=-14.0)

    export_wav(master, "output/akasha_preview_5min.wav")
    print("Done: output/akasha_preview_5min.wav")
