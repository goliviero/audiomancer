"""Generate stems for Akasha Portal V003 (111Hz Holy Frequency).

Produces loopable 5-min stems compatible with the Akasha ffmpeg pipeline.
These replace the Suno layer — 100% original synthesis.

Output: output/v003_stems/
  - drone_111hz.wav    (111Hz Om + harmonics, cathedral reverb)
  - pad_ambient.wav    (minor chord pad, hall reverb)

Both stems are designed to be looped seamlessly by ffmpeg (-stream_loop -1).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import chorus_subtle, lowpass, reverb_cathedral, reverb_hall
from audiomancer.layers import crossfade
from audiomancer.synth import chord_pad, drone
from audiomancer.utils import export_wav, mono_to_stereo, normalize

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DURATION_SEC = 300           # 5 minutes (looped by ffmpeg for 3h)
CROSSFADE_SEC = 5.0          # Crossfade for seamless loop point
OUTPUT_DIR = project_root / "output" / "v003_stems"

# V003 theme: 111 Hz Holy Frequency
HOLY_FREQ = 111.0


def make_loopable(signal: np.ndarray) -> np.ndarray:
    """Make a signal seamlessly loopable by crossfading head and tail."""
    return crossfade(signal, signal.copy(), crossfade_sec=CROSSFADE_SEC)


def generate_drone() -> np.ndarray:
    """111 Hz drone with warm harmonics + cathedral reverb."""
    print("  Generating 111 Hz holy drone...")
    harmonics = [
        (1, 1.0),    # 111 Hz fundamental
        (2, 0.5),    # 222 Hz octave
        (3, 0.25),   # 333 Hz fifth
        (4, 0.12),   # 444 Hz two octaves
        (5, 0.06),   # 555 Hz
        (6, 0.03),   # 666 Hz
    ]
    raw = drone(HOLY_FREQ, DURATION_SEC, harmonics=harmonics, amplitude=0.7)
    raw = lowpass(raw, cutoff_hz=1500)
    stereo = mono_to_stereo(raw)
    print("  Applying cathedral reverb...")
    stereo = reverb_cathedral(stereo)
    stereo = normalize(stereo, target_db=-1.0)
    return stereo


def generate_pad() -> np.ndarray:
    """Ambient pad — dark minor chord matching 111 Hz theme."""
    print("  Generating ambient pad...")
    # A minor-ish voicing rooted near 111 Hz region
    # 111 Hz (A2-ish), 132 Hz (C3), 166 Hz (E3) — A minor triad
    freqs = [111.0, 132.0, 166.0]
    raw = chord_pad(freqs, DURATION_SEC, voices=3, detune_cents=10.0, amplitude=0.5)
    stereo = mono_to_stereo(raw)
    stereo = chorus_subtle(stereo)
    print("  Applying hall reverb...")
    stereo = reverb_hall(stereo)
    stereo = lowpass(stereo, cutoff_hz=4000)
    stereo = normalize(stereo, target_db=-1.0)
    return stereo


def main():
    print("=== Akasha V003 Stem Generation ===")
    print(f"Duration: {DURATION_SEC // 60} min (loopable)")
    print()

    drone_stem = generate_drone()
    pad_stem = generate_pad()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_wav(drone_stem, OUTPUT_DIR / "drone_111hz.wav")
    print(f"  => {OUTPUT_DIR / 'drone_111hz.wav'}")

    export_wav(pad_stem, OUTPUT_DIR / "pad_ambient.wav")
    print(f"  => {OUTPUT_DIR / 'pad_ambient.wav'}")

    print()
    print("Done. Copy these to akasha-portal/sounds/processed/V003/")
    print("Then update presets/V003_111hz_holy_frequency_3h.json:")
    print('  "sounds": [')
    print('    { "file": "V003/drone_111hz.wav", "volume_db": -8 },')
    print('    { "file": "V003/pad_ambient.wav", "volume_db": -18 }')
    print("  ]")


if __name__ == "__main__":
    main()
