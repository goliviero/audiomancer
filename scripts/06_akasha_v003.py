"""Akasha Portal v003 — Audio Production Script.

Generates ~30 minutes of ambient meditation audio. 100% original synthesis.
No AI-generated audio (no Suno, no AudioCraft).

Layers:
  1. Drone — 136.1 Hz (Om) + harmonics → cathedral reverb
  2. Binaural — theta 4 Hz on 200 Hz carrier (deep meditation)
  3. Pad — C major chord (261/329/392 Hz) with subtle chorus + hall reverb
  4. Texture — pink noise bed, very low, for warmth

Output: output/akasha_v003_master.wav (stereo, 44100 Hz, ~-14 LUFS)
        + individual stems in output/stems/
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.synth import drone, chord_pad, pink_noise
from audiomancer.binaural import binaural
from audiomancer.effects import reverb_cathedral, reverb_hall, chorus_subtle, lowpass
from audiomancer.layers import layer, normalize_lufs
from audiomancer.utils import fade_in, fade_out, mono_to_stereo, export_wav

# ---------------------------------------------------------------------------
# Production parameters
# ---------------------------------------------------------------------------
DURATION_SEC = 1800          # 30 minutes
FADE_IN_SEC = 10.0           # Gentle fade in
FADE_OUT_SEC = 15.0          # Long fade out
TARGET_LUFS = -14.0          # YouTube standard
OUTPUT_DIR = project_root / "output"
STEMS_DIR = OUTPUT_DIR / "stems"

# Frequencies
OM_FREQ = 136.1              # Om frequency
C_MAJOR = [261.63, 329.63, 392.00]  # C4, E4, G4
BINAURAL_CARRIER = 200.0     # Hz
BINAURAL_BEAT = 4.0          # Theta (deep meditation)


def generate_drone_layer() -> np.ndarray:
    """Layer 1: Om drone with harmonics + cathedral reverb."""
    print("  [1/4] Generating Om drone (136.1 Hz)...")
    # Om fundamental + warm harmonics with slow roll-off
    harmonics = [
        (1, 1.0),    # Fundamental
        (2, 0.6),    # Octave
        (3, 0.3),    # Fifth
        (4, 0.15),   # Two octaves
        (5, 0.08),   # Major third
        (6, 0.04),   # Another fifth
    ]
    raw = drone(OM_FREQ, DURATION_SEC, harmonics=harmonics, amplitude=0.7)
    # Low-pass to keep it warm
    raw = lowpass(raw, cutoff_hz=2000)
    # Convert to stereo before reverb
    stereo = mono_to_stereo(raw)
    # Cathedral reverb for massive space
    print("  [1/4] Applying cathedral reverb...")
    wet = reverb_cathedral(stereo)
    return wet


def generate_binaural_layer() -> np.ndarray:
    """Layer 2: Theta binaural beat for deep meditation."""
    print("  [2/4] Generating theta binaural (4 Hz on 200 Hz)...")
    return binaural(BINAURAL_CARRIER, BINAURAL_BEAT, DURATION_SEC, amplitude=0.4)


def generate_pad_layer() -> np.ndarray:
    """Layer 3: C major chord pad with chorus + hall reverb."""
    print("  [3/4] Generating C major pad...")
    raw = chord_pad(C_MAJOR, DURATION_SEC, voices=3, detune_cents=8.0, amplitude=0.5)
    # Convert to stereo
    stereo = mono_to_stereo(raw)
    # Subtle chorus for movement
    stereo = chorus_subtle(stereo)
    # Hall reverb for lush sustain
    print("  [3/4] Applying hall reverb...")
    stereo = reverb_hall(stereo)
    return stereo


def generate_texture_layer() -> np.ndarray:
    """Layer 4: Pink noise bed for warmth."""
    print("  [4/4] Generating pink noise texture...")
    noise_l = pink_noise(DURATION_SEC, amplitude=0.3)
    noise_r = pink_noise(DURATION_SEC, amplitude=0.3)
    stereo = np.column_stack([noise_l, noise_r])
    # Roll off highs for a warm feel
    stereo = lowpass(stereo, cutoff_hz=3000)
    return stereo


def main():
    print(f"=== Akasha Portal v003 — Audio Production ===")
    print(f"Duration: {DURATION_SEC // 60} minutes ({DURATION_SEC}s)")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print()

    # Generate each layer
    print("Generating layers...")
    drone_stem = generate_drone_layer()
    binaural_stem = generate_binaural_layer()
    pad_stem = generate_pad_layer()
    texture_stem = generate_texture_layer()
    print()

    # Export individual stems
    print("Exporting stems...")
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    export_wav(drone_stem, STEMS_DIR / "01_drone_om.wav")
    export_wav(binaural_stem, STEMS_DIR / "02_binaural_theta.wav")
    export_wav(pad_stem, STEMS_DIR / "03_pad_cmajor.wav")
    export_wav(texture_stem, STEMS_DIR / "04_texture_pink.wav")
    print(f"  Stems saved to {STEMS_DIR}/")
    print()

    # Mix all layers
    print("Mixing layers...")
    # Volume balance: drone loud, binaural medium, pad medium-low, texture quiet
    master = layer(
        stems=[drone_stem, binaural_stem, pad_stem, texture_stem],
        volumes=[0.45, 0.30, 0.25, 0.10],
    )

    # Master processing
    print("Mastering...")
    master = fade_in(master, FADE_IN_SEC)
    master = fade_out(master, FADE_OUT_SEC)
    master = normalize_lufs(master, target_lufs=TARGET_LUFS)

    # Export master
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_path = OUTPUT_DIR / "akasha_v003_master.wav"
    export_wav(master, master_path)

    # Report
    duration_min = len(master) / SAMPLE_RATE / 60
    peak_db = 20 * np.log10(np.max(np.abs(master)) + 1e-10)
    rms_db = 20 * np.log10(np.sqrt(np.mean(master ** 2)) + 1e-10)
    print()
    print(f"=== DONE ===")
    print(f"Master:   {master_path}")
    print(f"Duration: {duration_min:.1f} min")
    print(f"Peak:     {peak_db:.1f} dB")
    print(f"RMS:      {rms_db:.1f} dB (target LUFS: {TARGET_LUFS})")
    print(f"Channels: {'stereo' if master.ndim == 2 else 'mono'}")


if __name__ == "__main__":
    main()
