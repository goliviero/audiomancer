"""V005 mix + granulated piano layer — example of sample bank integration.

Same pipeline as 25_v005_mix_preview.py but adds a 5th layer: piano C3
granulated into a drifting pad-like texture.

Requires samples/own/piano_C3_mezzo.wav (user's P-45 recording).
If absent, prints a clear message and exits cleanly.

Usage:
    python scripts/27_v005_with_piano.py
    python scripts/27_v005_with_piano.py --vary
"""

import argparse
import hashlib
import sys
from importlib import import_module
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np

from audiomancer.compose import density_profile
from audiomancer.field import clean
from audiomancer.layers import mix, normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.stochastic import micro_events
from audiomancer.synth import granular
from audiomancer.utils import (
    export_wav,
    load_audio,
    load_sample,
    mono_to_stereo,
    normalize,
)

SR = 48000
OUT = project_root / "output" / "V005"
INBOX = project_root / "inbox"
FIRECRACK_PATH = INBOX / "501417__visionear__aachen_burning-fireplace-crackling-fire-sounds.wav"
FIRECRACK_OFFSET = 60.0

DUR = 120
CHORD = [264.0, 297.0, 396.0]

SAMPLE_NAME = "piano_C3_mezzo"


def derived_seed(root: int, role: str) -> int:
    h = int(hashlib.md5(role.encode()).hexdigest()[:8], 16)
    return (root + h) % (2**31)


def main():
    parser = argparse.ArgumentParser(
        description="V005 mix with granulated piano layer (P-45)."
    )
    parser.add_argument("--vary", action="store_true",
                        help="Random root seed")
    args = parser.parse_args()

    root_seed = (int(np.random.default_rng().integers(0, 100000))
                 if args.vary else 42)

    # --- Pre-check: sample available? ---
    try:
        piano_stereo = load_sample(SAMPLE_NAME, target_sr=SR)
    except FileNotFoundError as e:
        print(f"[SKIP] Sample not available yet: {e}")
        print()
        print("To enable this script, record a C3 piano note and save to:")
        print(f"  samples/own/{SAMPLE_NAME}.wav")
        print()
        print("Tip: 10-20 seconds of sustained C3 gives enough material for")
        print("granular scattering over 2-5 minutes.")
        return

    print(f"=== V005 + Piano ({SAMPLE_NAME}) - {DUR}s ===")
    print(f"  Root seed: {root_seed}{' (random)' if args.vary else ''}")
    print()

    OUT.mkdir(parents=True, exist_ok=True)

    # --- Stems (same as 25) ---
    print("  [1/5] Warm pad C sus2...")
    pad_mod = import_module("21_v005_warm_pad")
    pad = pad_mod.build_stem(DUR, seed=derived_seed(root_seed, "pad"))

    print("  [2/5] Grounding bass...")
    bass_mod = import_module("22_v005_grounding_bass")
    bass = bass_mod.build_stem(DUR, seed=derived_seed(root_seed, "bass"))

    print("  [3/5] Binaural 40 Hz...")
    bin_mod = import_module("24_v005_binaural")
    binaural = bin_mod.build_stem(DUR, seed=derived_seed(root_seed, "binaural"))

    print(f"  [4/5] Firecrack...")
    fire_sig, fire_sr = load_audio(FIRECRACK_PATH, target_sr=SR)
    start = int(FIRECRACK_OFFSET * SR)
    end = start + int(DUR * SR)
    firecrack = clean(fire_sig[start:end], sample_rate=SR)
    firecrack = normalize(firecrack, target_db=-1.0)
    fire_mono = fire_sig.mean(axis=1) if fire_sig.ndim == 2 else fire_sig

    # --- [5] Piano granulated layer ---
    print(f"  [5/5] Piano granulated ({SAMPLE_NAME})...")
    # Use L channel as mono source (both should be very similar for single-note piano)
    piano_mono = piano_stereo[:, 0]
    piano_cloud_mono = granular(
        piano_mono, DUR,
        grain_size_ms=150, grain_density=4.0,
        pitch_spread=0.08, position_spread=1.0,
        amplitude=0.5,
        seed=derived_seed(root_seed, "piano_granular"),
        sample_rate=SR,
    )
    piano_layer = mono_to_stereo(piano_cloud_mono)

    # --- Micro-events (Phase D4) ---
    print("  [+] Micro-events (harmonic_bloom + grain_burst)...")
    events_layer = micro_events(
        DUR,
        event_specs=[
            {"type": "harmonic_bloom", "rate_per_min": 1.0,
             "volume_db": -24.0, "duration_range": (3.0, 8.0)},
            {"type": "grain_burst", "rate_per_min": 0.3,
             "volume_db": -28.0, "duration_range": (1.0, 3.0)},
        ],
        chord_freqs=CHORD,
        source=fire_mono,
        seed=derived_seed(root_seed, "micro_events"),
        sample_rate=SR,
    )

    # --- Length match ---
    min_len = min(pad.shape[0], bass.shape[0], binaural.shape[0],
                  firecrack.shape[0], piano_layer.shape[0],
                  events_layer.shape[0])
    pad = pad[:min_len]
    bass = bass[:min_len]
    binaural = binaural[:min_len]
    firecrack = firecrack[:min_len]
    piano_layer = piano_layer[:min_len]
    events_layer = events_layer[:min_len]

    # --- Mix (pad -9, bass -3, bin -8, fire -18, piano -12, events 0) ---
    print("  Mixing (piano -12 dB sits between pad and bass)...")
    stem = mix([pad, bass, binaural, firecrack, piano_layer, events_layer],
               volumes_db=[-9.0, -3.0, -8.0, -18.0, -12.0, 0.0])

    # --- Density profile (Phase D3) ---
    profile = density_profile(
        DUR, profile="random_walk",
        seed=derived_seed(root_seed, "density"),
        sample_rate=SR,
    )
    stem = stem * profile[:min_len, np.newaxis]

    stem = normalize_lufs(stem, target_lufs=-14.0, sample_rate=SR)
    stem = master_chain(stem, sample_rate=SR)

    path = OUT / "V005_mix_with_piano_2min.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"\n  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
