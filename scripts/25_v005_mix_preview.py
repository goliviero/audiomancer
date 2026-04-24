"""V005 mix preview — assemble the 3 stems + firecrack + Phase D enrichments.

Adds on top of the 3 stems:
    - Micro-events: harmonic blooms on C sus2 + grain bursts from firecrack
    - Density profile: random_walk over the full mix (non-periodic drift)
    - Seed-root coordination: each stem gets a derived seed per role

Usage:
    python scripts/25_v005_mix_preview.py
    python scripts/25_v005_mix_preview.py --vary
    python scripts/25_v005_mix_preview.py --no-pad
"""

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
from audiomancer.utils import export_wav, load_audio, normalize

SR = 48000
OUT = project_root / "output" / "V005"
FIRECRACK_PATH = project_root / "samples" / "cc0" / "fireplace_crackling.wav"
FIRECRACK_OFFSET = 60.0

DUR = 120  # 2 min
CHORD = [264.0, 297.0, 396.0]  # C sus2 — shared with warm_pad


def derived_seed(root: int, role: str) -> int:
    """Coordinated-but-not-sync seed derivation per role."""
    h = int(hashlib.md5(role.encode()).hexdigest()[:8], 16)
    return (root + h) % (2**31)


def load_firecrack(duration: int) -> tuple[np.ndarray, np.ndarray]:
    """Load firecrack layer. Returns (stereo_section, mono_source_full)."""
    sig, sr = load_audio(FIRECRACK_PATH, target_sr=SR)  # auto-resample if needed

    start_sample = int(FIRECRACK_OFFSET * SR)
    end_sample = start_sample + int(duration * SR)
    section = sig[start_sample:end_sample]

    section = clean(section, sample_rate=SR)
    section = normalize(section, target_db=-1.0)

    # Mono buffer for grain_burst micro-events (reuse full source for variety)
    mono_source = sig.mean(axis=1) if sig.ndim == 2 else sig
    return section, mono_source


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="V005 mix preview - enriched with micro-events + density."
    )
    parser.add_argument("--no-pad", action="store_true",
                        help="Skip warm pad layer (side-by-side comparison)")
    parser.add_argument("--vary", action="store_true",
                        help="Random root seed (each render uniquely varies)")
    args = parser.parse_args()
    skip_pad = args.no_pad

    root_seed = (int(np.random.default_rng().integers(0, 100000))
                 if args.vary else 42)

    OUT.mkdir(parents=True, exist_ok=True)

    variant = "no-pad" if skip_pad else "full"
    print(f"=== V005 Mix Preview - {DUR}s (2 min) [{variant}] ===")
    layers = "bass + binaural 40Hz + firecrack" if skip_pad \
             else "warm pad + bass + binaural 40Hz + firecrack"
    print(f"  Stems: {layers}")
    print(f"  Root seed: {root_seed}{' (random)' if args.vary else ''}")
    print(f"  + Micro-events: harmonic_bloom + grain_burst")
    print(f"  + Density profile: random_walk")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print()

    if not skip_pad:
        print("  [1/4] Warm pad C sus2...")
        pad_mod = import_module("21_v005_warm_pad")
        pad = pad_mod.build_stem(DUR, seed=derived_seed(root_seed, "pad"))

    print("  [2/4] Grounding bass...")
    bass_mod = import_module("22_v005_grounding_bass")
    bass = bass_mod.build_stem(DUR, seed=derived_seed(root_seed, "bass"))

    print("  [3/4] Binaural 40 Hz...")
    bin_mod = import_module("24_v005_binaural")
    binaural = bin_mod.build_stem(DUR, seed=derived_seed(root_seed, "binaural"))

    print(f"  [4/4] Firecrack (offset {FIRECRACK_OFFSET}s)...")
    firecrack, fire_mono = load_firecrack(DUR)

    # Phase D4: scatter typed micro-events
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

    # Match lengths
    lengths = [bass.shape[0], binaural.shape[0], firecrack.shape[0],
               events_layer.shape[0]]
    if not skip_pad:
        lengths.append(pad.shape[0])
    min_len = min(lengths)
    bass = bass[:min_len]
    binaural = binaural[:min_len]
    firecrack = firecrack[:min_len]
    events_layer = events_layer[:min_len]

    if skip_pad:
        print("  Mixing (no pad)...")
        stem = mix([bass, binaural, firecrack, events_layer],
                   volumes_db=[-3.0, -8.0, -18.0, 0.0])
    else:
        pad = pad[:min_len]
        print("  Mixing (full, pad -9dB)...")
        stem = mix([pad, bass, binaural, firecrack, events_layer],
                   volumes_db=[-9.0, -3.0, -8.0, -18.0, 0.0])

    # Phase D3: long-term density envelope (non-periodic)
    print("  [+] Density profile random_walk...")
    profile = density_profile(
        DUR, profile="random_walk",
        seed=derived_seed(root_seed, "density"),
        sample_rate=SR,
    )
    stem = stem * profile[:, np.newaxis]

    stem = normalize_lufs(stem, target_lufs=-14.0, sample_rate=SR)
    stem = master_chain(stem, sample_rate=SR)

    suffix = "_no_pad" if skip_pad else ""
    path = OUT / f"V005_mix_preview_2min{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"\n  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
