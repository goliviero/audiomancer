"""V005 full mix 5 min — warm pad + arpege + binaural + fire + enrichments.

5-min production render using:
  - Warm pad C sus2 (multi_lfo + jitter)
  - Arpege bass C sus2 palindrome (NOT the pendulum — arpege variant)
  - Binaural 40 Hz Gamma
  - Firecrack loop from inbox (clean + normalize)
  - Micro-events: harmonic_bloom + grain_burst
  - Density profile: random_walk (evolution over 5 min)
  - Seed-root coordination across all stems

Mix levels (expert baseline):
    pad -6 dB, bass -9 dB, binaural -14 dB, firecrack -18 dB

Usage:
    python scripts/29_v005_mix_full_5min.py
    python scripts/29_v005_mix_full_5min.py --vary
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

from audiomancer.compose import density_profile, make_loopable, verify_loop
from audiomancer.field import clean
from audiomancer.layers import mix, normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.stochastic import micro_events
from audiomancer.utils import export_wav, load_audio, normalize

SR = 48000
OUT = project_root / "output" / "V005"
INBOX = project_root / "inbox"
FIRECRACK_PATH = INBOX / "501417__visionear__aachen_burning-fireplace-crackling-fire-sounds.wav"
FIRECRACK_OFFSET = 30.0  # more source material for 5 min

DUR = 300  # 5 min
CHORD = [264.0, 297.0, 396.0]  # C sus2


def derived_seed(root: int, role: str) -> int:
    h = int(hashlib.md5(role.encode()).hexdigest()[:8], 16)
    return (root + h) % (2**31)


def load_firecrack_loop(duration: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load firecrack and tile/loop it to cover the full duration.

    Returns:
        (stereo_section_matched_to_duration, mono_source_full)
    """
    sig, sr = load_audio(FIRECRACK_PATH, target_sr=SR)
    if sig.ndim == 1:
        sig = np.column_stack([sig, sig])

    needed = int(duration * SR)
    start = int(FIRECRACK_OFFSET * SR)
    available = len(sig) - start
    rng = np.random.default_rng(seed)

    if available >= needed:
        section = sig[start:start + needed]
    else:
        # Tile with a small randomized offset each loop to avoid strict repetition
        pieces = []
        remaining = needed
        while remaining > 0:
            off = int(rng.uniform(0, 30.0) * SR) + start
            off = min(off, len(sig) - 1)
            chunk_len = min(remaining, len(sig) - off)
            pieces.append(sig[off:off + chunk_len])
            remaining -= chunk_len
        section = np.concatenate(pieces, axis=0)[:needed]

    section = clean(section, sample_rate=SR)
    section = normalize(section, target_db=-1.0)
    mono_full = sig.mean(axis=1) if sig.ndim == 2 else sig
    return section, mono_full


def main():
    parser = argparse.ArgumentParser(
        description="V005 full 5-min mix with arpege + evolution."
    )
    parser.add_argument("--vary", action="store_true",
                        help="Random root seed (unique each render)")
    args = parser.parse_args()

    root_seed = (int(np.random.default_rng().integers(0, 100000))
                 if args.vary else 42)

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== V005 Full Mix - 5 min (with arpege + evolution) ===")
    print(f"  Root seed: {root_seed}{' (random)' if args.vary else ''}")
    print(f"  Layers: warm pad + ARPEGE bass + binaural 40Hz + firecrack")
    print(f"  Enrichments: micro_events + density_profile random_walk")
    print(f"  Target: -14 LUFS, 48 kHz WAV, loopable")
    print()

    print("  [1/5] Warm pad C sus2 (multi_lfo + jitter)...")
    pad_mod = import_module("21_v005_warm_pad")
    pad = pad_mod.build_stem(DUR, seed=derived_seed(root_seed, "pad"))

    print("  [2/5] ARPEGE bass C sus2 palindrome...")
    bass_mod = import_module("28_v005_arpege_bass")
    bass = bass_mod.build_stem(DUR, seed=derived_seed(root_seed, "bass"))

    print("  [3/5] Binaural 40 Hz Gamma...")
    bin_mod = import_module("24_v005_binaural")
    binaural = bin_mod.build_stem(DUR, seed=derived_seed(root_seed, "binaural"))

    print("  [4/5] Firecrack (tiled if needed for 5 min)...")
    firecrack, fire_mono = load_firecrack_loop(
        DUR, seed=derived_seed(root_seed, "fire"))

    print("  [+] Micro-events (harmonic_bloom 1/min + grain_burst 0.3/min)...")
    events_layer = micro_events(
        DUR,
        event_specs=[
            {"type": "harmonic_bloom", "rate_per_min": 1.0,
             "volume_db": -24.0, "duration_range": (3.0, 8.0)},
            {"type": "grain_burst", "rate_per_min": 0.3,
             "volume_db": -28.0, "duration_range": (1.0, 3.0)},
            {"type": "overtone_whisper", "rate_per_min": 0.5,
             "volume_db": -30.0, "duration_range": (2.0, 5.0)},
        ],
        chord_freqs=CHORD,
        source=fire_mono,
        seed=derived_seed(root_seed, "micro_events"),
        sample_rate=SR,
    )

    # Match lengths
    min_len = min(pad.shape[0], bass.shape[0], binaural.shape[0],
                  firecrack.shape[0], events_layer.shape[0])
    pad = pad[:min_len]
    bass = bass[:min_len]
    binaural = binaural[:min_len]
    firecrack = firecrack[:min_len]
    events_layer = events_layer[:min_len]

    # Mix: expert baseline (pad anchor, bass support, binaural audible, fire subtle)
    print("  Mixing (pad -6, bass -9, binaural -14, fire -18)...")
    stem = mix(
        [pad, bass, binaural, firecrack, events_layer],
        volumes_db=[-6.0, -9.0, -14.0, -18.0, 0.0],
    )

    # Density profile random_walk — the 5-min evolution
    print("  [+] Density profile random_walk (5-min evolution)...")
    profile = density_profile(
        DUR, profile="random_walk",
        seed=derived_seed(root_seed, "density"),
        sample_rate=SR,
    )
    stem = stem * profile[:min_len, np.newaxis]

    # LUFS + master + loop seal
    stem = normalize_lufs(stem, target_lufs=-14.0, sample_rate=SR)
    stem = master_chain(stem, sample_rate=SR)
    stem = make_loopable(stem, crossfade_sec=5.0, sample_rate=SR)

    # Loop quality check
    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    path = OUT / "V005_mix_full_5min.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"\n  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")
    print()
    print("ffmpeg loop 3h:")
    print(f"  ffmpeg -stream_loop -1 -i {path.name} -t 10800 V005_3h.wav")


if __name__ == "__main__":
    main()
