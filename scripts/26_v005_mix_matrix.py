"""V005 mix matrix — 10 variants with different pad/bass/binaural balance.

Builds the 3 stems + firecrack ONCE, then renders 10 mix variants
by varying volumes. Applies full mastering chain to each.

Variant sweep:
    V01-V04 : pad-forward spectrum (pad louder than bass)
    V05-V07 : bass-forward spectrum (bass louder than pad)
    V08-V10 : binaural sweep (subliminal to forward) at expert baseline

Usage:
    python scripts/26_v005_mix_matrix.py
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
INBOX = project_root / "inbox"
FIRECRACK_PATH = INBOX / "501417__visionear__aachen_burning-fireplace-crackling-fire-sounds.wav"
FIRECRACK_OFFSET = 60.0

DUR = 120  # 2 min
CHORD = [264.0, 297.0, 396.0]  # C sus2 — shared with warm_pad
ROOT_SEED = 42  # deterministic across matrix runs


def derived_seed(root: int, role: str) -> int:
    h = int(hashlib.md5(role.encode()).hexdigest()[:8], 16)
    return (root + h) % (2**31)

# Mix matrix: (label, pad_db, bass_db, binaural_db, firecrack_db)
# Hierarchy: pad > bass > binaural > firecrack is the expert recommendation
VARIANTS = [
    # Pad-forward spectrum (pad dominant, bass in support)
    ("v01_pad_aggressive",  -3.0,  -12.0, -14.0, -20.0),  # pad very front
    ("v02_pad_expert",      -6.0,  -12.0, -14.0, -20.0),  # expert baseline
    ("v03_pad_lead_bass_close", -6.0,  -9.0,  -14.0, -20.0),
    ("v04_balanced",        -9.0,  -9.0,  -14.0, -20.0),  # pad=bass

    # Bass-forward spectrum (bass dominant — current default style)
    ("v05_bass_slight",     -9.0,  -6.0,  -14.0, -20.0),
    ("v06_bass_lead",       -12.0, -6.0,  -14.0, -20.0),
    ("v07_bass_aggressive", -12.0, -3.0,  -14.0, -20.0),  # like current

    # Binaural sweep (pad -6, bass -12 baseline, vary binaural)
    ("v08_bin_forward",     -6.0,  -12.0, -8.0,  -20.0),  # may fatigue on 3h
    ("v09_bin_balanced",    -6.0,  -12.0, -14.0, -20.0),  # same as v02 (expert)
    ("v10_bin_subliminal",  -6.0,  -12.0, -20.0, -20.0),
]


def load_firecrack(duration: int) -> tuple[np.ndarray, np.ndarray]:
    """Load firecrack section and prep. Returns (stereo_section, mono_full)."""
    sig, sr = load_audio(FIRECRACK_PATH, target_sr=SR)
    start = int(FIRECRACK_OFFSET * SR)
    end = start + int(duration * SR)
    section = sig[start:end]
    section = clean(section, sample_rate=SR)
    section = normalize(section, target_db=-1.0)
    mono_source = sig.mean(axis=1) if sig.ndim == 2 else sig
    return section, mono_source


def apply_mix(pad: np.ndarray, bass: np.ndarray, binaural: np.ndarray,
              firecrack: np.ndarray, events: np.ndarray,
              profile: np.ndarray,
              volumes_db: list[float]) -> np.ndarray:
    """Mix layers + enrichment + density, then LUFS + master."""
    stem = mix([pad, bass, binaural, firecrack, events],
               volumes_db=volumes_db + [0.0])
    stem = stem * profile[:, np.newaxis]
    stem = normalize_lufs(stem, target_lufs=-14.0, sample_rate=SR)
    stem = master_chain(stem, sample_rate=SR)
    return stem


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== V005 Mix Matrix - {DUR}s x {len(VARIANTS)} variants ===")
    print(f"  Target: -14 LUFS, -1 dBTP ceiling, 48 kHz WAV")
    print()

    # --- Build stems ONCE ---
    print("--- Building stems (one-time) ---")
    print("  [1/4] Warm pad C sus2 (Phase D multi_lfo + jitter)...")
    pad_mod = import_module("21_v005_warm_pad")
    pad = pad_mod.build_stem(DUR, seed=derived_seed(ROOT_SEED, "pad"))

    print("  [2/4] Grounding bass (note jitter + random_walk)...")
    bass_mod = import_module("22_v005_grounding_bass")
    bass = bass_mod.build_stem(DUR, seed=derived_seed(ROOT_SEED, "bass"))

    print("  [3/4] Binaural 40 Hz...")
    bin_mod = import_module("24_v005_binaural")
    binaural = bin_mod.build_stem(DUR, seed=derived_seed(ROOT_SEED, "binaural"))

    print(f"  [4/4] Firecrack (offset {FIRECRACK_OFFSET}s)...")
    firecrack, fire_mono = load_firecrack(DUR)

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
        seed=derived_seed(ROOT_SEED, "micro_events"),
        sample_rate=SR,
    )

    print("  [+] Density profile random_walk...")
    profile = density_profile(
        DUR, profile="random_walk",
        seed=derived_seed(ROOT_SEED, "density"),
        sample_rate=SR,
    )

    # Match lengths
    min_len = min(pad.shape[0], bass.shape[0], binaural.shape[0],
                  firecrack.shape[0], events_layer.shape[0], profile.shape[0])
    pad = pad[:min_len]
    bass = bass[:min_len]
    binaural = binaural[:min_len]
    firecrack = firecrack[:min_len]
    events_layer = events_layer[:min_len]
    profile = profile[:min_len]
    print(f"  All stems: {min_len / SR:.0f}s")
    print()

    # --- Render each variant ---
    print("--- Rendering mix variants ---")
    for label, p_db, b_db, bi_db, f_db in VARIANTS:
        stem = apply_mix(pad, bass, binaural, firecrack, events_layer, profile,
                         [p_db, b_db, bi_db, f_db])
        path = OUT / f"V005_mix_{label}.wav"
        export_wav(stem, path, sample_rate=SR)
        peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
        print(f"  {label:32s}  pad={p_db:+5.1f} bass={b_db:+5.1f} "
              f"bin={bi_db:+5.1f} fire={f_db:+5.1f}  peak={peak_db:+5.1f} dBFS")

    print()
    print(f"Done - {len(VARIANTS)} variants in {OUT}/")
    print()
    print("Listening guide:")
    print("  v02_pad_expert        = recommended baseline (pad ancre)")
    print("  v01_pad_aggressive    = pad very forward")
    print("  v07_bass_aggressive   = current style (bass leads)")
    print("  v08_bin_forward       = binaural very present (test for fatigue)")
    print("  v10_bin_subliminal    = binaural almost inaudible")


if __name__ == "__main__":
    main()
