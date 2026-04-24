"""V005 Stem 4 — Binaural 40 Hz Gamma, carrier 264 Hz (C4).

Left: 264 Hz, Right: 304 Hz. Pristine stereo separation.
Loop seal (5s crossfade) to avoid click at loop junction.
Mastered -14 LUFS, 48 kHz WAV.

Usage:
    python scripts/24_v005_binaural.py
    python scripts/24_v005_binaural.py --preview
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

import audiomancer.quick as q
from audiomancer.compose import make_loopable, verify_loop
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.utils import export_wav

SR = 48000
OUT = project_root / "output" / "V005"


def build_stem(duration: int = 300, seed: int = 42) -> np.ndarray:
    """Build the binaural 40 Hz Gamma stem.

    Args:
        duration: Duration in seconds.
        seed: Random seed. Varies carrier +-2 cents; beat stays exact at 40 Hz.
    """
    rng = np.random.default_rng(seed)

    # Random carrier micro-variation +-2 cents (beat stays exactly 40 Hz)
    carrier_cents = rng.uniform(-2.0, 2.0)
    carrier_hz = 264.0 * 2 ** (carrier_cents / 1200)

    stereo = q.binaural_custom(carrier_hz=carrier_hz, beat_hz=40.0,
                               duration_sec=duration, volume_db=-6.0,
                               sample_rate=SR)

    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V005 Stem 4 - Binaural 40 Hz Gamma."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 15s preview")
    parser.add_argument("--vary", action="store_true",
                        help="Random seed (carrier micro-varies +-2 cents)")
    args = parser.parse_args()

    duration = 15 if args.preview else 300
    seed = int(np.random.default_rng().integers(0, 100000)) if args.vary else 42
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 15s" if args.preview else "5 min"
    print(f"=== V005 Binaural 40 Hz Gamma - {mode} ===")
    print(f"  Carrier: 264 Hz (C4 JI)")
    print(f"  Beat: 40 Hz Gamma (L=264, R=304)")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print(f"  Seed: {seed}{' (random)' if args.vary else ' (deterministic)'}")
    print()

    stem = build_stem(duration, seed=seed)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V005_binaural_40hz{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
