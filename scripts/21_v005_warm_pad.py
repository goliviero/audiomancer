"""V005 Stem 1 — Warm pad (validated: D2 alive deep, C2+C3+E3+G4).

Production choice after A/B comparison: D2 alive moderate with deep voicing.
  - Chord: C2 (66) + C3 (132) + E3 (165) + G4 (396) — major open, no G5 strident
  - 4 voices, 3 detuned sines each, per-note jitter +-1 cent
  - Per-voice amplitude modulation at different rates -> voices take turns
  - LP filter drift 1000-4000 Hz (random_walk, non-periodic)
  - Auto-pan 0.04 Hz, chorus, reverb room=0.80

Delegates to build_alive_pad() from 23_v005_pad_alive.py.

Usage:
    python scripts/21_v005_warm_pad.py
    python scripts/21_v005_warm_pad.py --preview
    python scripts/21_v005_warm_pad.py --vary
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np

from audiomancer.compose import verify_loop
from audiomancer.utils import export_wav

SR = 48000
OUT = project_root / "output" / "V005"

# Deep chord (validated): C major open, no G5 stridence
CHORD_DEEP = [66.0, 132.0, 165.0, 396.0]  # C2 + C3 + E3 + G4


def build_stem(duration: int = 300, seed: int = 42) -> np.ndarray:
    """Build the warm pad using validated D2 alive deep approach.

    Args:
        duration: Duration in seconds.
        seed: Random seed.
    """
    alive_mod = import_module("23_v005_pad_alive")
    return alive_mod.build_alive_pad(
        intensity="moderate",
        seed=seed,
        duration=duration,
        chord=CHORD_DEEP,
    )


def main():
    parser = argparse.ArgumentParser(
        description="V005 Stem 1 - Warm pad (D2 alive deep, validated)."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 15s preview")
    parser.add_argument("--vary", action="store_true",
                        help="Random seed")
    args = parser.parse_args()

    duration = 15 if args.preview else 300
    seed = int(np.random.default_rng().integers(0, 100000)) if args.vary else 42
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 15s" if args.preview else "5 min"
    print(f"=== V005 Warm Pad (D2 alive deep) - {mode} ===")
    print(f"  Chord: C2 + C3 + E3 + G4 = {CHORD_DEEP} Hz")
    print(f"  Intensity: moderate (filter drift, voice rotation, auto-pan)")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print(f"  Seed: {seed}{' (random)' if args.vary else ' (deterministic)'}")
    print()

    stem = build_stem(duration, seed=seed)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V005_warm_pad{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
