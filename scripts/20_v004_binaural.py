"""V004 Stem 4 — Binaural 4 Hz theta, carrier G3 (196 Hz).

Pure binaural beat layer. Carrier at 196 Hz (G3), beat frequency 4 Hz (theta).
Mastered -14 LUFS, 48 kHz WAV, loopable.

Usage:
    python scripts/20_v004_binaural.py
    python scripts/20_v004_binaural.py --preview
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
OUT = project_root / "output" / "V004"


def build_stem(duration: int = 300) -> np.ndarray:
    """Build the binaural 4 Hz theta stem."""
    stereo = q.binaural_custom(carrier_hz=196.0, beat_hz=4.0,
                               duration_sec=duration, volume_db=-6.0,
                               sample_rate=SR)

    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V004 Stem 4 — Binaural 4 Hz theta."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    args = parser.parse_args()

    duration = 30 if args.preview else 300
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 30s" if args.preview else "5 min"
    print(f"=== V004 Binaural 4 Hz Theta — {mode} ===")
    print(f"  Carrier: 196 Hz (G3)")
    print(f"  Beat: 4 Hz (theta)")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print()

    stem = build_stem(duration)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V004_binaural_4hz{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
