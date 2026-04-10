"""V004 Stem 3 — Crystal shimmer G4 (392 Hz), sparse chime-like.

Sparse random chime hits (avg 1 every 45-60s), short attack + long decay.
Heavy reverb (room=0.95) + delay (500ms). Lowpass 4kHz.
Very quiet (-25 dB). Mastered -14 LUFS, 48 kHz WAV.

Usage:
    python scripts/18_v004_shimmer.py
    python scripts/18_v004_shimmer.py --preview
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable, verify_loop
from audiomancer.effects import delay, lowpass, reverb
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.utils import export_wav, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "V004"

ROOT = 392.0  # G4
CHIME_FREQS = [
    ROOT,
    ROOT * 1.005,
    ROOT * 1.498,
    ROOT * 2.003,
    ROOT * 2.51,
]


def build_stem(duration: int = 300) -> np.ndarray:
    """Build the crystal shimmer stem with sparse chime hits."""
    n_samples = int(duration * SR)
    rng = np.random.default_rng(42)

    output = np.zeros(n_samples)
    pos = 0.0
    hit_count = 0

    while pos < duration:
        gap = rng.uniform(45.0, 60.0)
        pos += gap
        if pos >= duration:
            break

        freq = rng.choice(CHIME_FREQS)
        chime_dur = rng.uniform(3.0, 5.0)

        # Chime: short attack + long exponential decay
        n = int(chime_dur * SR)
        t = np.linspace(0, chime_dur, n, endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        attack_n = int(0.01 * SR)
        env = np.zeros(n)
        env[:attack_n] = np.linspace(0, 1, attack_n)
        env[attack_n:] = np.exp(-(t[attack_n:] - t[attack_n]) / 1.2)
        chime = tone * env * rng.uniform(0.5, 1.0)

        start = int(pos * SR)
        end = min(start + len(chime), n_samples)
        chunk = end - start
        output[start:end] += chime[:chunk]
        hit_count += 1

    print(f"    Shimmer: {hit_count} chime hits over {duration}s")

    output = lowpass(output, cutoff_hz=4000, sample_rate=SR)
    stereo = mono_to_stereo(output)
    stereo = reverb(stereo, room_size=0.95, damping=0.3, wet_level=0.8,
                    sample_rate=SR)
    stereo = delay(stereo, delay_seconds=0.5, feedback=0.35, mix=0.3,
                   sample_rate=SR)

    # Very quiet
    gain = 10 ** (-25.0 / 20)
    stereo = stereo * gain

    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V004 Stem 3 — Crystal shimmer G4 (392 Hz), sparse."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    args = parser.parse_args()

    duration = 30 if args.preview else 300
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 30s" if args.preview else "5 min"
    print(f"=== V004 Crystal Shimmer G4 — {mode} ===")
    print(f"  Root: {ROOT} Hz (G4)")
    print(f"  Hits: sparse, avg 1 every 45-60s")
    print(f"  Reverb: heavy (room=0.95)")
    print(f"  Delay: 500ms, feedback=0.35")
    print(f"  Lowpass: 4 kHz")
    print(f"  Volume: -25 dB")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print()

    stem = build_stem(duration)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V004_shimmer{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
