"""V004 Stem 2 — Mid pad G3 (196 Hz), warm sustained layer.

Breathing pad 196 Hz, 4 voices / 5 cents detune.
Breathing envelope 30s cycle. Cathedral reverb (room=0.90).
Lowpass 2kHz, subtle chorus. Mastered -14 LUFS, 48 kHz WAV.

Usage:
    python scripts/17_v004_mid_pad.py
    python scripts/17_v004_mid_pad.py --preview
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable, verify_loop
from audiomancer.effects import chorus_subtle, lowpass, reverb
from audiomancer.envelope import breathing
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import apply_amplitude_mod
from audiomancer.synth import chord_pad
from audiomancer.utils import export_wav, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "V004"

ROOT = 196.0  # G3


def build_stem(duration: int = 300) -> np.ndarray:
    """Build the G3 mid pad stem."""
    raw = chord_pad([ROOT], duration, voices=4, detune_cents=5.0,
                    amplitude=0.5, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=2000, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    stereo = chorus_subtle(stereo, sample_rate=SR)
    stereo = reverb(stereo, room_size=0.90, damping=0.4, wet_level=0.6,
                    sample_rate=SR)

    breath_env = breathing(duration, breath_rate=1.0 / 30.0, depth=0.25,
                           floor=0.75, sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, breath_env)

    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V004 Stem 2 — Mid pad G3 (196 Hz)."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    args = parser.parse_args()

    duration = 30 if args.preview else 300
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 30s" if args.preview else "5 min"
    print(f"=== V004 Mid Pad G3 — {mode} ===")
    print(f"  Root: {ROOT} Hz (G3)")
    print(f"  Voices: 4, detune: 5 cents")
    print(f"  Filter: lowpass 2 kHz")
    print(f"  Reverb: cathedral (room=0.90)")
    print(f"  Breathing: 30s cycle")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print()

    stem = build_stem(duration)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V004_mid_pad{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
