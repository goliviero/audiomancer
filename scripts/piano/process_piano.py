"""Process a raw piano .wav into an ambient stem using Audiomancer.

Thin CLI wrapper around audiomancer.piano_presets (library module).

3 presets:
    bass_drone    piano -> deep sub drone (LP 500 + cathedral + slow compression + loop)
    mid_pad       piano -> warm chord pad (LP 3k + subtle chorus + hall reverb + loop)
    sparse_notes  piano -> long-decay notes (no LP + shimmer-style reverb + soft compression)

Usage:
    python scripts/piano/process_piano.py \\
        --input raw/pad.wav --preset mid_pad --output stems/pad_mid.wav
    python scripts/piano/process_piano.py \\
        --input raw/long.wav --preset sparse_notes --output stems/sparkle.wav
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.layers import normalize_lufs
from audiomancer.piano_presets import PRESETS
from audiomancer.utils import export_wav, load_audio


SR = 48000  # Output sample rate (auto-resamples if source differs)


def main():
    parser = argparse.ArgumentParser(
        description="Process a piano .wav into an ambient stem."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input piano .wav")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output stem .wav")
    parser.add_argument("--preset", required=True, choices=list(PRESETS),
                        help="Processing preset")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Target duration in seconds (default 60). "
                             "Ignored by sparse_notes preset.")
    parser.add_argument("--lufs", type=float, default=None,
                        help="Target integrated LUFS (default: preset-specific)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[!] Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    preset_fn, preset_default_lufs = PRESETS[args.preset]
    target_lufs = args.lufs if args.lufs is not None else preset_default_lufs

    print(f"[*] Loading {args.input} -> target SR {SR} Hz")
    sig, sr = load_audio(args.input, target_sr=SR)
    print(f"[*] Loaded {sig.shape[0] / sr:.1f}s ({sig.ndim}ch)")

    print(f"[*] Applying preset: {args.preset} (target {args.duration:.0f}s, "
          f"{target_lufs:+.1f} LUFS)")
    stem = preset_fn(sig, args.duration, sr)

    print(f"[*] Normalizing to {target_lufs:+.1f} LUFS")
    stem = normalize_lufs(stem, target_lufs=target_lufs, sample_rate=sr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_wav(stem, args.output, sample_rate=sr)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    dur_out = stem.shape[0] / sr
    print(f"[*] -> {args.output}  ({dur_out:.1f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
