"""Process a raw piano .wav into an ambient stem using Audiomancer.

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

from audiomancer.compose import make_loopable
from audiomancer.effects import (
    chorus_subtle,
    compress,
    delay,
    lowpass,
    reverb,
    reverb_cathedral,
    reverb_hall,
)
from audiomancer.layers import normalize_lufs
from audiomancer.utils import (
    export_wav,
    fade_in,
    fade_out,
    load_audio,
    mono_to_stereo,
)


SR = 48000  # Output sample rate (auto-resamples if source differs)


def _ensure_stereo(sig: np.ndarray) -> np.ndarray:
    return mono_to_stereo(sig) if sig.ndim == 1 else sig


def _pad_or_trim_to_duration(sig: np.ndarray, duration: float,
                             sample_rate: int) -> np.ndarray:
    """Extend (loop-tile) or trim signal to match target duration."""
    target_n = int(duration * sample_rate)
    n = sig.shape[0]
    if n == target_n:
        return sig
    if n > target_n:
        return sig[:target_n]
    # Tile to cover target duration
    repeats = (target_n + n - 1) // n
    if sig.ndim == 2:
        tiled = np.tile(sig, (repeats, 1))
    else:
        tiled = np.tile(sig, repeats)
    return tiled[:target_n]


def preset_bass_drone(sig: np.ndarray, duration: float,
                      sample_rate: int) -> np.ndarray:
    """piano -> deep sub drone: LP 500 + cathedral + slow compression + loop."""
    # LP 500 Hz kills hammer attacks + keeps sub frequencies
    sig = lowpass(sig, cutoff_hz=500, sample_rate=sample_rate)
    sig = _ensure_stereo(sig)
    # Cathedral reverb — very long tail
    sig = reverb_cathedral(sig, sample_rate=sample_rate)
    # Slow compression: ratio 3:1, attack 50ms, release 500ms
    sig = compress(sig, threshold_db=-18.0, ratio=3.0,
                   attack_ms=50.0, release_ms=500.0, sample_rate=sample_rate)
    # Fit duration, then loop-seal with 3s crossfade
    sig = _pad_or_trim_to_duration(sig, duration, sample_rate)
    sig = make_loopable(sig, crossfade_sec=3.0, sample_rate=sample_rate)
    # Soft 2s fade in/out on the resulting block
    sig = fade_in(sig, 2.0, sample_rate=sample_rate)
    sig = fade_out(sig, 2.0, sample_rate=sample_rate)
    return sig


def preset_mid_pad(sig: np.ndarray, duration: float,
                   sample_rate: int) -> np.ndarray:
    """piano -> warm pad: LP 3k + subtle chorus + hall reverb + soft compression + loop."""
    sig = lowpass(sig, cutoff_hz=3000, sample_rate=sample_rate)
    sig = _ensure_stereo(sig)
    # Subtle chorus for body
    sig = chorus_subtle(sig, sample_rate=sample_rate)
    # Hall reverb 6-8s character
    sig = reverb_hall(sig, sample_rate=sample_rate)
    # Gentle compression: ratio 2:1, attack 20ms, release 300ms
    sig = compress(sig, threshold_db=-20.0, ratio=2.0,
                   attack_ms=20.0, release_ms=300.0, sample_rate=sample_rate)
    sig = _pad_or_trim_to_duration(sig, duration, sample_rate)
    sig = make_loopable(sig, crossfade_sec=2.0, sample_rate=sample_rate)
    sig = fade_in(sig, 1.5, sample_rate=sample_rate)
    sig = fade_out(sig, 1.5, sample_rate=sample_rate)
    return sig


def preset_sparse_notes(sig: np.ndarray, duration: float,
                        sample_rate: int) -> np.ndarray:
    """piano -> sparse notes with ultra-long decay.

    No LP (keep piano identity). Shimmer-style reverb via cathedral + long delay.
    Very gentle compression. No forced loopable — keeps source duration.
    """
    sig = _ensure_stereo(sig)
    # "Shimmer" approximation: long delay (1.2s feedback 0.4) + cathedral reverb
    # This gives the "supermassive with shimmer tail" feel without a dedicated module
    sig = delay(sig, delay_seconds=1.2, feedback=0.4, mix=0.35,
                sample_rate=sample_rate)
    sig = reverb_cathedral(sig, sample_rate=sample_rate)
    # Very gentle compression: ratio 1.5:1
    sig = compress(sig, threshold_db=-24.0, ratio=1.5,
                   attack_ms=50.0, release_ms=1000.0, sample_rate=sample_rate)
    # Sparse notes: do NOT force make_loopable — the natural decay is the content
    # Fade 3s in, 5s out
    sig = fade_in(sig, 3.0, sample_rate=sample_rate)
    sig = fade_out(sig, 5.0, sample_rate=sample_rate)
    return sig


PRESETS = {
    "bass_drone": (preset_bass_drone, -18.0),
    "mid_pad": (preset_mid_pad, -18.0),
    "sparse_notes": (preset_sparse_notes, -22.0),
}


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
    import numpy as _np
    peak_db = 20 * _np.log10(_np.max(_np.abs(stem)) + 1e-10)
    dur_out = stem.shape[0] / sr
    print(f"[*] -> {args.output}  ({dur_out:.1f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
