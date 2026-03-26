"""Process piano/guitar/instrument recordings into ambient stems.

Takes a dry instrument recording and transforms it into a lush ambient
texture using reverb, spectral processing, and looping.

Usage:
    python scripts/13_instrument_stems.py samples/piano.wav
    python scripts/13_instrument_stems.py guitar.wav --mode freeze --duration 300
    python scripts/13_instrument_stems.py keys.wav --mode wash -o output/keys_ambient.wav

Modes:
    wash    — Cathedral reverb + lowpass for infinite wash (default)
    freeze  — Spectral freeze for infinite sustain pad
    granular — Granular cloud from source material
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.utils import load_audio, export_wav, normalize, fade_in, fade_out
from audiomancer.effects import reverb_cathedral, reverb, lowpass, highpass
from audiomancer.spectral import freeze, blur
from audiomancer.synth import granular
from audiomancer.layers import normalize_lufs
from audiomancer.compose import make_loopable

SR = 44100


def process_wash(signal: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Cathedral reverb wash — infinite, warm, spacious."""
    signal = highpass(signal, cutoff_hz=80.0, sample_rate=sr)
    signal = reverb_cathedral(signal, sample_rate=sr)
    signal = lowpass(signal, cutoff_hz=6000.0, sample_rate=sr)
    # Extend via looping if source is shorter than target
    target = int(sr * duration_sec)
    if len(signal) < target:
        reps = (target // len(signal)) + 1
        signal = np.tile(signal, reps if signal.ndim == 1 else (reps, 1))
    signal = signal[:target]
    signal = blur(signal, amount=0.4, sample_rate=sr)
    return signal


def process_freeze(signal: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Spectral freeze — capture one moment, sustain forever."""
    # Find the loudest moment for the best freeze point
    if signal.ndim == 2:
        env = np.abs(signal).mean(axis=1)
    else:
        env = np.abs(signal)
    # Smooth envelope to find the peak region
    window = int(sr * 0.1)
    if window > 0 and len(env) > window:
        kernel = np.ones(window) / window
        smooth_env = np.convolve(env, kernel, mode="same")
        freeze_sample = int(np.argmax(smooth_env))
    else:
        freeze_sample = len(signal) // 4
    freeze_time = freeze_sample / sr

    frozen = freeze(signal, freeze_time=freeze_time, duration_sec=duration_sec,
                    sample_rate=sr)
    # Add gentle reverb for depth
    frozen = reverb(frozen, room_size=0.8, wet_level=0.4, sample_rate=sr)
    return frozen


def process_granular(signal: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Granular cloud — shimmering texture from source material."""
    mono = signal if signal.ndim == 1 else signal.mean(axis=1)
    cloud = granular(mono, duration_sec,
                     grain_size_ms=80.0, grain_density=10.0,
                     pitch_spread=0.3, position_spread=0.8,
                     seed=42, sample_rate=sr)
    # Reverb for spaciousness
    cloud = reverb(cloud, room_size=0.7, wet_level=0.5, sample_rate=sr)
    return cloud


MODES = {
    "wash": process_wash,
    "freeze": process_freeze,
    "granular": process_granular,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform instrument recordings into ambient stems."
    )
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("--mode", choices=MODES.keys(), default="wash",
                        help="Processing mode (default: wash)")
    parser.add_argument("--duration", type=float, default=300,
                        help="Target duration in seconds (default: 300)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable loop seal")
    args = parser.parse_args()

    signal, sr = load_audio(args.input)
    print(f"Loaded: {args.input.name} ({len(signal)/sr:.1f}s, {'stereo' if signal.ndim == 2 else 'mono'})")
    print(f"Mode: {args.mode} | Duration: {args.duration}s")

    processor = MODES[args.mode]
    result = processor(signal, sr, args.duration)

    # Fades
    result = fade_in(result, 3.0, sample_rate=sr)
    result = fade_out(result, 5.0, sample_rate=sr)

    # LUFS normalize
    result = normalize_lufs(result, target_lufs=-16.0, sample_rate=sr)

    # Loop seal
    if not args.no_loop:
        result = make_loopable(result, crossfade_sec=5.0)

    suffix = f"_{args.mode}"
    output = args.output or Path("output") / f"{args.input.stem}{suffix}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    export_wav(result, output, sample_rate=sr)
    peak_db = 20 * np.log10(np.max(np.abs(result)) + 1e-10)
    print(f"Done: {output} ({len(result)/sr:.0f}s, peak={peak_db:.1f} dBFS)")
