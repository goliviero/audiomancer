"""Integrate field recordings (Zoom H1n etc.) into ambient stems.

Loads WAV files from an input directory, cleans them, and layers them
into a single ambient texture ready for mixing with synthesized stems.

Usage:
    python scripts/12_field_integration.py recordings/
    python scripts/12_field_integration.py recordings/ -o output/field_layer.wav
    python scripts/12_field_integration.py recordings/ --duration 300 --reverb 0.4
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.utils import load_audio, export_wav, normalize, fade_in, fade_out
from audiomancer.field import clean, noise_gate, process_field
from audiomancer.effects import reverb, lowpass, highpass
from audiomancer.layers import mix, normalize_lufs, layer_at_offset
from audiomancer.compose import make_loopable

SR = 44100


def process_recordings(input_dir: Path, duration_sec: float = 300,
                       reverb_wet: float = 0.3, loopable: bool = True,
                       target_lufs: float = -18.0) -> np.ndarray:
    """Load, clean, and layer all WAV files from a directory.

    Args:
        input_dir: Directory containing WAV recordings.
        duration_sec: Target output duration in seconds.
        reverb_wet: Reverb wet level.
        loopable: If True, apply crossfade loop seal.
        target_lufs: Target loudness in LUFS.

    Returns:
        Processed stereo signal.
    """
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    print(f"Found {len(wav_files)} recordings:")
    processed = []

    for wav_path in wav_files:
        signal, sr = load_audio(wav_path)
        dur = len(signal) / sr
        print(f"  {wav_path.name}: {dur:.1f}s, {'stereo' if signal.ndim == 2 else 'mono'}")

        # Clean: DC removal, subsonic filter, ultrasonic filter
        signal = clean(signal, sample_rate=sr)

        # Noise gate to remove quiet background
        signal = noise_gate(signal, threshold_db=-50.0, sample_rate=sr)

        # Gentle highpass to remove wind rumble
        signal = highpass(signal, cutoff_hz=60.0, sample_rate=sr)

        # Soft lowpass for warmth
        signal = lowpass(signal, cutoff_hz=12000.0, sample_rate=sr)

        # Reverb for spaciousness
        if reverb_wet > 0:
            signal = reverb(signal, room_size=0.7, wet_level=reverb_wet,
                            sample_rate=sr)

        # Normalize individual recording
        signal = normalize(signal, target_db=-6.0)
        processed.append(signal)

    # Layer all recordings with staggered offsets
    n_samples = int(SR * duration_sec)
    result = np.zeros((n_samples, 2), dtype=np.float64)

    for i, sig in enumerate(processed):
        # Stagger each recording by a fraction of the duration
        offset_sec = (i / max(len(processed), 1)) * duration_sec * 0.3
        result = layer_at_offset(result, sig, offset_sec=offset_sec, volume_db=-3.0)

    # Trim to target duration
    result = result[:n_samples]

    # Fades
    result = fade_in(result, 3.0, sample_rate=SR)
    result = fade_out(result, 5.0, sample_rate=SR)

    # LUFS normalize
    result = normalize_lufs(result, target_lufs=target_lufs)

    # Loop seal
    if loopable:
        result = make_loopable(result, crossfade_sec=5.0)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrate field recordings into an ambient layer."
    )
    parser.add_argument("input_dir", type=Path,
                        help="Directory containing WAV recordings")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/field_layer.wav)")
    parser.add_argument("--duration", type=float, default=300,
                        help="Target duration in seconds (default: 300)")
    parser.add_argument("--reverb", type=float, default=0.3,
                        help="Reverb wet level 0-1 (default: 0.3)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable loop seal")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    result = process_recordings(
        args.input_dir,
        duration_sec=args.duration,
        reverb_wet=args.reverb,
        loopable=not args.no_loop,
    )

    output = args.output or Path("output") / "field_layer.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    export_wav(result, output)
    peak_db = 20 * np.log10(np.max(np.abs(result)) + 1e-10)
    print(f"Done: {output} ({len(result)/SR:.0f}s, peak={peak_db:.1f} dBFS)")
