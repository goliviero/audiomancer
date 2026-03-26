"""Load a WAV file and apply massive reverb — great for piano/keys textures.

Usage:
    python scripts/03_piano_reverb.py samples/piano_note.wav
    python scripts/03_piano_reverb.py input.wav -o output/piano_washed.wav
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiomancer.utils import load_audio, fade_in, fade_out, normalize, export_wav
from audiomancer.effects import reverb_cathedral, lowpass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply massive reverb to audio")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    args = parser.parse_args()

    signal, sr = load_audio(args.input)
    print(f"Loaded: {args.input.name} ({len(signal)/sr:.1f}s, {sr} Hz)")

    # Cathedral reverb for that infinite wash
    signal = reverb_cathedral(signal, sample_rate=sr)
    signal = lowpass(signal, cutoff_hz=6000, sample_rate=sr)  # Soften the highs
    signal = fade_in(signal, 1.0, sample_rate=sr)
    signal = fade_out(signal, 5.0, sample_rate=sr)
    signal = normalize(signal, target_db=-1.0)

    output = args.output or Path("output") / f"{args.input.stem}_reverb.wav"
    export_wav(signal, output, sample_rate=sr)
    print(f"Done: {output}")
