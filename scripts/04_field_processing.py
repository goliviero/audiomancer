"""Process a field recording: clean, gate, reverb, normalize.

Usage:
    python scripts/04_field_processing.py samples/forest.wav
    python scripts/04_field_processing.py rain.wav --reverb 0.5 -o output/rain_clean.wav
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiomancer.utils import load_audio, export_wav
from audiomancer.field import process_field

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a field recording")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    parser.add_argument("--reverb", type=float, default=0.3, help="Reverb wet level (0-1)")
    args = parser.parse_args()

    signal, sr = load_audio(args.input)
    print(f"Loaded: {args.input.name} ({len(signal)/sr:.1f}s, {sr} Hz)")

    signal = process_field(signal, sample_rate=sr, reverb_wet=args.reverb)

    output = args.output or Path("output") / f"{args.input.stem}_processed.wav"
    export_wav(signal, output, sample_rate=sr)
    print(f"Done: {output}")
