"""Process a field recording: normalize, filter, add reverb."""

import argparse
from pathlib import Path

from audiomancer.utils import load_audio, normalize, fade_in, fade_out, export_wav
from audiomancer.effects import highpass, reverb

parser = argparse.ArgumentParser(description="Process a field recording")
parser.add_argument("input", type=Path, help="Input audio file")
parser.add_argument("-o", "--output", type=Path, default=None, help="Output path")
parser.add_argument("--reverb", type=float, default=0.3, help="Reverb wet level (0-1)")
args = parser.parse_args()

signal, sr = load_audio(args.input)
signal = highpass(signal, cutoff_hz=80, sample_rate=sr)  # Remove rumble
signal = reverb(signal, room_size=0.6, wet_level=args.reverb, sample_rate=sr)
signal = fade_in(signal, 2.0, sample_rate=sr)
signal = fade_out(signal, 3.0, sample_rate=sr)
signal = normalize(signal, target_db=-1.0)

output = args.output or Path("output") / f"{args.input.stem}_processed.wav"
export_wav(signal, output, sample_rate=sr)
print(f"Done: {output}")
