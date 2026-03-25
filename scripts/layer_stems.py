"""Layer multiple audio stems into a single mix."""

import argparse
from pathlib import Path

from audiomancer.utils import load_audio, normalize, export_wav
from audiomancer.layers import mix

parser = argparse.ArgumentParser(description="Layer audio stems")
parser.add_argument("stems", nargs="+", type=Path, help="Audio files to layer")
parser.add_argument("-o", "--output", type=Path, default=Path("output/layered_mix.wav"))
parser.add_argument("-v", "--volumes", nargs="*", type=float, default=None,
                    help="Volume per stem in dB (default: 0dB each)")
args = parser.parse_args()

signals = []
sample_rate = None
for stem in args.stems:
    sig, sr = load_audio(stem)
    signals.append(sig)
    if sample_rate is None:
        sample_rate = sr
    print(f"  Loaded: {stem.name} ({len(sig)/sr:.1f}s, {sr}Hz)")

result = mix(signals, volumes_db=args.volumes)
result = normalize(result, target_db=-1.0)
export_wav(result, args.output, sample_rate=sample_rate)
print(f"Done: {args.output}")
