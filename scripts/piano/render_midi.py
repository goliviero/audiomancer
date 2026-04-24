"""Render a .mid file to .wav offline via FluidSynth + SoundFont.

Uses FluidSynth as a subprocess (no Python bindings required, works cross-platform).

Usage:
    python scripts/piano/render_midi.py \\
        --midi recordings/pad.mid \\
        --soundfont assets/soundfonts/piano.sf2 \\
        --output raw/pad.wav

    python scripts/piano/render_midi.py --midi pad.mid --soundfont piano.sf2 \\
        --output raw/pad.wav --sample-rate 48000 --gain 0.8

System dependency (non-Python):
    macOS:   brew install fluidsynth
    Windows: winget install FluidSynth.FluidSynth
    Linux:   apt install fluidsynth
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _check_fluidsynth() -> str:
    """Return path to fluidsynth binary or exit with install instructions."""
    exe = shutil.which("fluidsynth")
    if exe:
        return exe
    print(
        "[!] FluidSynth not found in PATH. Install it:\n"
        "    macOS:   brew install fluidsynth\n"
        "    Windows: winget install FluidSynth.FluidSynth\n"
        "    Linux:   apt install fluidsynth   (or dnf / pacman equivalent)\n",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Render a .mid file to .wav using FluidSynth + SoundFont (offline)."
    )
    parser.add_argument("--midi", required=True, type=Path,
                        help="Input .mid file")
    parser.add_argument("--soundfont", required=True, type=Path,
                        help="SoundFont .sf2 file (piano etc.)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .wav path")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="Output sample rate (default 44100)")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Master gain (default 1.0)")
    args = parser.parse_args()

    if not args.midi.exists():
        print(f"[!] MIDI file not found: {args.midi}", file=sys.stderr)
        sys.exit(2)
    if not args.soundfont.exists():
        print(f"[!] SoundFont not found: {args.soundfont}", file=sys.stderr)
        sys.exit(3)

    fluidsynth = _check_fluidsynth()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # FluidSynth offline rendering. Flags:
    #   -ni       non-interactive
    #   -F out    write output to file (WAV)
    #   -r SR     sample rate
    #   -g GAIN   gain multiplier
    #   -O s16    16-bit PCM output (universal)
    cmd = [
        fluidsynth,
        "-ni",
        "-F", str(args.output),
        "-r", str(args.sample_rate),
        "-g", str(args.gain),
        "-O", "s16",
        str(args.soundfont),
        str(args.midi),
    ]

    print(f"[*] FluidSynth: {fluidsynth}")
    print(f"[*] MIDI: {args.midi}  SoundFont: {args.soundfont}")
    print(f"[*] Rendering -> {args.output}  ({args.sample_rate} Hz, gain={args.gain})")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        print(f"[!] Failed to run fluidsynth: {e}", file=sys.stderr)
        sys.exit(4)

    if proc.returncode != 0:
        print(f"[!] FluidSynth exited with code {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    if not args.output.exists():
        print(f"[!] FluidSynth claimed success but {args.output} does not exist",
              file=sys.stderr)
        sys.exit(5)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"[*] Done. {args.output.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
