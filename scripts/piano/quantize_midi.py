"""Quantize a MIDI recording to a grid (post-record cleanup).

Snaps note/event timing to the nearest grid division (1/4, 1/8, 1/16, 1/32
of a beat). Use --strength to blend between raw performance and hard snap.

Usage:
    python scripts/piano/quantize_midi.py \\
        --input recordings/raw.mid --output recordings/clean.mid \\
        --grid 1/16 --strength 1.0

    python scripts/piano/quantize_midi.py \\
        --input raw.mid --output cleaner.mid --grid 1/8 --strength 0.6
"""

import argparse
import sys
from pathlib import Path


def _require_mido():
    try:
        import mido  # noqa: F401
    except ImportError:
        print(
            "[!] mido required. Install:\n    pip install mido\n",
            file=sys.stderr,
        )
        sys.exit(1)


GRIDS = {
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
}


def quantize_track(track, ticks_per_beat: int, grid: str, strength: float):
    """Return a new list of messages with quantized delta times."""
    import mido

    grid_ticks = int(ticks_per_beat * GRIDS[grid])
    if grid_ticks <= 0:
        return list(track)

    # Delta -> absolute, snap, strength blend, then back to delta
    abs_time = 0
    events = []
    for msg in track:
        abs_time += msg.time
        snapped = round(abs_time / grid_ticks) * grid_ticks
        new_abs = int(round(abs_time + (snapped - abs_time) * strength))
        events.append((new_abs, msg))

    # Stable sort to preserve order on ties (avoids reordering note_off/note_on)
    events.sort(key=lambda x: x[0])

    new_track = mido.MidiTrack()
    last = 0
    for abs_t, msg in events:
        delta = max(0, abs_t - last)
        new_track.append(msg.copy(time=delta))
        last = abs_t
    return new_track


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a .mid file to a time grid."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input .mid file")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .mid file")
    parser.add_argument("--grid", choices=list(GRIDS), default="1/16",
                        help="Quantization grid (default 1/16)")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Blend 0.0 (no change) to 1.0 (hard snap). "
                             "Default 1.0.")
    args = parser.parse_args()

    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength must be in [0.0, 1.0]")
    if not args.input.exists():
        print(f"[!] Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    _require_mido()
    import mido

    mid = mido.MidiFile(str(args.input))
    tpb = mid.ticks_per_beat
    print(f"[*] Loaded {args.input}  ({len(mid.tracks)} tracks, "
          f"{tpb} ticks/beat)")
    print(f"[*] Grid: {args.grid}  |  strength: {args.strength:.2f}")

    new_mid = mido.MidiFile(ticks_per_beat=tpb)
    for track in mid.tracks:
        new_mid.tracks.append(
            quantize_track(track, tpb, args.grid, args.strength)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_mid.save(str(args.output))
    print(f"[*] Saved {args.output}")


if __name__ == "__main__":
    main()
