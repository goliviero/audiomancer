"""Record MIDI from a USB keyboard (e.g. Yamaha P45) to a standard .mid file.

Captures everything on the selected input port with delta-time timing and
saves a Type 0 MIDI file when the user hits Ctrl+C.

Usage:
    python scripts/piano/record_piano.py --output recordings/pad.mid
    python scripts/piano/record_piano.py --output my.mid --port "Yamaha P45"
    python scripts/piano/record_piano.py --output my.mid --bpm 90

Dependencies:
    pip install mido python-rtmidi
"""

import argparse
import signal
import sys
import time
from pathlib import Path

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name(midi_note: int) -> str:
    octave = midi_note // 12 - 1
    return f"{NOTE_NAMES[midi_note % 12]}{octave}"


def _require_mido():
    try:
        import mido  # noqa: F401
        import rtmidi  # noqa: F401
    except ImportError:
        print(
            "[!] mido + python-rtmidi required. Install:\n"
            "    pip install mido python-rtmidi\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _pick_port(preferred: str | None) -> str:
    import mido
    available = mido.get_input_names()
    if not available:
        print("[!] No MIDI input ports found. Is the keyboard connected + powered on?",
              file=sys.stderr)
        sys.exit(2)

    if preferred:
        # Exact match first, then substring fallback
        for name in available:
            if name == preferred or preferred.lower() in name.lower():
                return name
        print(f"[!] Port {preferred!r} not found. Available:", file=sys.stderr)
        for name in available:
            print(f"    - {name}", file=sys.stderr)
        sys.exit(3)

    # Auto-detect Yamaha / P45 / Piano in the name
    for name in available:
        n = name.lower()
        if any(k in n for k in ("yamaha", "p45", "p-45", "piano", "digital keyboard")):
            return name

    # Fallback: first port
    print(f"[!] No keyboard auto-detected. Using first port: {available[0]!r}",
          file=sys.stderr)
    return available[0]


def main():
    parser = argparse.ArgumentParser(
        description="Capture MIDI from USB keyboard to .mid file (Ctrl+C to stop)."
    )
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .mid path (will be created)")
    parser.add_argument("--port", default=None,
                        help="MIDI input port name (substring OK). Auto-detect if absent.")
    parser.add_argument("--bpm", type=int, default=120,
                        help="Tempo for ticks metadata (default 120)")
    args = parser.parse_args()

    _require_mido()
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo

    port_name = _pick_port(args.port)
    print(f"[*] Using MIDI port: {port_name}")
    print(f"[*] Tempo: {args.bpm} BPM  |  Ctrl+C to stop and save to {args.output}")
    print()

    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(args.bpm), time=0))

    tempo = bpm2tempo(args.bpm)
    note_count = 0
    stop = False

    def _on_sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        inp = mido.open_input(port_name)
    except (OSError, IOError) as e:
        print(f"[!] Failed to open port {port_name!r}: {e}", file=sys.stderr)
        sys.exit(4)

    start_time = time.monotonic()
    last_event_time = start_time
    try:
        while not stop:
            for msg in inp.iter_pending():
                if msg.is_meta:
                    continue
                now = time.monotonic()
                delta_sec = now - last_event_time
                delta_ticks = int(mido.second2tick(delta_sec, mid.ticks_per_beat, tempo))
                last_event_time = now

                # Drop channel info beyond 0 — keep it simple mono-channel
                if hasattr(msg, "channel"):
                    msg = msg.copy(time=delta_ticks)
                else:
                    msg = msg.copy(time=delta_ticks)
                track.append(msg)

                if msg.type == "note_on" and msg.velocity > 0:
                    note_count += 1
                    elapsed = now - start_time
                    name = _note_name(msg.note)
                    print(
                        f"\r  [rec] notes: {note_count:3d}  |  {elapsed:5.1f}s  "
                        f"|  last: {name:>3s} v{msg.velocity:3d}     ",
                        end="", flush=True,
                    )

            time.sleep(0.005)  # 5ms poll, cheap and responsive
    finally:
        inp.close()

    print()  # newline after the live counter
    print(f"[*] Stopped. Total notes: {note_count}")

    # Write the file — ensure parent dir exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        mid.save(str(args.output))
    except OSError as e:
        print(f"[!] Failed to save MIDI file: {e}", file=sys.stderr)
        sys.exit(5)
    duration = time.monotonic() - start_time
    print(f"[*] Saved {args.output}  ({duration:.1f}s, {note_count} notes)")


if __name__ == "__main__":
    main()
