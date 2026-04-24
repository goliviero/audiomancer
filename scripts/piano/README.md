# Piano Workflow

Capture MIDI from a USB-MIDI keyboard (e.g. Yamaha P45), render offline to WAV
via a SoundFont, then process into loopable ambient stems. Pure CLI, no DAW,
no realtime synthesis.

## 3-command chain

```bash
# 1. Record (Ctrl+C to stop). Auto-detects Yamaha / P45 among MIDI ports.
python scripts/piano/record_piano.py --output recordings/pad.mid

# 2. Render MIDI -> WAV via FluidSynth + SoundFont (offline, non-realtime)
python scripts/piano/render_midi.py \
    --midi recordings/pad.mid \
    --soundfont assets/soundfonts/piano.sf2 \
    --output raw/pad.wav

# 3. Audiomancer chain: pick a preset
python scripts/piano/process_piano.py \
    --input raw/pad.wav --preset mid_pad \
    --output stems/pad_mid.wav --duration 60
```

## Concrete example — record a 30s pad, render, process

```bash
# Record a slow chord progression (30-60s is enough — process_piano loops it)
python scripts/piano/record_piano.py --output recordings/cmaj_pad.mid
# (play C-E-G held for ~30s, then Ctrl+C)

# Render with a felt piano SoundFont
python scripts/piano/render_midi.py \
    --midi recordings/cmaj_pad.mid \
    --soundfont assets/soundfonts/salamander_felt.sf2 \
    --output raw/cmaj_pad.wav

# Turn into a 60s warm mid pad stem (loopable)
python scripts/piano/process_piano.py \
    --input raw/cmaj_pad.wav \
    --preset mid_pad \
    --output stems/cmaj_pad_mid.wav \
    --duration 60
```

## Presets

| Preset | Use case | Chain |
|---|---|---|
| `bass_drone` | Piano -> deep sub drone | LP 500 Hz, cathedral reverb, slow compression (3:1, 50/500ms), 3s crossfade loop, -18 LUFS |
| `mid_pad` | Piano -> warm chord pad | LP 3 kHz, subtle chorus, hall reverb, gentle compression (2:1, 20/300ms), 2s crossfade loop, -18 LUFS |
| `sparse_notes` | Piano -> notes with long decay | No LP (keep piano identity), long delay + cathedral (shimmer-like), very gentle compression (1.5:1), no forced loop, 3s/5s fade in/out, -22 LUFS |

## Dependencies

### Python (optional — only for piano workflow)

```bash
pip install mido python-rtmidi
```

- `mido` — high-level MIDI file I/O + port handling
- `python-rtmidi` — realtime MIDI backend (USB keyboard capture)

### System (non-Python)

FluidSynth must be installed on the system. The script calls it as a subprocess.

- **macOS:** `brew install fluidsynth`
- **Windows:** `winget install FluidSynth.FluidSynth`
- **Linux:** `apt install fluidsynth`  (or `dnf` / `pacman` equivalent)

### SoundFonts (manual download)

SoundFont files (`.sf2`) are 100-400 MB each, NOT versioned. Drop them into
`assets/soundfonts/` (gitignored) manually.

Recommended CC0 / free SoundFonts:
- **Salamander Grand Piano** (CC0) — https://freepats.zenvoid.org/Piano/
- **Salamander Felt** — same source, muted felt variant
- Piano Book community — https://piano.book

## Notes / limits

- Single MIDI port / channel — no multi-keyboard or multi-channel routing.
- No realtime monitoring during capture — just record, then render.
- No MIDI quantization / editing post-recording — that would be a separate script.
- Scripts are idempotent: re-running overwrites the output cleanly.
