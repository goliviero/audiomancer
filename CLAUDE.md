# CLAUDE.md — audiomancer

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply unless overridden here.

---

## Summary

Minimal Python audio scripts for ambient sound design.
Collection of disposable scripts for generating binaural beats, drones, pads,
and processing field recordings. Not a DAW, not a framework.

**Active production use**: generating audio for Akasha Portal v003.
Script `scripts/06_akasha_v003.py` produces 30 min of ambient meditation audio.

---

## Tech Stack

- Python 3.11+
- numpy, scipy (signal generation, filters)
- pedalboard (Spotify — reverb, delay, chorus, compression)
- soundfile (audio I/O)
- pytest (testing)

---

## Project Architecture

```
audiomancer/
├── CLAUDE.md
├── pyproject.toml
├── audiomancer/
│   ├── __init__.py          # SAMPLE_RATE, DEFAULT_AMPLITUDE constants
│   ├── synth.py             # Waveforms, drones, pads, noise
│   ├── binaural.py          # Binaural beats + presets
│   ├── effects.py           # Scipy filters + pedalboard wrappers
│   ├── layers.py            # Mixing, layering, crossfade, LUFS normalization
│   ├── field.py             # Field recording processing
│   └── utils.py             # I/O, normalize, fade, signal helpers
├── scripts/
│   ├── 01_binaural_432hz.py   # 10-min binaural beat
│   ├── 02_drone_pad.py        # Ambient drone pad
│   ├── 03_piano_reverb.py     # WAV + cathedral reverb
│   ├── 04_field_processing.py # Field recording cleanup
│   ├── 05_layer_akasha.py     # 5-min preview mix
│   └── 06_akasha_v003.py      # PRODUCTION: 30-min Akasha Portal audio
├── tests/                     # 70 pytest tests
├── samples/                   # Source samples (gitignored audio)
├── output/                    # Generated audio (gitignored)
└── _fractal_backup/           # Full backup of original Fractal audio code
```

---

## Commands

```bash
python -m pytest tests/ -v                          # run all tests (70)
python scripts/06_akasha_v003.py                    # PRODUCTION: Akasha v003 audio
python scripts/01_binaural_432hz.py                 # binaural beat
python scripts/02_drone_pad.py                      # ambient drone
python scripts/04_field_processing.py input.wav     # process recording
python scripts/05_layer_akasha.py                   # 5-min preview
```

---

## What NOT to Do

- Never add DAW features (use REAPER for that)
- Never add MIDI support (use REAPER + Vital for that)
- Never add AI audio generation (no Suno, no AudioCraft)
- Never commit audio files (.wav, .flac) — gitignored
- Never over-engineer — scripts > framework
