# CLAUDE.md — audiomancer

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply unless overridden here.

---

## Summary

Minimal Python audio scripts for ambient sound design.
Collection of disposable scripts for generating binaural beats, drones, pads,
and processing field recordings. Not a DAW, not a framework.

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
│   ├── synth.py             # Waveforms, drones, pads
│   ├── binaural.py          # Binaural beats + layered variants
│   ├── effects.py           # Scipy filters + pedalboard wrappers
│   ├── layers.py            # Mixing, layering, crossfade
│   └── utils.py             # I/O, normalize, fade, signal helpers
├── scripts/
│   ├── make_binaural.py     # Generate binaural beat
│   ├── drone_pad.py         # Create ambient drone pad
│   ├── process_field.py     # Process field recordings
│   └── layer_stems.py       # Layer multiple audio stems
├── tests/
│   ├── test_synth.py        # Waveform + drone + pad tests (13 tests)
│   ├── test_binaural.py     # Binaural + constants tests (8 tests)
│   ├── test_effects.py      # Filter tests (3 tests)
│   ├── test_layers.py       # Mix + layer + crossfade tests (8 tests)
│   └── test_utils.py        # Signal helpers + I/O tests (12 tests)
├── _fractal_backup/         # Full backup of original Fractal audio code
└── output/                  # Generated audio (gitignored)
```

---

## Commands

```bash
python -m pytest tests/ -v                          # run all tests (44)
python scripts/make_binaural.py                     # binaural beat
python scripts/drone_pad.py                         # ambient drone
python scripts/process_field.py input.wav           # process recording
python scripts/layer_stems.py a.wav b.wav -o out.wav  # layer stems
```

---

## What NOT to Do

- Never add DAW features (use REAPER for that)
- Never add MIDI support (use REAPER + Vital for that)
- Never add AI audio generation (no Suno, no AudioCraft)
- Never commit audio files (.wav, .flac) — gitignored
- Never over-engineer — scripts > framework
