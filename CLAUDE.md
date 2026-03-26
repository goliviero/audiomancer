# CLAUDE.md — Audiomancer

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply.
> Full tree in ARCHITECTURE.md.

## Summary

Minimal Python audio toolkit for ambient/meditation sound design. Scripts > framework.
Active production: Akasha Portal stems via `scripts/10_akasha_stems.py`.

## Stack

Python 3.10+, numpy, scipy, pedalboard (Spotify), soundfile, pytest.

## Project Architecture

```
audiomancer/
├── audiomancer/          # 14 modules
│   ├── __init__.py          # SAMPLE_RATE, DEFAULT_AMPLITUDE constants
│   ├── synth.py             # Waveforms, drones, pads, noise
│   ├── binaural.py          # Binaural beats + presets
│   ├── effects.py           # Scipy filters + pedalboard wrappers
│   ├── layers.py            # Mixing, layering, crossfade, LUFS normalization
│   ├── field.py             # Field recording processing
│   ├── utils.py             # I/O, normalize, fade, signal helpers
│   ├── modulation.py        # LFOs, drift, filter sweeps
│   ├── textures.py          # 9 evolving ambient presets + registry
│   ├── compose.py           # Temporal composition, stitching, looping
│   ├── quick.py             # One-liner API (q.drone, q.pad, q.mix)
│   ├── spectral.py          # FFT: freeze, blur, pitch shift, gate, morph
│   ├── spatial.py           # Pan, auto-pan, stereo width, mid/side, Haas
│   ├── harmony.py           # Scales, tuning systems, chord generators
│   └── envelope.py          # ADSR, multi-segment, breathing, gate patterns
├── scripts/              # 15 scripts (numbered + utility)
├── tests/                # 13 test files
├── samples/              # Source samples (gitignored audio)
├── output/               # Generated audio (gitignored)
└── _fractal_backup/      # Full backup of original Fractal audio code
```

## Commands

```bash
python -m pytest tests/ -v                    # all tests
python scripts/10_akasha_stems.py             # production: 6 loopable 5-min stems
python scripts/09_progressive_stem.py         # progressive stem (5-min arc)
python scripts/08_showcase.py                 # 53 audition clips
python scripts/06_akasha_v003.py              # 30-min Om meditation
```

## What NOT to Do

- Never add DAW features (use REAPER)
- Never add MIDI support
- Never add AI audio generation (no Suno, no AudioCraft)
- Never commit audio files (.wav, .flac)
- Never over-engineer — scripts > framework
