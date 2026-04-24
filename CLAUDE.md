# CLAUDE.md — Audiomancer

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply.
> Full architecture in ARCHITECTURE.md.

## Summary

Minimal Python audio toolkit for ambient/meditation sound design. Scripts > framework.
15 lib modules (including `builders.py` for config-driven stems) + textures package.
`configs/` (V005, V006+) + `scripts/` (generic + archived per-video) + `scripts/piano/` (MIDI workflow).
Active production: Akasha Portal stems (throat chakra V004 done, focus gamma V005 done, V006+ uses configs).

## Stack

Python 3.10+, numpy, scipy, pedalboard (Spotify), soundfile, pytest.
Optional: mido + python-rtmidi (piano capture) + FluidSynth system (MIDI->WAV).

## Commands

```bash
# Tests
python -m pytest tests/ -v

# Config-driven rendering (V006+ pattern)
python scripts/render_stem.py --config V005 --stem warm_pad       # single stem
python scripts/render_stem.py --config V005 --stem warm_pad --preview
python scripts/render_mix.py --config V005                        # full mix
python scripts/render_mix.py --config V005 --vary                 # random seed

# Archived per-video production scripts (V004, V005)
python scripts/10_akasha_stems.py             # V003 multi-stem production
python scripts/29_v005_mix_full_5min.py       # V005 final mix

# Piano workflow (record -> render -> process)
python scripts/piano/record_piano.py --output recordings/pad.mid
python scripts/piano/render_midi.py --midi recordings/pad.mid --soundfont assets/soundfonts/piano.sf2 --output raw/pad.wav
python scripts/piano/process_piano.py --input raw/pad.wav --preset mid_pad --output stems/pad_mid.wav
```

## What NOT to Do

- Never add DAW features (use REAPER for mastering if needed)
- Never add AI audio generation (no Suno, no AudioCraft, no MusicGen)
- Never commit audio files (.wav, .flac, .mp3, .sf2)
- Never over-engineer — scripts remain disposable, lib provides primitives

## What IS allowed (post-V005)

- MIDI capture from hardware keyboards via `scripts/piano/` (record to .mid only, no
  realtime synthesis, no DAW). Offline SoundFont rendering via FluidSynth subprocess.
- Config-driven rendering via `configs/<name>.py` + generic `render_stem.py`/`render_mix.py`.
  Per-video script forks are deprecated for new videos — use configs instead.

## Naming conventions (scripts)

- `scripts/render_*.py` — generic, config-driven (keep forever)
- `scripts/NN_vXXX_*.py` — archived per-video production (frozen post-release, no edits)
- `scripts/piano/*.py` — piano capture + rendering chain
