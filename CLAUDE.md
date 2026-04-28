# CLAUDE.md — Audiomancer

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply.
> Full architecture in ARCHITECTURE.md.

## Summary

Minimal Python audio toolkit for ambient/meditation sound design. Scripts > framework.
24 lib modules (incl. `builders.py` for config-driven stems) + textures package.
`configs/` (V005-V011) + `scripts/` (generic renderers + archived per-video) + `scripts/piano/` (MIDI workflow).
Active production: Akasha Portal stems. V004 (throat) / V005 (focus gamma) / V006 (Muladhara + 6 variants) / V007 (brown noise sleep) — livrés.
V008 (Svadhisthana) / V009 (rain) / V010 (pink noise) / V011 (Manipura) — WIP.

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
- `scripts/_archive/<vXXX>/` — exploration scripts superseded after release (V007, gallery v2/v3)
- `scripts/piano/*.py` — piano capture + rendering chain

## Frozen configs (do not edit)

Production-livré, scripts d'exploration archivés dans `scripts/_archive/<vXXX>/` :
- V005 (focus gamma) — scripts archivés in-place via convention `NN_vXXX_*.py`
- V006 + 6 variants (Muladhara : braise / cavern / ember / ember_v2 / tidal / warmth)
- V007 (brown noise 10h) — exploration dans `scripts/_archive/v007/`

WIP éditables : V008 / V009 / V010 / V011 (samples CC0 à sourcer pour V008/V009/V011).
