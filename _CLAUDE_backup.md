# CLAUDE.md — fractal

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply unless overridden here.

---

## Summary

Python-native audio pipeline for composing, layering, and exporting music without a DAW.
Built on numpy, scipy, and pedalboard, it lets you design sounds programmatically — from raw signal generation to full mix — entirely from the command line.
A script IS a session file. Everything must be reproducible and versionable.

---

## Tech Stack

- Python 3.11+
- numpy, scipy (signal generation, DSP)
- pedalboard (audio effects — optional, isolated import)
- soundfile (I/O)
- pytest (testing), ruff (linting)

---

## Project Architecture

```
fractal/
├── CLAUDE.md
├── pyproject.toml
├── requirements.txt
├── .claude/settings.local.json   ← ruff auto-fix hook
├── src/fractal/
│   ├── __init__.py               ← public API re-exports
│   ├── constants.py              ← SAMPLE_RATE, BIT_DEPTH, paths
│   ├── signal.py                 ← Signal utilities (mono/stereo, normalize, pad, trim, mix)
│   ├── generators.py             ← sine, square, saw, triangle, noise, binaural, load_sample
│   ├── envelopes.py              ← ADSR, fades, swell, gate, tremolo, automation curves
│   ├── effects.py                ← LPF, HPF, BPF, EQ, reverb, delay, distortion, stereo width
│   ├── track.py                  ← Track dataclass (volume, pan, effects, bus routing)
│   ├── mixer.py                  ← Bus, Session (mixdown, export)
│   ├── sequencer.py              ← Clip, Pattern, Sequencer (timeline, tempo, loops)
│   ├── music_theory.py           ← Notes, scales, chords, progressions (12-TET)
│   ├── synth.py                  ← FM, additive, wavetable, subtractive, pulse, unison
│   ├── drums.py                  ← kick, snare, hihat, clap, tom, cymbal, drum_kit
│   ├── modulation.py             ← LFO, vibrato, filter sweep, param automation
│   ├── presets.py                ← SynthPreset, 20 named presets, get_preset()
│   ├── generative.py             ← random_melody, ambient_texture, chord_progression_render
│   └── export.py                 ← WAV/FLAC export via soundfile
├── scripts/
│   └── activity_log.py           ← symlink → dotfiles
├── tests/
│   ├── test_smoke.py             ← import + constant checks (3 tests)
│   ├── test_signal.py            ← signal utility tests (19 tests)
│   ├── test_generators.py        ← generator tests (20 tests)
│   ├── test_export.py            ← export roundtrip tests (11 tests)
│   ├── test_envelopes.py         ← envelope tests (26 tests)
│   ├── test_effects.py           ← effects tests (25 tests)
│   ├── test_track.py             ← track + panning tests (14 tests)
│   ├── test_mixer.py             ← session + bus tests (18 tests)
│   ├── test_sequencer.py         ← sequencer + pattern tests (28 tests)
│   ├── test_music_theory.py      ← notes, scales, chords tests (49 tests)
│   ├── test_synth.py             ← synth tests (32 tests)
│   ├── test_drums.py            ← drum synthesis tests (23 tests)
│   ├── test_modulation.py       ← modulation tests (22 tests)
│   ├── test_presets.py          ← preset tests (15 tests)
│   └── test_generative.py      ← generative composition tests (18 tests)
├── sounds/
│   ├── raw/                      ← source audio samples
│   └── processed/                ← processed/enhanced samples
├── outputs/
│   ├── audio/                    ← rendered audio files
│   └── renders/                  ← final mixdowns
├── configs/                      ← presets, session configs
├── examples/                     ← example session scripts
└── docs/
    ├── decisions.md
    ├── SWOT.md
    └── activity_log.jsonl
```

---

## Signal Flow

```
Generator(s) → Envelope → Effects chain → Track → Mixer/Session → Export
```

Each step is a pure function: `Signal → Signal` (where Signal = np.ndarray).

---

## Implementation Phases

| Phase | Module | Status |
|-------|--------|--------|
| 1 | constants, signal, generators, export | DONE |
| 2 | envelopes (ADSR, fade, swell, gate, tremolo, automation) | DONE |
| 3 | effects (LPF, HPF, BPF, EQ, reverb, delay, distortion, stereo width) | DONE |
| 4 | track + mixer (Track, Bus, Session, pan, bus routing) | DONE |
| 5 | sequencer (Clip, Pattern, Sequencer, tempo, loops) | DONE |
| 7 | music_theory (notes, scales, chords, progressions) | DONE |
| 8 | synth (FM, additive, wavetable, subtractive, unison) | DONE |
| 9 | drums (kick, snare, hihat, clap, tom, cymbal, drum_kit) | DONE |
| 10 | modulation (LFO, vibrato, filter sweep, param automation) | DONE |
| 11 | presets (20 named synthesis recipes, SynthPreset) | DONE |
| 12 | generative (random melody, ambient_texture, chord progressions) | DONE |

---

## Commands

```bash
python -m pytest tests/ -v          # run all tests
pip install -r requirements.txt     # install dependencies
```

---

## What NOT to Do

- Never hardcode sample rates — use `constants.SAMPLE_RATE`
- Never import from DAW-specific libraries (keep it pure Python + numpy/scipy)
- Never commit generated audio files (.wav, .mp3, .flac) — gitignored
- Never mutate signals in place — return new arrays
