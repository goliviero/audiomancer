# CLAUDE.md — Audiomancer

> Project-specific rules. Global rules in ~/.claude/CLAUDE.md apply.
> Full architecture in ARCHITECTURE.md.

## Summary

Minimal Python audio toolkit for ambient/meditation sound design. Scripts > framework.
14 modules, 12 scripts, 13 test files. Active production: Akasha Portal stems.

## Stack

Python 3.10+, numpy, scipy, pedalboard (Spotify), soundfile, pytest.

## Commands

```bash
python -m pytest tests/ -v                    # all tests
python scripts/10_akasha_stems.py             # production: 6 loopable 5-min stems
python scripts/11_gallery.py                  # visual + audio gallery (14 PNG + 10 WAV)
python scripts/06_akasha_v003.py              # 30-min Om meditation
```

## What NOT to Do

- Never add DAW features (use REAPER)
- Never add MIDI support
- Never add AI audio generation (no Suno, no AudioCraft)
- Never commit audio files (.wav, .flac)
- Never over-engineer — scripts > framework
