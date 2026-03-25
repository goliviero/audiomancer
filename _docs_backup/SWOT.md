# SWOT — Fractal

> Last updated: 2026-03-25 — Phases 1-5 + 7-12 complete. Vibe coding roadmap complete.

## Strengths

- Pure Python — runs on any machine with numpy/scipy/soundfile
- Signal = np.ndarray — composes naturally, no abstraction overhead
- Fully reproducible: a script IS the session file (versionable, diffable)
- 18 modules covering the full signal chain: generators -> envelopes -> effects -> track -> mixer -> sequencer -> synth -> drums -> modulation -> presets -> generative -> export
- Session/Track/Bus architecture mirrors professional DAW workflows
- Sequencer with tempo, beat-based placement, patterns, and looping
- Equal-power panning, Schroeder reverb, FFT-based pink noise — proper DSP
- Full public API re-exported from `__init__.py` — clean import ergonomics
- Effect base class with consistent `.process(signal, sample_rate)` interface
- Music theory module: note_to_hz, 15 scales, 18 chord types, 9 progressions
- 12-TET tuning with enharmonic support (C#/Db, Cb=B, etc.)
- Synthesizers: FM, additive (8 harmonic presets), wavetable, subtractive, pulse, unison/detune
- Drum synthesis: kick, snare, hihat, clap, tom, cymbal + 5 kits (808, 909, acoustic, lo-fi, industrial)
- Modulation: LFO (6 shapes), vibrato, filter sweep, generic param automation
- 20 named presets with get_preset() partial matching (5 pads, 5 leads, 5 basses, 2 keys, 3 textures)
- Generative composition: ambient_texture(), random_melody(), chord_progression_render()
- 323 tests, 100% pass rate, ~3s runtime
- 63 runnable examples covering all 12 phases

## Weaknesses

- No real-time playback — offline render only (by design for v1)
- `pip install -e .` required manually — no installer script yet
- No undo/history in Session (write your script = your undo system)
- Long audio generation (1h+) may hit RAM limits with full in-memory arrays
- Reverb allpass filter is sample-by-sample Python loop — slow for long signals
- No resampling support in load_sample() — sample rate mismatch only warns

## Opportunities

- Absorb all Akasha Portal audio logic into reusable modules
- pedalboard gives access to studio-quality VST-like effects for free
- Could replace ffmpeg-based mixing in Akasha Portal with native Python
- Future: audiocraft / MusicGen integration for AI-assisted generation
- Could serve as the audio backend for other projects (game audio, podcasts, etc.)
- Vectorize reverb allpass with scipy.signal.lfilter for 10-50x speedup

## Threats

- pedalboard (Spotify) may change API without warning
- numpy array approach breaks if signals need metadata (tempo, key, time sig) — dataclass wrapper may be needed in v2
- Long audio generation (1h+) may hit RAM limits with full in-memory arrays

## Backlog (prioritized)

| Priority | Item | Status |
|----------|------|--------|
| P0 | Phase 6: Flagship demos + docs polish | TODO |
| P1 | pedalboard_fx.py wrapper module | TODO |
| P1 | Resampling support in load_sample() | TODO |
| P1 | Vectorize reverb allpass (lfilter) | TODO |
| P2 | Memory-efficient streaming for long renders (1h+) | TODO |
| P2 | Real-time preview (optional, v2) | TODO |
