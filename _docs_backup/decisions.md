# Decision Log — Fractal

> Design decisions made during sessions — logged to avoid re-doing work.
> Format: `## DEC-XXX — Title [STATUS]` where STATUS is `ACTIVE` or `SUPERSEDED by DEC-XXX`.

---

## DEC-001 — Signal = raw np.ndarray, no wrapper class (v1) [ACTIVE]

**Date:** 2026-03-24

- **Decision:** A Signal is a plain `np.ndarray`, not a custom class. Mono: `(n_samples,)`. Stereo: `(n_samples, 2)`. Every function takes `(signal, sample_rate)` explicitly.
- **Why:** numpy arrays compose naturally (+, *, slicing, concatenation). No overhead, no leaky abstraction. Adding a dataclass wrapper in v2 is easy; removing one is painful.
- **Impact:** `signal.py`, `generators.py`, `export.py` all operate on raw arrays. Sample rate is always explicit — no hidden global state.

---

## DEC-002 — pedalboard isolated in its own module [ACTIVE]

**Date:** 2026-03-24

- **Decision:** pedalboard effects live in `pedalboard_fx.py`, not in `effects.py`. The import is lazy/optional.
- **Why:** pedalboard is heavy and may not be installed everywhere. Core Fractal (Phase 1-3 without pedalboard) should run with just numpy/scipy/soundfile.
- **Impact:** `effects.py` has no pedalboard import. `pedalboard_fx.py` is only imported when the user explicitly uses it.

---

## DEC-003 — No in-place mutation: always return new arrays [ACTIVE]

**Date:** 2026-03-24

- **Decision:** All signal processing functions return a new array. Never modify the input in place.
- **Why:** Reproducibility and testability. If `apply_envelope(sig)` mutates `sig`, intermediate states can't be inspected or tested. Pure functions are also easier to cache.
- **Impact:** All functions in `signal.py`, `generators.py`, `effects.py` must follow this rule.

---

## DEC-004 — pip install -e . required to run examples [ACTIVE]

**Date:** 2026-03-24

- **Decision:** The fractal package must be installed in editable mode (`pip install -e .`) to resolve imports from `examples/` and scripts.
- **Why:** `examples/` is outside `src/`, so `python examples/foo.py` won't find `fractal` unless it's installed. Editable mode links `src/fractal/` without copying.
- **Impact:** Added `pip install -e .` to Quick Start in README. Tests use `pyproject.toml` pythonpath so they don't need it.

---

## DEC-005 — outputs/ is gitignored, examples produce to outputs/audio/ [ACTIVE]

**Date:** 2026-03-24

- **Decision:** All generated audio files (.wav, .flac, .mp3, .ogg) are gitignored. Examples write to `outputs/audio/` by default.
- **Why:** Audio files are large binary artifacts. They belong in local storage or Proton Drive, not in git history.
- **Impact:** `.gitignore` covers `*.wav`, `*.flac`, `*.mp3`, `*.ogg`. Empty dirs preserved with `.gitkeep`.

---

## DEC-006 — Equal-power panning for stereo positioning [ACTIVE]

**Date:** 2026-03-25

- **Decision:** Track panning uses equal-power (constant-power) law: `gain_L = cos(angle)`, `gain_R = sin(angle)` where angle maps pan [-1,+1] to [0, pi/2].
- **Why:** Linear panning causes a ~3dB dip at center. Equal-power maintains perceived loudness across the stereo field, which is standard in all professional DAWs.
- **Impact:** `track.py::apply_pan()`. All Track.render() output is stereo (n, 2).

---

## DEC-007 — Session as top-level container with bus routing [ACTIVE]

**Date:** 2026-03-25

- **Decision:** Session holds tracks and buses. Signal flow: Track.render() -> sum by bus -> bus effects -> sum buses -> master effects -> output. Master bus always exists.
- **Why:** Mirrors standard DAW workflow. Buses allow shared effects (e.g., reverb send) without duplicating per-track. Solo/mute logic is centralized in Session.
- **Impact:** `mixer.py::Session`. Tracks route to buses via `bus="name"` attribute. Unknown bus names fall back to master.

---

## DEC-008 — Sequencer operates in beats or seconds, no MIDI [ACTIVE]

**Date:** 2026-03-25

- **Decision:** The Sequencer uses beat positions (converted via BPM) or absolute seconds. No MIDI protocol, no note-on/off events, no velocity curves.
- **Why:** MIDI is complex and unnecessary for Fractal's use case (offline render, pre-generated signals). Beats + seconds are simpler and composable. Clip = signal + offset, Pattern = group of clips.
- **Impact:** `sequencer.py`. `Sequencer.add_clip(signal, start_beat)` converts beats to seconds internally. `Pattern.repeat(n)` handles looping by duplicating clips with offsets.
