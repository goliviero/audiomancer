# SWOT — Audiomancer

> Last updated: 2026-03-27 (senior-dev-analyze)

## Strengths
- Pure Python, minimal deps (numpy/scipy/pedalboard/soundfile)
- 16 modules, 13 scripts, 336 tests — zero failures
- GitHub Actions CI (pytest + ruff on push, Python 3.11/3.12 matrix)
- Pinned dependencies with upper bounds
- I/O error handling with descriptive messages
- Direct integration with Akasha Portal pipeline
- Spectral processing, spatial audio, harmony/tuning systems
- Loopable progressive stems (compose.py)
- Mastering chain (mono_bass, soft_clip, limiter, K-weighted LUFS)
- Clean git history — conventional commits, single branch, no secrets
- Consistent code style — no TODO/FIXME/HACK markers, no dead code
- Visual + audio gallery (11_gallery.py) for showcasing capabilities
- One-liner API (quick.py) lowers entry barrier

## Weaknesses
- 🟡 No lock file — reproducible builds not guaranteed
- 🟡 9/17 modules exceed 200-line limit (DEC-001 constraint outdated — harmony.py at 397)
- 🟢 No real-time preview (batch-only, by design)
- 🟢 No GUI — CLI scripts only (by design)

## Opportunities
- Reusable as standalone library (PyPI)
- Field recording integration (Zoom H1n, Annecy)
- Piano/guitar stem processing pipeline
- Extract synth.py normalization pattern into helper (DRY, 6 duplicates)

## Threats
- Scope creep toward DAW features (ARCHITECTURE.md "Next Steps" lists "live performance" — contradicts DEC-001)
- Audio quality ceiling without DSP expertise
- Single maintainer, no contributor docs
- pedalboard hard dependency — breaks if Spotify drops/changes API
