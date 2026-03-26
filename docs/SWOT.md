# SWOT — Audiomancer

> Last updated: 2026-03-26 (post-audit)

## Strengths
- Pure Python, minimal deps (numpy/scipy/pedalboard/soundfile)
- 14 modules, 12 scripts, 290+ tests — zero failures
- GitHub Actions CI (pytest + ruff on push)
- Pinned dependencies with upper bounds
- I/O error handling with descriptive messages
- Direct integration with Akasha Portal pipeline
- Spectral processing, spatial audio, harmony/tuning systems
- Loopable progressive stems (compose.py)
- Clean git history — conventional commits, single branch, no secrets
- Consistent code style — no TODO/FIXME/HACK markers, no dead code
- Visual + audio gallery (11_gallery.py) for showcasing capabilities
- One-liner API (quick.py) lowers entry barrier

## Weaknesses
- No real-time preview (batch-only)
- No GUI — CLI scripts only

## Opportunities
- Reusable as standalone library (PyPI)
- Field recording integration (Zoom H1n, Annecy)
- Piano/guitar stem processing pipeline

## Threats
- Scope creep toward DAW features
- Audio quality ceiling without DSP expertise
- Single maintainer, no contributor docs
