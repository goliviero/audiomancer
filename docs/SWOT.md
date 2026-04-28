# SWOT — Audiomancer

> Last updated: 2026-04-28 (senior-dev-analyze)

## Strengths

- **Mature lib**: 24 modules, clear separation (synth/effects/modulation/spectral/spatial/...) — no module is a god-class
- **Test coverage**: 489 pytest passing in ~8s, 23 test files (one per lib module + tests/test_builders)
- **CI**: GitHub Actions, ruff lint + pytest on Python 3.11/3.12 matrix
- **Pinned + reproducible**: `pyproject.toml` (version-bounded) + `requirements.lock` (pip-compile)
- **Config-driven pipeline (V006+)**: `configs/V0XX.py` (META + STEMS + MIX) + generic `render_stem.py` / `render_mix.py` — replaces per-video forks cleanly
- **Stem builder REGISTRY** (`audiomancer.builders`): 14 parametric generators reusable across configs
- **Decisions logged**: `docs/decisions.md` with 5 ACTIVE decisions tracing rationale (gated LUFS, ambient master, layered brown stereo, loop boundary continuity)
- **Clean git**: 29 commits, conventional, single branch, no secrets, no dead branches
- **Production proven**: V003/V004/V005/V006 (+6 variants)/V007 livrés
- **Loop-safe by design**: `make_loopable` + opt-in `boundary_continuity` (DEC-005) + `verify_loop` quality score
- **No-DAW discipline held**: no realtime, no VST hosting, no AI audio (Suno-free)

## Weaknesses

- 🟡 **Doc/code drift just fixed**: README + ARCHITECTURE counts were stale (367/398 → 489 tests, 15 → 24 modules). Cleanup applied 2026-04-28; recurrence risk if `/update-docs` not run pre-commit
- 🟡 **Configs V008/V009/V011 blocked**: 5× `# TODO: source` for CC0 samples (singing bowl A3, heavy rain 180s, brass sustain E3). Configs unrendable until sourced
- 🟡 **Ruff scope mismatch**: CI lints only `audiomancer/` + `tests/`, but `scripts/` + `configs/` cumulate 98 errors (mostly I001 import-order, some E501 line-length, F401 unused). Auto-fixable (71 of 98) but never run
- 🟡 **V006 variant proliferation**: 7 V006*.py configs (~120 lines each) duplicate META + ~80% of STEMS. No shared base config helper; deltas would fit in 20-30 lines each
- 🟢 **V007 exploration scripts (60-66)**: 7 iteration scripts (~190 lines each) for one final builder — could compress to single archived `60_v007_brown_exploration.py` with the validated recipe inline
- 🟢 **Gallery iteration spread**: `11_gallery.py` (893 lines), `50_gallery_v2.py` (425), `51_gallery_v3.py` (385), `52_gallery_v4.py` (319) — only the most recent is "current"; older could be deleted
- 🟢 **Long lib modules acceptable**: harmony.py 500 / spectral.py 463 / synth.py 424. DEC-001 amended 2026-04-28

## Opportunities

- **Shared V006 base** (~30min, low effort): extract V006 META + foundation/ochre/subliminal stems into `configs/_v006_base.py`, variants override deltas only — 5-7 files shrink ~70%
- **Ruff scope expansion** (~30min): add `scripts/` + `configs/` to CI (or auto-fix the 71 fixable errors first), aligns with `pyproject.toml` per-file-ignore that already covers E402
- **Sample sourcing batch** (1-2h): source the 3 missing CC0 samples on Freesound, unblock V008/V009/V011 renders together
- **Archive script cleanup** (~1h): drop `50/51_gallery_v2/v3.py` (superseded by v4), drop `60-65_v007_brown_v1-v6.py` (superseded by v7); keep validated final + latest gallery only
- **Reusable as PyPI lib**: structure already supports it (clean entry points, version pinned). Would need `__all__` in modules and minimal usage docs

## Threats

- 🟡 **Doc rot recurrence**: with V006 → V011 cadence, counts and "Next Steps" drift faster than commits update them. Either run `/update-docs` per release or replace counts with `pytest --collect-only | wc -l`-style auto-derivation
- 🟡 **Ecosystem dependency on Akasha pipeline**: every config targets Akasha's mastering chain (LUFS −20, ceiling −3 dBTP, 48kHz/24-bit). If Akasha pivots away from ffmpeg loop, configs need rework
- 🟢 **Pedalboard upstream churn**: hard dep on Spotify's pedalboard (reverb/delay/chorus). Upper-bounded `<1` — bump to 1.x will need testing
- 🟢 **Single maintainer, no contributor doc**: README is user-facing; CONTRIBUTING.md absent. Low risk for solo project
- 🟢 **CC0 sample dependency**: V008/V009/V011 require external samples — supply chain depends on Freesound availability and license
