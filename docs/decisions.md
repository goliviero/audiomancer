# Decision Log — Audiomancer

> Format: ## DEC-XXX — Title [STATUS]

---

## DEC-001 — Pure numpy/scipy audio toolkit [ACTIVE]

**Date:** 2026-03-24 (updated 2026-03-27)
- **Decision:** No DAW features, no MIDI, no AI generation. Minimal Python audio toolkit.
- **Why:** Keep scope tight — audiomancer produces stems for Akasha, nothing more.
- **Impact:** Each module stays under 400 lines. No external audio frameworks.
