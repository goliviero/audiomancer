# Decision Log — Audiomancer

> Format: ## DEC-XXX — Title [STATUS]

---

## DEC-001 — Pure numpy/scipy audio toolkit [ACTIVE]

**Date:** 2026-03-24
- **Decision:** No DAW features, no MIDI, no AI generation. Minimal Python audio toolkit.
- **Why:** Keep scope tight — audiomancer produces stems for Akasha, nothing more.
- **Impact:** All modules stay under 200 lines. No external audio frameworks.
