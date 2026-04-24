# Decision Log — Audiomancer

> Format: ## DEC-XXX — Title [STATUS]

---

## DEC-001 — Pure numpy/scipy audio toolkit [ACTIVE]

**Date:** 2026-03-24 (updated 2026-03-27)
- **Decision:** No DAW features, no MIDI, no AI generation. Minimal Python audio toolkit.
- **Why:** Keep scope tight — audiomancer produces stems for Akasha, nothing more.
- **Impact:** Each module stays under 400 lines. No external audio frameworks.

## DEC-002 — Gated LUFS (pyloudnorm) for ambient target loudness [ACTIVE]

**Date:** 2026-04-24
- **Decision:** For low-LUFS ambient targets (-20 and below), use
  `audiomancer.layers.normalize_lufs_gated` (pyloudnorm BS.1770-4, gated),
  not the legacy `normalize_lufs` (ungated K-weighted mean).
- **Why:** The legacy ungated mean diverges 2-3 dB on ambient pieces with
  silence windows (e.g. sparse didgeridoo events). DAWs, YouTube, Spotify
  report the gated value — users/streamers will hear that.
- **Impact:** Adds pyloudnorm dependency. Ambient configs (V006+) with
  `master_mode: "ambient"` in META automatically use the gated path.

## DEC-003 — Ambient master (no maximizer) for meditation targets [ACTIVE]

**Date:** 2026-04-24
- **Decision:** `audiomancer.mastering.ambient_master_chain` replaces the
  default `master_chain` for configs with `META["master_mode"] = "ambient"`.
  Chain: highpass → mono-bass → gated LUFS normalize → passive peak cap.
- **Why:** Default `master_chain` runs a pedalboard Limiter that acts as a
  loudness maximizer — brings any signal up to within ~1 dB of the ceiling,
  boosting LUFS by +10 dB. Fine at -14 LUFS YouTube target, catastrophic
  for the -20 LUFS target that sleep/meditation pieces need.
- **Impact:** `scripts/render_mix.py` and `scripts/render_stem.py` dispatch
  between chains based on META. V006 (Muladhara) is the first config using
  ambient mode; V004 / V005 continue using the default maximizer chain.
