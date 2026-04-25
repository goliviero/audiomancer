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

## DEC-004 — Layered stereo brown noise builder [ACTIVE]

**Date:** 2026-04-25
- **Decision:** New builder `audiomancer.builders.layered_brown_stereo`
  generates multi-band brown noise: N independent cumsum streams summed
  per channel per layer, with truly decorrelated L/R (each channel uses
  its own stream set). Layers are tuples `(lp_hz, db_offset)` mixed at
  per-layer dB. The `texture/noise_wash` preset is kept for backward
  compat but not used for V007.
- **Why:** Single-stream brown noise has perceptible amplitude swells from
  random low-freq bunching (CLT). Multi-stream summing reduces envelope
  variance (MindAmend "smoothed" principle). Independent L/R streams
  produce the wide stereo image that single mono-to-stereo duplication
  cannot ("sleepy cacophony" target). Multi-band stack (sub + bed + body
  + air) gives the deep+aéré quality that pure-brown filtering alone misses.
- **Impact:** V007 ships with 4 layers (LP 500 / 1550 / 2500 / 3500),
  6 streams/channel/layer, `n_streams_per_channel` and `hp_hz` exposed
  as params. Optional `breath_cycle_sec`/`breath_depth_db` for anti-fatigue
  on extended listening (off by default). Reusable for any noise-based
  config. Validated via the v1→v7 exploration in
  `scripts/60_v007_brown_variations.py` … `66_v007_brown_v7.py`.

## DEC-005 — Opt-in loop boundary continuity for aperiodic content [ACTIVE]

**Date:** 2026-04-25
- **Decision:** `compose.make_loopable` gains a `boundary_continuity: bool`
  flag (default False). When True, applies a global DC tilt of magnitude
  `(result[-1] - result[0])` across the whole signal so the file boundary
  is sample-clean. Renderers wire it from `META["loop_boundary_continuity"]`.
- **Why:** The 5s crossfade ensures *audible* continuity inside the loop,
  but `result[-1]` ends at `stem[crossfade_samples - 1]`, not `stem[0]`.
  For musical content (V005/V006) the difference is tiny (signal is
  near-periodic). For pure noise (V007), the cumsum nature gives a
  jump of ~0.16 → audible click on `ffmpeg -stream_loop`. The DC tilt
  correction is per-sample ~1e-8, inaudible vs the noise floor.
- **Impact:** Existing configs unchanged (flag default off → exact
  byte-for-byte output preserved, test_not_modified_in_middle still
  passes). V007 opts in via `loop_boundary_continuity: True` in META.
  Other future aperiodic configs (rain, wind, fire) should opt in.
