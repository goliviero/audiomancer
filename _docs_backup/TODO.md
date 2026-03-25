# TODO — Fractal

> Post-audit task list. Updated: 2026-03-25.

## Done

| Task | Phase | Date |
|------|-------|------|
| Signal generation + export (constants, signal, generators, export) | 1 | 2026-03-24 |
| Envelopes (ADSR, fades, swell, gate, tremolo, automation) | 2 | 2026-03-24 |
| Effects (LPF, HPF, BPF, EQ, reverb, delay, distortion, stereo width) | 3 | 2026-03-24 |
| Track + Mixer (Track, Bus, Session, pan, bus routing) | 4 | 2026-03-25 |
| Sequencer (Clip, Pattern, Sequencer, tempo, loops) | 5 | 2026-03-25 |
| Audit: fix Reverb tail truncation in comb filter | audit | 2026-03-25 |
| Audit: extract `_apply_sos_stereo` helper (DRY filters) | audit | 2026-03-25 |
| Audit: name allpass coefficient constant | audit | 2026-03-25 |
| Audit: complete `__init__.py` public API re-exports | audit | 2026-03-25 |
| Audit: fix hardcoded 44100 in example 50 | audit | 2026-03-25 |
| Audio rebalance: flagship examples 10, 20, 30, 40 (noise-led mix) | fix | 2026-03-25 |

## In Progress

_Nothing currently._

## Next Up

| Priority | Task | Notes |
|----------|------|-------|
| P0 | Phase 6: 5 flagship demos (51-55) | Meditation, lo-fi beat, evolving drone, sound design, Akasha workflow |
| P0 | Phase 6: docs polish (README, CLAUDE.md final) | After demos |
| P1 | `pedalboard_fx.py` — wrapper for Spotify pedalboard effects | Isolated module per DEC-002 |
| P1 | Resampling in `load_sample()` | scipy.signal.resample or librosa |
| P1 | Vectorize reverb allpass filter | Replace Python loop with scipy.signal.lfilter |
| P2 | Memory-efficient streaming for 1h+ renders | Chunked buffer approach |
| P2 | Real-time preview (optional, v2) | sounddevice or pyaudio |
| P3 | Signal metadata wrapper (v2) | Dataclass with tempo, key, time_sig if needed |
