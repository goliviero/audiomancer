# Test Log — Audiomancer

> Parameter iterations with exact params + results. Append-only.

---

## V006 — Muladhara Root Chakra — 2026-04-24

Brief: 5:00 grounding loop, C minor, -20 LUFS / -3 dBTP / 48kHz / 24-bit,
3 layers max simultaneous + optional 60Hz subliminal, pre-fade 1.5s in/out.

### Iteration 1 — initial autonomous run

Config: 3 stems (foundation / ochre_mid / didgeridoo) + subliminal.
Ochre: single accent note (G3 at -20dB), LP 3000 Hz, sat drive 1.1.

**Verdict:** targets met, but mids/uppers felt empty — LP 3000 + low drive
killed too much upper-mid body for the "last embers through dusk" palette.

### Iteration 2 — ochre body boost (current)

Changes:
- Ochre chord: `[(C3, 0dB), (G3, -20dB), (C4, -22dB), (G4, -26dB)]`
  — 4 voices instead of 2, still monomodal C-minor compatible (no new
  note classes, just harmonic reinforcement).
- Ochre LP: 3000 → **4000 Hz** (more upper-mid body).
- Ochre sat drive: 1.1 → **1.6** (warmer even-harmonic bloom).
- Ochre mix volume: -10 → **-8 dB** (carries more of the piece).

### Final mix validation (seed=42, V006 / iteration 2)

| metric            | target     | measured   | pass |
|-------------------|------------|------------|------|
| duration          | 300 s      | 300.0 s    | ✓    |
| sample rate       | 48000 Hz   | 48000 Hz   | ✓    |
| bit depth         | 24-bit     | PCM_24     | ✓    |
| integrated LUFS   | ~-20       | -20.04     | ✓    |
| true peak ceiling | -3 dBTP    | -9.31 dBTP | ✓ (well under) |
| loop jump         | ~0         | 0.00000    | ✓    |
| pre-fade in/out   | 1.5 s      | 1.5 s      | ✓    |

### Per-stem levels (seed=42)

| stem        | peak dBFS | TP dBTP   | integrated LUFS |
|-------------|----------:|----------:|----------------:|
| foundation  | -10.43    | -10.42    | -20.03          |
| ochre_mid   | -11.15    | -11.14    | -20.04          |
| didgeridoo  |  -5.50    |  -5.50    | -20.00          |
| subliminal  | -14.52    | -14.52    | -20.00          |

### Calibration notes

- LFO periods tuned as sub-multiples of 300s for cleaner loop closure:
  foundation breath = 300/14 ≈ 21.43s, ochre_mid breath = 300/11 ≈ 27.27s,
  subliminal tremolo = 300/12 = 25.0s.
- `loop=CHECK` reported by verify_loop is a metric artifact: pre-fade
  within the 5s RMS window skews `level_diff_db` upward even though the
  actual sample jump is 0. Ignore for pre-faded exports.

### Variants (2026-04-24)

All share the V006 base (foundation_drone + ochre_pad + sparse_didgeridoo +
subliminal_sine). Differences below. All hit -20.00 ± 0.06 LUFS / TP below
-3 dBTP / 300.0s / PCM_24.

| variant       | foundation env | ochre       | didgeridoo | extras                           |
|---------------|---------------|-------------|-----------:|----------------------------------|
| V006          | flat          | single (C)  | 2 events   | —                                |
| V006_tidal    | **sparse**    | single (C)  | 3 events (wider drift) | harmonic_bloom ~1/min |
| V006_ember    | flat          | **C + Eb** (opposite-phase arc/breathing envs) | 2 events | blooms + overtone whispers |
| V006_cavern   | **breathing** | single (C)  | 4 events (bigger room reverb) | brown noise wash + whispers |

New MIX field: `stem_envelopes: {stem_name: profile_name}` — applies
`density_profile` to a single stem before mixing. Enables per-stem dropouts
without touching the stem builder itself.

### Mid-focused variants (no-highs fix)

User feedback on V006_ember: overtone_whisper micro-events produced
ear-piercing pure sines at ~45s / 1min / 4:45 (overtone_whisper generates
sines at 5x/7x/9x fundamental, so Eb4 → 1555-2800 Hz cut through the bed).

Fix: drop overtone_whisper + restrict bloom chord pool. Three variants
emphasizing mid body with **no synthesized highs**:

| variant         | mid source                                   | HF energy >2kHz |
|-----------------|----------------------------------------------|----------------:|
| V006_ember      | overtone_whisper + harmonic_bloom            | 0.01%           |
| V006_ember_v2   | harmonic_bloom only (low chord pool)         | 0.00%           |
| V006_warmth     | + `warm_drone` = evolving_drone @ A3 220Hz   | 0.00%           |
| V006_braise     | + `breath_pad` = breathing_pad C-minor       | 0.00%           |

All hit -20.00 ± 0.02 LUFS / TP below -3 dBTP / 300.0s / PCM_24.
