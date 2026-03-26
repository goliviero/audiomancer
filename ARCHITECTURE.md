# ARCHITECTURE.md — Audiomancer

Minimal Python audio toolkit for ambient/meditation sound design. Scripts > framework.

---

## Structure

```
audiomancer/
├── audiomancer/
│   ├── __init__.py          # SAMPLE_RATE (44100), DEFAULT_AMPLITUDE (0.5)
│   ├── synth.py             # Waveforms: sine, saw, square, triangle, noise (white/pink/brown)
│   │                        # Drones: harmonic overtones. Pads: detuned unison, chord pads
│   │                        # Granular synthesis: grain clouds from source buffers
│   ├── binaural.py          # Stereo binaural beats, 6 presets (theta/alpha/delta/solfeggio/om)
│   ├── effects.py           # Scipy filters (LP/HP) + Pedalboard (reverb/delay/chorus/compress)
│   │                        # Presets: reverb_hall, reverb_cathedral, delay_long, chorus_subtle
│   ├── layers.py            # mix (dB), layer (linear), crossfade, loop_seamless
│   │                        # normalize_lufs (K-weighted ITU-R BS.1770), measure_lufs
│   ├── modulation.py        # LFO (sine/triangle), Brownian drift, evolving LFO
│   │                        # Amplitude modulation, time-varying filter sweep
│   ├── textures.py          # 9 ready-to-use evolving ambient presets (texture bank)
│   │                        # Registry + generate() dispatcher
│   ├── compose.py           # Temporal composition: fade_envelope (breakpoints),
│   │                        # tremolo, stitch (sections + crossfade), make_loopable, verify_loop
│   ├── mastering.py         # Final chain: mono_bass (Linkwitz-Riley), soft_clip, limiter, master_chain
│   ├── stochastic.py        # Random micro-event placement: scatter_events, DEFAULT_EVENTS
│   ├── quick.py             # One-liner API: q.drone, q.pad, q.binaural, q.texture, q.mix
│   │                        # note() converter, FREQS dict, HARMONICS_* presets
│   ├── field.py             # Field recording pipeline: clean, noise_gate, process_field
│   ├── utils.py             # I/O (WAV), normalize, fade_in/out, trim, mono/stereo, duration
│   ├── spectral.py          # FFT: STFT/ISTFT, freeze, blur, pitch_shift, spectral_gate, morph
│   │                        # spectral_balance (multi-stem frequency analysis)
│   ├── spatial.py           # pan, auto_pan, stereo_width, mid/side, haas_width, rotate
│   ├── harmony.py           # Scales (22 types), tuning (just/Pythagorean), chord generators,
│   │                        # sacred/solfeggio/planetary freqs, harmonic/subharmonic series
│   └── envelope.py          # ADSR (linear/exp), AR, multi-segment, breathing, swell, gate_pattern
├── scripts/
│   ├── 01_binaural_432hz.py    # 10-min alpha binaural beat
│   ├── 02_drone_pad.py         # 1-min ambient drone + pad
│   ├── 03_piano_reverb.py      # WAV → cathedral reverb (CLI)
│   ├── 04_field_processing.py  # Field recording cleanup (CLI)
│   ├── 05_layer_akasha.py      # 5-min preview mix (drone + binaural + pad)
│   ├── 06_akasha_v003.py       # 30-min production: Om drone + theta + C major + pink noise
│   ├── 07_stems_v003.py        # 5-min loopable stems for Akasha V003
│   ├── 08_showcase.py          # 53 x 15s clips across 5 categories (audition tool)
│   ├── 09_progressive_stem.py  # 5-min progressive loopable stem with sections
│   ├── 10_akasha_stems.py      # PRODUCTION: 6 stems (--vary, --preview, --standalone)
│   ├── 11_gallery.py           # Visual + audio gallery (17 PNG + 13 WAV)
│   ├── 12_field_integration.py # Field recording pipeline (Zoom H1n → ambient layer)
│   └── 13_instrument_stems.py  # Instrument → ambient (wash/freeze/granular modes)
├── tests/                      # 16 test files
├── samples/                    # Source audio (gitignored)
└── output/                     # Generated WAV + gallery PNG (gitignored)
```

---

## Stack

- **Python 3.10+**
- **numpy** — signal generation, array operations
- **scipy** — Butterworth IIR filters, FFT, windowing
- **pedalboard** (Spotify) — reverb, delay, chorus, compression (VST-quality)
- **soundfile** — WAV read/write (via libsndfile)
- **pytest** — testing

---

## Signal Flow

```
Synthesis (synth.py)          Processing (effects.py)
  sine/saw/square/triangle      lowpass / highpass
  drone (harmonic stacking)     reverb (hall/cathedral)
  chord_pad (detuned voices)    delay, chorus, compress
  noise (white/pink/brown)      chain (custom effect list)
         │                              │
         ▼                              ▼
    Modulation (modulation.py)     Spectral (spectral.py)
      LFO sine/triangle             freeze, blur, pitch_shift
      Brownian drift                 spectral_gate, morph
      Evolving LFO                   STFT/ISTFT engine
      Amplitude mod, filter sweep
         │                              │
         ▼                              ▼
    Textures (textures.py)         Compose (compose.py)
      9 presets combining             fade_envelope (breakpoints)
      synth + mod + effects           tremolo (slow LFO)
      generate("deep_space", 300)     stitch (sections + crossfade)
         │                            make_loopable (loop seal)
         ▼                              │
    Spatial (spatial.py)               │
      pan, auto_pan, rotate            │
      stereo_width, mid/side           │
      haas_width                       │
         │                              │
         ▼                              ▼
    Layering (layers.py)
      mix / layer (volume control)  ←──┘
      crossfade / loop_seamless
      normalize_lufs (-14 dB YouTube)
         │
         ▼
    quick.py (one-liner API)       Export (utils.py)
      q.drone / q.pad / q.mix   →  export_wav (16/24/32-bit)
      q.texture / q.binaural        → output/*.wav

    Harmony (harmony.py)           Envelope (envelope.py)
      scales (22 types)              ADSR (linear/exp), AR
      just intonation / Pythagorean  multi-segment, breathing
      sacred/solfeggio/planetary     swell, gate_pattern
      chord generators               (feed into any synthesis)
```

---

## Key Decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | Scripts > framework | Fractal was over-engineered (18 modules, DAW routing). Scripts are disposable. |
| 2 | No AI audio | YouTube 2026 flags AI content. Suno under lawsuit. 100% synthesis. |
| 3 | numpy in-memory | Simple, correct. 30 min ≈ 600 MB RAM. For 3h+, generate 5-10 min and loop in ffmpeg. |
| 4 | Pedalboard for effects | VST-quality, maintained by Spotify, zero config. Scipy for filters only. |
| 5 | -14 LUFS target | YouTube streaming standard. K-weighted ITU-R BS.1770 measurement. |

---

## Integration with Akasha Portal

Audiomancer generates WAV stems → placed in `akasha-portal/sounds/processed/` → looped by ffmpeg via `mix_soundscape.py`. No code dependency between the two repos.

---

## Texture Bank (9 presets)

| Preset | Role | Tonal | Description |
|--------|------|-------|-------------|
| evolving_drone | foundation | yes | Harmonic drone with drifting amplitude and filter |
| breathing_pad | foundation | yes | Chord pad with slow inhale/exhale movement |
| deep_space | foundation | no | Dark vast texture — sculpted brown noise |
| ocean_bed | bed | no | Underwater ambient — pink noise with filter waves |
| crystal_shimmer | overlay | yes | High-frequency cluster — sparkly, ethereal |
| earth_hum | foundation | yes | Sub-bass drone + brown noise — primal, grounding |
| ethereal_wash | foundation | yes | Saw pad drenched in reverb — dreamy, floating |
| singing_bowl | accent | yes | Inharmonic resonance — metallic, meditative |
| noise_wash | bed | no | Colored noise with evolving filter — ambient bed |

---

## Progressive Stem Structure (09_progressive_stem.py)

```
0:00 – 1:30  AWAKENING  Drone alone, filter closed (800 Hz), volume 0.25
1:30 – 3:00  DEEPENING  Pad enters, filter opens (→ 2500 Hz), tremolo 0.15 Hz
3:00 – 4:00  FULLNESS   Drone + pad + noise_wash, peak density
4:00 – 5:00  RETURN     Texture exits, pad fades, filter closes, volume → 0.25
Loop seal: 5s crossfade on loop point via make_loopable()
```

---

## Next Steps

1. **V003 Mycelium production** — 174 Hz solfège stems with earth_hum textures
2. **Live performance** — real-time parameter control for streaming
