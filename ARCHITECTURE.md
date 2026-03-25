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
│   ├── binaural.py          # Stereo binaural beats, 6 presets (theta/alpha/delta/solfeggio/om)
│   ├── effects.py           # Scipy filters (LP/HP) + Pedalboard (reverb/delay/chorus/compress)
│   │                        # Presets: reverb_hall, reverb_cathedral, delay_long, chorus_subtle
│   ├── layers.py            # mix (dB), layer (linear), crossfade, loop_seamless, normalize_lufs
│   ├── modulation.py        # LFO (sine/triangle), Brownian drift, evolving LFO
│   │                        # Amplitude modulation, time-varying filter sweep
│   ├── textures.py          # 9 ready-to-use evolving ambient presets (texture bank)
│   │                        # Registry + generate() dispatcher
│   ├── field.py             # Field recording pipeline: clean, noise_gate, process_field
│   └── utils.py             # I/O (WAV), normalize, fade_in/out, trim, mono/stereo, duration
├── scripts/
│   ├── 01_binaural_432hz.py    # 10-min alpha binaural beat
│   ├── 02_drone_pad.py         # 1-min ambient drone + pad
│   ├── 03_piano_reverb.py      # WAV → cathedral reverb (CLI)
│   ├── 04_field_processing.py  # Field recording cleanup (CLI)
│   ├── 05_layer_akasha.py      # 5-min preview mix (drone + binaural + pad)
│   ├── 06_akasha_v003.py       # 30-min production: Om drone + theta + C major + pink noise
│   └── 07_stems_v003.py        # 5-min loopable stems for Akasha V003
├── tests/                      # 128 pytest tests
├── samples/                    # Source audio (gitignored)
├── output/                     # Generated WAV (gitignored)
└── _fractal_backup/            # Archived Fractal audio code (18 modules)
```

---

## Stack

- **Python 3.10+**
- **numpy** — signal generation, array operations
- **scipy** — Butterworth IIR filters (lowpass/highpass)
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
    Modulation (modulation.py)
      LFO sine/triangle
      Brownian drift (random, non-repeating)
      Evolving LFO (drifting rate/depth)
      Amplitude mod, filter sweep
         │
         ▼
    Textures (textures.py)         Layering (layers.py)
      9 presets combining             mix / layer (volume control)
      synth + mod + effects           crossfade / loop_seamless
      generate("deep_space", 300)     normalize_lufs (-14 dB YouTube)
         │                              │
         ▼                              ▼
    Export (utils.py)
      export_wav (16/24/32-bit)
      → output/*.wav
```

---

## Key Decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | Scripts > framework | Fractal was over-engineered (18 modules, DAW routing). Scripts are disposable. |
| 2 | No AI audio | YouTube 2026 flags AI content. Suno under lawsuit. 100% synthesis. |
| 3 | numpy in-memory | Simple, correct. 30 min ≈ 600 MB RAM. For 3h+, generate 5-10 min and loop in ffmpeg. |
| 4 | Pedalboard for effects | VST-quality, maintained by Spotify, zero config. Scipy for filters only. |
| 5 | -14 LUFS target | YouTube streaming standard. RMS approximation sufficient for ambient content. |

---

## Integration with Akasha Portal

Audiomancer generates WAV stems → placed in `akasha-portal/sounds/processed/` → looped by ffmpeg via `mix_soundscape.py`. No code dependency between the two repos.

---

## Current State

- **v0.2.0** — 8 modules, 7 scripts, 128 tests
- Production script (06) generates 30 min ambient audio
- V003 stem generator (07) produces 5-min loopable stems
- Modulation system: LFO, Brownian drift, evolving LFO
- Texture bank: 9 evolving presets (evolving_drone, breathing_pad, deep_space, ocean_bed, crystal_shimmer, earth_hum, ethereal_wash, singing_bowl, noise_wash)

---

## Texture Bank

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

## Next Steps

1. **Field recording integration** — Zoom H1n recordings from Annecy
2. **Piano/guitar stems** — live instrument processing pipeline
3. **Additional textures** — expand the bank based on production needs
