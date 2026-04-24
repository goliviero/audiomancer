# ARCHITECTURE.md — Audiomancer

Minimal Python audio toolkit for ambient/meditation sound design. Scripts > framework.

---

## Structure

```
audiomancer/
├── audiomancer/
│   ├── __init__.py          # SAMPLE_RATE (44100), DEFAULT_AMPLITUDE (0.5)
│   ├── synth.py             # Waveforms + drones + pads (seeded jitter_cents) + granular + karplus_strong
│   ├── binaural.py          # Stereo binaural beats, 6 presets (theta/alpha/delta/solfeggio/om)
│   ├── effects.py           # Scipy filters (lowpass/highpass) + Pedalboard (reverb/delay/chorus/compress)
│   │                        # Presets: reverb_hall, reverb_cathedral, delay_long, chorus_subtle
│   ├── saturation.py        # tape_saturate (asymmetric soft-clip) + tape_hiss + vinyl_wow (pitch flutter)
│   ├── layers.py            # mix (dB), layer (linear), crossfade, loop_seamless
│   │                        # normalize_lufs (K-weighted ITU-R BS.1770), measure_lufs
│   ├── modulation.py        # LFOs (sine/triangle), drift (fast: uniform_filter1d), evolving_lfo
│   │                        # multi_lfo (stack of N non-sync LFOs), random_walk (OU bounded, lfilter-vectorized)
│   │                        # apply_amplitude_mod, apply_filter_sweep
│   ├── textures/            # 9 ready-to-use evolving ambient presets (texture bank)
│   │   ├── _presets.py      # Preset implementations (evolving_drone, deep_space, etc.)
│   │   └── _registry.py     # Registry + generate() dispatcher
│   ├── compose.py           # fade_envelope, tremolo, stitch, make_loopable, verify_loop
│   │                        # density_profile (flat/breathing/arc/random_walk/sparse)
│   ├── mastering.py         # mono_bass (Linkwitz-Riley), soft_clip (cascade stages for warmth), limiter
│   │                        # master_chain (default: 3-stage cascade)
│   ├── stochastic.py        # scatter_events (texture micro-events), DEFAULT_EVENTS
│   │                        # micro_events (typed: harmonic_bloom/grain_burst/overtone_whisper)
│   │                        # micro_silence_env (multiplicative duck envelope)
│   ├── builders.py          # Parametric stem generators for config-driven rendering
│   │                        # REGISTRY: pad_alive, arpege_bass, pendulum_bass, binaural_beat,
│   │                        #          texture (wraps any textures/ preset), piano_processed
│   │                        # derived_seed helper for per-role stem coordination
│   ├── piano_presets.py     # 3 piano processing presets (bass_drone/mid_pad/sparse_notes)
│   │                        # Shared between scripts/piano/process_piano.py and builders.piano_processed
│   ├── quick.py             # One-liner API: q.drone, q.pad, q.binaural, q.texture, q.mix
│   │                        # note() converter, FREQS dict, HARMONICS_* presets
│   ├── field.py             # Field recording pipeline: clean, noise_gate, process_field
│   ├── utils.py             # I/O (load_audio auto-resample, export_wav), load_sample (bank loader)
│   │                        # normalize, fade_in/out, trim, mono/stereo, duration
│   ├── spectral.py          # FFT: STFT/ISTFT, freeze, blur, pitch_shift, spectral_gate, morph
│   │                        # paulstretch (extreme time-stretch via phase randomization)
│   ├── spatial.py           # pan, auto_pan, stereo_width, mid/side, haas_width, rotate
│   ├── harmony.py           # Scales, tuning (just/Pythagorean), chord generators, sacred/solfeggio freqs
│   └── envelope.py          # ADSR (linear/exp), AR, multi-segment, breathing, swell, gate_pattern
├── configs/
│   ├── __init__.py
│   └── V005.py              # V005 focus gamma config (META + STEMS + MIX)
├── scripts/
│   ├── render_stem.py       # GENERIC: python render_stem.py --config V005 --stem warm_pad
│   ├── render_mix.py        # GENERIC: python render_mix.py --config V005 --preview
│   ├── 01-09_*.py           # V003-era one-shot scripts (binaural, drones, progressive, etc.)
│   ├── 10_akasha_stems.py   # V003 production: 6 stems (--vary, --preview, --standalone)
│   ├── 11_gallery.py        # Visual + audio gallery (17 PNG + 13 WAV)
│   ├── 12_field_integration.py, 13_instrument_stems.py
│   ├── 16-20_v004_*.py      # V004 archived production (throat chakra)
│   ├── 21-29_v005_*.py      # V005 archived production (focus gamma) — frozen post-release
│   └── piano/               # Piano workflow (USB-MIDI -> WAV -> ambient stem)
│       ├── record_piano.py  # mido + python-rtmidi capture (live note name monitoring)
│       ├── render_midi.py   # FluidSynth subprocess wrapper (offline MIDI->WAV)
│       ├── process_piano.py # CLI wrapper around piano_presets (bass_drone/mid_pad/sparse_notes)
│       ├── quantize_midi.py # Post-record grid quantization (1/4..1/32, strength param)
│       └── README.md        # 3-command workflow documentation
├── tests/                   # 20 test files, 398 tests passing
├── samples/                 # Source audio (gitignored, cc0/ + own/ structure)
├── assets/                  # Soundfonts etc. (gitignored)
└── output/                  # Generated WAV (gitignored)
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
