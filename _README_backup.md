# Fractal

> Headless, code-first audio workstation in pure Python. No DAW. No GUI. Just signal.

From oscillators to full mixdowns — entirely from the command line.
A script IS a session file. Everything is reproducible and versionable.

---

## Quick Start

```bash
# 1. Clone and install
git clone git@github.com:goliviero/fractal.git
cd fractal
pip install -r requirements.txt
pip install -e .   # install fractal as a package (editable)

# 2. Run your first example
python examples/01_sine_440hz.py
# => outputs/audio/01_sine_440hz.wav

# 3. Run the ambient session demo
python examples/10_ambient_layer_mix.py
# => outputs/audio/10_ambient_layer_mix.wav (+ .flac)
# Use with headphones — contains binaural beat!
```

---

## What It Does

Fractal generates, processes, and mixes audio from pure math.
No samples required. No DAW. No plugins.

```python
from fractal.generators import sine, pink_noise, binaural
from fractal.envelopes import SmoothFade, Swell
from fractal.mixer import Session
from fractal.effects import Reverb, EQ, NormalizePeak

# Build a session with tracks, buses, and master chain
session = Session(master_effects=[NormalizePeak(target_db=-1.0)])
session.add_bus("atmosphere", effects=[Reverb(decay=0.35, mix=0.2)])

session.add_track("drone", Swell(rise_time=8.0).apply(sine(110.0, 30.0)), volume_db=-3.0)
session.add_track("noise", SmoothFade(fade_in=5.0).apply(pink_noise(30.0)), bus="atmosphere", volume_db=-12.0)
session.add_track("theta", binaural(200.0, 6.0, 30.0, amplitude=0.1), volume_db=-24.0)

session.export("outputs/audio/my_session.wav")
```

---

## Signal Flow

```
Generator(s) -- Envelope --> Effects chain --> Track --> Mixer --> Export
(phase 1)       (phase 2)     (phase 3)       (phase 4)  (phase 4)  (phase 1)
```

Everything is a `np.ndarray`. No wrapper classes. No magic.

---

## Examples

| Script | What it generates |
|--------|-------------------|
| `examples/01_sine_440hz.py` | Pure A4 tone (440 Hz), 5s |
| `examples/02_major_chord.py` | C major triad (C4+E4+G4), 4s |
| `examples/03_binaural_alpha.py` | Alpha binaural beat — 200Hz carrier, 10Hz beat, 30s |
| `examples/04_harmonic_drone.py` | Harmonic drone — 55Hz root + 5 overtones, 10s |
| `examples/05_pentatonic_arpeggio.py` | A minor pentatonic arpeggio, 0.4s/note |
| `examples/06_noise_texture.py` | Pink noise + 80Hz sub tone, 15s |
| `examples/07_stereo_ping_pong.py` | Alternating L/R sine pulses (440Hz <-> 523Hz) |
| `examples/08_schumann_resonance.py` | Schumann 7.83Hz binaural + 136.1Hz carrier, 60s |
| `examples/09_waveform_comparison.py` | Sine / square / sawtooth / triangle at 220Hz |
| `examples/10_ambient_layer_mix.py` | Full ambient layer mix: sub + harmonics + noise + theta binaural |
| **Phase 2 — Envelopes** | |
| `examples/11_fade_in_out.py` | Basic fade in/out on a sine wave |
| `examples/12_adsr_keys.py` | ADSR envelope — plucked note effect |
| `examples/13_smooth_vs_linear_fade.py` | Smooth vs linear crossfade comparison |
| `examples/14_exponential_fade.py` | Exponential fade with adjustable steepness |
| `examples/15_swell_pad.py` | Swell envelope — rising pad sound |
| `examples/16_gate_rhythm.py` | Gate envelope — rhythmic volume chopping |
| `examples/17_tremolo_sine.py` | Tremolo effect on sustained note |
| `examples/18_automation_curve.py` | Custom automation curve with breakpoints |
| `examples/19_binaural_with_envelope.py` | Binaural beat with envelope shaping |
| `examples/20_ambient_session_v2.py` | Phase 2 flagship: full session with envelopes |
| **Phase 3 — Effects** | |
| `examples/21_lowpass_sweep.py` | Low-pass filter sweep |
| `examples/22_eq_sculpted_drone.py` | Parametric EQ — sculpted drone |
| `examples/23_stereo_widening.py` | Stereo width adjustment via mid/side |
| `examples/24_reverb_room_sizes.py` | Algorithmic reverb — small/medium/large rooms |
| `examples/25_delay_echo.py` | Delay effect — echo patterns |
| `examples/26_distortion_warmth.py` | Soft distortion — clean to warm to fuzz |
| `examples/27_effect_chain.py` | EffectChain: HPF + EQ + Reverb + Normalize |
| `examples/28_bandpass_radio.py` | Bandpass 'radio' effect |
| `examples/29_reverb_wash.py` | 100% wet reverb wash |
| `examples/30_ambient_session_v3.py` | Phase 3 flagship: 5 layers + effects + master chain |
| **Phase 4 — Track + Mixer** | |
| `examples/31_track_basic.py` | Single track with volume and panning |
| `examples/32_multitrack_mix.py` | Multi-track mix via Session |
| `examples/33_panning_demo.py` | Stereo panning: left, center, right |
| `examples/34_mute_solo.py` | Mute/Solo comparison |
| `examples/35_bus_routing.py` | Bus routing: drone bus + texture bus |
| `examples/36_bus_effects.py` | Per-bus effects (reverb, EQ) |
| `examples/37_session_export.py` | Session.export() one-liner |
| `examples/38_automation_volume.py` | Volume automation via AutomationCurve |
| `examples/39_complex_session.py` | Complex session: 6 tracks, 2 buses |
| `examples/40_ambient_session_v4.py` | Phase 4 flagship: full Session + Bus routing |
| **Phase 5 — Sequencer** | |
| `examples/41_clip_basic.py` | Clip placed at a specific time |
| `examples/42_drum_pattern.py` | Kick + snare drum pattern |
| `examples/43_pattern_loop.py` | Pattern looped 4x |
| `examples/44_tempo_demo.py` | Same pattern at 80, 120, 160 BPM |
| `examples/45_arrangement.py` | Multi-pattern: intro + verse + chorus |
| `examples/46_arpeggio.py` | C major arpeggio via sequencer |
| `examples/47_polyrhythm.py` | 3:4 polyrhythm |
| `examples/48_drone_pattern_overlay.py` | Drone + rhythmic overlay |
| `examples/49_song_structure.py` | 16-bar ABA song form |
| `examples/50_ambient_session_v5.py` | Phase 5 flagship: Session + Sequencer combined |
| **Phase 7 -- Music Theory** | |
| `examples/51_music_theory_demo.py` | Scales, chords, and I-V-vi-IV progression |
| **Phase 8 -- Synthesizers** | |
| `examples/52_fm_bell.py` | FM synthesis bell — harmonic and metallic variants |
| `examples/53_subtractive_bass.py` | Subtractive bass — saw + LPF + filter envelope |
| `examples/54_supersaw_pad.py` | 7-voice supersaw unison Dm7 pad |
| **Phase 9 -- Drums** | |
| `examples/55_808_kit.py` | 808 drum kit showcase + kick-snare-hat pattern |
| `examples/56_lofi_beat.py` | Lo-fi hip-hop beat with swing timing |
| **Phase 10 -- Modulation** | |
| `examples/57_filter_sweep.py` | Lowpass filter sweep (linear + exponential) |
| `examples/58_vibrato_lead.py` | Vibrato at three intensities (subtle, medium, wide) |
| `examples/59_evolving_drone.py` | LFO-modulated drone with filter breathing + tremolo |
| **Phase 11 -- Presets** | |
| `examples/60_preset_showcase.py` | One preset per category (pad, lead, bass, key, texture) |
| `examples/61_blade_runner_pad.py` | Vangelis-style FM pad playing a Dm7 chord |
| **Phase 12 -- Generative** | |
| `examples/62_generative_melody.py` | Random melody over A minor pentatonic with pluck preset |
| `examples/63_ambient_5min.py` | 5-minute generative ambient texture (D minor pentatonic) |

Run any example from the project root:
```bash
python examples/<name>.py
```

---

## Module Structure

```
src/fractal/
├── __init__.py        # public API re-exports
├── constants.py       # SAMPLE_RATE, BIT_DEPTH, MAX_AMPLITUDE
├── signal.py          # Signal utilities: mono/stereo, normalize, pad, trim, mix
├── generators.py      # sine, square, saw, triangle, noise, binaural, load_sample
├── envelopes.py       # ADSR, FadeInOut, SmoothFade, Swell, Gate, Tremolo, AutomationCurve
├── effects.py         # LPF, HPF, BPF, EQ, Reverb, Delay, Distortion, StereoWidth
├── track.py           # Track dataclass (volume, pan, effects, bus routing)
├── mixer.py           # Bus, Session (mixdown, bus routing, export)
├── sequencer.py       # Clip, Pattern, Sequencer (timeline, tempo, loops)
├── music_theory.py    # note_to_hz, scales, chords, progressions (12-TET)
├── synth.py           # FM, additive, wavetable, subtractive, pulse, unison
├── drums.py           # kick, snare, hihat, clap, tom, cymbal, drum_kit
├── modulation.py      # LFO, vibrato, filter sweep, param automation
├── presets.py         # SynthPreset, 20 named presets, get_preset()
├── generative.py      # random_melody, ambient_texture, chord_progression_render
└── export.py          # WAV / FLAC export via soundfile
```

---

## Project Layout

```
fractal/
├── src/fractal/       # library code
├── examples/          # runnable session scripts
├── tests/             # pytest test suite
├── sounds/
│   ├── raw/           # source audio samples
│   └── processed/     # processed/enhanced samples
├── outputs/
│   ├── audio/         # rendered audio files (gitignored)
│   └── renders/       # final mixdowns (gitignored)
├── configs/           # presets, session configs
└── docs/              # decisions log, SWOT, activity log
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

323 tests, all passing (~3s).

---

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1 | Signal generation + export (constants, signal, generators, export) | **DONE** |
| 2 | Envelopes — ADSR, fades, swell, gate, tremolo, automation | **DONE** |
| 3 | Effects — LPF, HPF, BPF, EQ, reverb, delay, distortion, stereo width | **DONE** |
| 4 | Track + Mixer — Track, Bus, Session, pan, bus routing | **DONE** |
| 5 | Sequencer — Clip, Pattern, Sequencer, tempo, loops | **DONE** |
| 7 | Music theory — notes, scales, chords, progressions | **DONE** |
| 8 | Synthesizers — FM, additive, wavetable, subtractive, unison | **DONE** |
| 9 | Drums — kick, snare, hihat, clap, tom, cymbal, drum kits | **DONE** |
| 10 | Modulation — LFO, vibrato, filter sweep, param automation | **DONE** |
| 11 | Presets — 20 named synthesis recipes ("blade_runner_pad") | **DONE** |
| 12 | Generative — random melody, ambient_texture, chord progressions | **DONE** |

---

## Design Principles

- **A script is a session file** — deterministic, versionable, diffable
- **Signal = numpy array** — no wrapper classes, no magic
- **Explicit sample rate** — every function takes `sample_rate` as a parameter
- **No DAW dependency** — runs on any machine with Python 3.11+
- **Offline render only** (v1) — no real-time audio required
