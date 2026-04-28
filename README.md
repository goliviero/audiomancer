# Audiomancer

> Minimal Python audio scripts for ambient sound design.

Not a DAW. Not a framework. Just useful scripts that produce WAV files.

Born from [Fractal](https://github.com/goliviero/fractal), which pivoted to procedural visual generation.
**Current production use:** generating 100% original audio for [Akasha Portal](https://github.com/goliviero/akasha-portal) videos,
replacing AI-generated stems (Suno) with pure synthesis.

---

## Quick Start

```bash
git clone git@github.com:goliviero/audiomancer.git
cd audiomancer
pip install -e .

# Config-driven (V006+ pattern)
python scripts/render_mix.py --config V005 --preview
# => output/V005/V005_mix_preview.wav

# One-shot script (V003 legacy)
python scripts/06_akasha_v003.py
# => output/akasha_v003_master.wav (stereo, 44100 Hz, -14 LUFS)
```

---

## Examples

### Binaural beat

```python
from audiomancer.binaural import binaural
from audiomancer.utils import fade_in, fade_out, export_wav

signal = binaural(carrier_hz=432.0, beat_hz=10.0, duration_sec=600)
signal = fade_in(signal, 5.0)
signal = fade_out(signal, 10.0)
export_wav(signal, "output/binaural.wav")
```

### Ambient drone

```python
from audiomancer.synth import drone
from audiomancer.effects import reverb_cathedral
from audiomancer.utils import mono_to_stereo, export_wav

signal = drone(136.1, 120.0)  # Om frequency, 2 minutes
signal = mono_to_stereo(signal)
signal = reverb_cathedral(signal)
export_wav(signal, "output/om_drone.wav")
```

### Layer multiple stems

```python
from audiomancer.synth import chord_pad, pink_noise
from audiomancer.binaural import binaural
from audiomancer.layers import layer, normalize_lufs
from audiomancer.utils import mono_to_stereo, export_wav
import numpy as np

pad = mono_to_stereo(chord_pad([261.63, 329.63, 392.0], 60.0))
beats = binaural(200.0, 4.0, 60.0)
noise = np.column_stack([pink_noise(60.0), pink_noise(60.0)])

master = layer([pad, beats, noise], volumes=[0.5, 0.3, 0.1])
master = normalize_lufs(master, target_lufs=-14.0)
export_wav(master, "output/layered_mix.wav")
```

---

## Modules

| Module | What it does |
|--------|-------------|
| `synth` | Sine, square, saw, triangle, noises, drones, chord pads (w/ seeded jitter), granular (w/ pitch_curve), **karplus_strong** (plucked), **bowed_string** (cello/violin) |
| `binaural` | Stereo binaural beats, 9 presets (theta/alpha/delta/solfeggio/om + beta/SMR/high_gamma) |
| `effects` | Scipy filters (LP/HP) + pedalboard (reverb, delay, chorus, compression) + `delay_pingpong` |
| `ir_reverb` | Convolution reverb: `load_ir`, `convolve_reverb`, 4 synthetic presets (room/hall/cathedral/plate) |
| `sidechain` | Envelope follower + ducking compressor — make pads breathe when chimes emerge |
| `instruments` | Synthetic ethnic: didgeridoo, handpan, oud, sitar, derbouka_hit + derbouka_pattern |
| `sampler` | Load any CC0 sample + pitch-shift (polyphase) + `pitched_pad` (pitch + paulstretch) + multisample loader |
| `layers` | Mix signals, layer stems, crossfade, LUFS normalization, **suggest_eq_cuts** (masking detector) |
| `field` | Field recording processing: cleanup, noise gate, reverb, fades |
| `utils` | WAV I/O (auto-resample), normalize, fade in/out, mono/stereo conversion, `load_sample()` |
| `modulation` | LFOs, drift, evolving_lfo, `multi_lfo` (stack), `random_walk` (OU bounded), filter sweep |
| `textures` | 9 evolving ambient presets with registry + generate() dispatcher |
| `compose` | Fade envelopes, tremolo, stitch, make_loopable, `density_profile` (5-min arc) |
| `stochastic` | `scatter_events` (textures) + `micro_events` (typed: bloom/grain/whisper) + `micro_silence_env` |
| `mastering` | `master_chain` (highpass/mono-bass/soft_clip cascade/limiter), LUFS-safe |
| `saturation` | `tape_saturate` (asymmetric), `tape_hiss` (subliminal pink noise), `vinyl_wow` (pitch flutter) |
| `builders` | Parametric stem generators for config-driven render_stem/render_mix (pad_alive, arpege_bass, binaural_beat, pendulum_bass, texture, piano_processed) |
| `piano_presets` | 3 presets (bass_drone, mid_pad, sparse_notes) for raw piano WAVs — shared by CLI + builder |
| `quick` | One-liner API: q.drone, q.pad, q.binaural, q.texture, q.mix, q.save |
| `spectral` | FFT processing: freeze, blur, pitch shift, spectral gate, morph, **paulstretch** (extreme time-stretch) |
| `viz` | matplotlib PNG helpers (waveform / spectrum / combo) — optional `[viz]` extra |
| `harmony` | Scales, just intonation, Pythagorean tuning, chord generators, **arpeggio_from_chord** |
| `spatial` | Pan, auto-pan, stereo width, mid/side, Haas effect, rotate |
| `envelope` | ADSR (linear/exponential), AR, multi-segment, breathing, swell, gate patterns |

---

## Dependencies

```
numpy, scipy, soundfile, pedalboard
```

Install: `pip install -e .` or `pip install numpy scipy soundfile pedalboard`

**Optional (piano workflow):**
- `mido + python-rtmidi` (Python) — MIDI capture from USB keyboard
- `FluidSynth` (system binary) — offline MIDI -> WAV rendering with SoundFont
  - macOS: `brew install fluidsynth`
  - Windows: `winget install FluidSynth.FluidSynth`
  - Linux: `apt install fluidsynth`

**Optional (PNG visualization):**
- `matplotlib` via `pip install audiomancer[viz]`

---

## Piano Workflow

Capture from a USB-MIDI keyboard (e.g. Yamaha P45), render with SoundFont offline,
process into loopable ambient stems. Pure CLI, no DAW, no realtime VST.

```bash
# 1. Record MIDI from keyboard (Ctrl+C to stop)
python scripts/piano/record_piano.py --output recordings/pad.mid

# 2. Render MIDI -> WAV via FluidSynth + SoundFont
python scripts/piano/render_midi.py \
  --midi recordings/pad.mid \
  --soundfont assets/soundfonts/piano.sf2 \
  --output raw/pad.wav

# 3. Apply Audiomancer chain (bass_drone / mid_pad / sparse_notes presets)
python scripts/piano/process_piano.py \
  --input raw/pad.wav --preset mid_pad --output stems/pad_mid.wav
```

Full doc: [scripts/piano/README.md](scripts/piano/README.md).

---

## Tests

```bash
python -m pytest tests/ -v   # 489 tests passing
```

---

## Philosophy

- **Scripts > Framework**: lib provides primitives, scripts orchestrate. No hidden magic.
- **Disposable**: per-video scripts are archived post-release (V004, V005). V006+ uses `configs/` + generic renderers.
- **No AI audio**: no Suno, no AudioCraft. Original synthesis + (optional) MIDI from real keyboards.
- **Loop-safe**: every stem is designed for seamless 3h ffmpeg loops — multi-scale LFOs + random_walk modulation break the "tell" of repetition.
