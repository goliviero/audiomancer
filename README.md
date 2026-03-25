# Audiomancer

> Minimal Python audio scripts for ambient sound design.

Audiomancer is a collection of simple, focused Python scripts for generating ambient/meditation audio.
Not a DAW. Not a framework. Just useful scripts that produce WAV files.

Born from [Fractal](https://github.com/goliviero/fractal), which pivoted from audio to procedural visual generation.
Audiomancer inherits the audio synthesis code and strips away the DAW architecture.

**Current production use:** generating 100% original audio for [Akasha Portal](https://github.com/goliviero/akasha-portal) videos,
replacing AI-generated stems (Suno) with pure synthesis to avoid YouTube content ID issues.

---

## Quick Start

```bash
# Install
git clone git@github.com:goliviero/audiomancer.git
cd audiomancer
pip install -e .

# Generate 30 minutes of ambient meditation audio
python scripts/06_akasha_v003.py
# => output/akasha_v003_master.wav (stereo, 44100 Hz, -14 LUFS)
# => output/stems/ (individual layers)
```

---

## Examples

### 1. Generate a binaural beat

```python
from audiomancer.binaural import binaural
from audiomancer.utils import fade_in, fade_out, export_wav

signal = binaural(carrier_hz=432.0, beat_hz=10.0, duration_sec=600)
signal = fade_in(signal, 5.0)
signal = fade_out(signal, 10.0)
export_wav(signal, "output/binaural.wav")
```

### 2. Create an ambient drone

```python
from audiomancer.synth import drone
from audiomancer.effects import reverb_cathedral
from audiomancer.utils import mono_to_stereo, export_wav

signal = drone(136.1, 120.0)  # Om frequency, 2 minutes
signal = mono_to_stereo(signal)
signal = reverb_cathedral(signal)
export_wav(signal, "output/om_drone.wav")
```

### 3. Layer multiple stems

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

## Architecture

```
audiomancer/
├── audiomancer/
│   ├── synth.py         # Waveforms, drones, pads, noise
│   ├── binaural.py      # Binaural beats + presets
│   ├── effects.py       # Reverb, delay, chorus, compression, filters
│   ├── layers.py        # Mixing, layering, crossfading, LUFS normalization
│   ├── field.py         # Field recording processing
│   └── utils.py         # I/O, normalize, fade, signal helpers
├── scripts/
│   ├── 01_binaural_432hz.py   # 10-min binaural beat
│   ├── 02_drone_pad.py        # Ambient drone pad
│   ├── 03_piano_reverb.py     # WAV + cathedral reverb
│   ├── 04_field_processing.py # Field recording cleanup
│   ├── 05_layer_akasha.py     # 5-min preview mix
│   └── 06_akasha_v003.py      # PRODUCTION: 30-min Akasha audio
├── tests/               # 70 pytest tests
├── samples/             # Source samples (field recordings, etc.)
├── output/              # Generated audio (gitignored)
└── _fractal_backup/     # Original Fractal audio codebase
```

---

## Modules

| Module | What it does |
|--------|-------------|
| `synth` | Sine, square, saw, triangle, white/pink/brown noise, drones, chord pads |
| `binaural` | Stereo binaural beats with presets (theta, alpha, delta, solfeggio) |
| `effects` | Scipy filters (LP/HP) + pedalboard effects (reverb, delay, chorus, compression) |
| `layers` | Mix signals, layer stems, crossfade, loop, LUFS normalization |
| `field` | Field recording processing: cleanup, noise gate, reverb, fades |
| `utils` | WAV I/O, normalize, fade in/out, mono/stereo conversion, trim |

---

## Dependencies

```
numpy, scipy, soundfile, pedalboard
```

Install: `pip install -e .` or `pip install numpy scipy soundfile pedalboard`

---

## Tests

```bash
python -m pytest tests/ -v   # 70 tests, <2s
```

---

## Philosophy

- **Scripts > Framework**: each script = one sound. No architecture.
- **Disposable**: scripts are cheap to write, modify, or throw away.
- **No AI audio**: no Suno, no AudioCraft. Original synthesis only.
- **Real tools for real work**: use REAPER + Vital for serious DAW work.
