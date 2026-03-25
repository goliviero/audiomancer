# Audiomancer

> Minimal Python audio scripts for ambient sound design.

Collection of simple, focused Python scripts for generating sounds and textures.
Not a DAW. Not a framework. Just useful scripts for ambient/meditation audio content.

---

## Quick Start

```bash
# 1. Clone and install
git clone git@github.com:goliviero/audiomancer.git
cd audiomancer
pip install -e .

# 2. Generate a binaural beat
python scripts/make_binaural.py
# => output/binaural_alpha_432hz_10min.wav

# 3. Create an ambient drone pad
python scripts/drone_pad.py
# => output/drone_pad_ambient.wav

# 4. Process a field recording
python scripts/process_field.py my_recording.wav --reverb 0.4
# => output/my_recording_processed.wav

# 5. Layer multiple stems
python scripts/layer_stems.py stem1.wav stem2.wav stem3.wav -o output/mix.wav
```

---

## Architecture

```
audiomancer/
├── audiomancer/
│   ├── synth.py         # Waveforms (sine, saw, square), drones, pads
│   ├── binaural.py      # Binaural beats + layered variants
│   ├── effects.py       # Reverb, delay, chorus, compression (pedalboard)
│   ├── layers.py        # Mixing, layering, crossfading
│   └── utils.py         # I/O, normalize, fade, signal helpers
├── scripts/
│   ├── make_binaural.py # Generate binaural beat
│   ├── drone_pad.py     # Create ambient drone
│   ├── process_field.py # Process field recordings
│   └── layer_stems.py   # Mix audio stems
├── tests/               # 44 pytest tests
└── _fractal_backup/     # Full backup of original Fractal audio code
```

---

## Modules

| Module | What it does |
|--------|-------------|
| `synth` | Sine, square, saw, triangle, white/pink noise, drones, detuned pads |
| `binaural` | Stereo binaural beats, layered with pink noise bed |
| `effects` | Scipy filters (LP/HP) + pedalboard effects (reverb, delay, chorus, compression) |
| `layers` | Mix signals, layer at offset, crossfade between stems |
| `utils` | WAV I/O, normalize, fade in/out, mono/stereo conversion |

---

## Dependencies

```
numpy
scipy
soundfile
pedalboard    # Spotify — pro-grade audio effects
```

---

## Tests

```bash
python -m pytest tests/ -v   # 44 tests, <2s
```

---

## Philosophy

- **Scripts > Framework**: each script = one sound or texture. No architecture.
- **Disposable**: scripts are cheap to write, modify, or throw away.
- **Real tools for real work**: use REAPER + Vital for serious DAW work, Audiomancer for quick generation.
- **No AI audio**: no Suno, no AudioCraft, no MusicGen. Original synthesis only.

---

## History

Audiomancer was born from [Fractal](https://github.com/goliviero/fractal), which pivoted from audio to procedural visual generation.
The `_fractal_backup/` directory contains the complete original audio codebase (18 modules, 323 tests, 63 examples).
