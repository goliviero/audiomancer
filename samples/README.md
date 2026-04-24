# Samples Bank

Acoustic sample library used by Audiomancer for enrichment (granular, loop, layer).
**Audio files are gitignored** — only this README is tracked.

## Structure

```
samples/
├── README.md       # this file (tracked)
├── cc0/            # external CC0 / CC-BY sources (gitignored)
└── own/            # personal recordings — P-45 piano, F310 guitar, field (gitignored)
```

## Loading in scripts

```python
from audiomancer.utils import load_sample

sig = load_sample("piano_C3_mezzo", target_sr=48000)
# Searches samples/own/ first, falls back to samples/cc0/
# Auto-resamples if source SR differs from target_sr
# Returns stereo, normalized to -1 dBFS
```

## Naming convention

`{instrument}_{note}_{articulation}[_extra].wav`

Examples :
- `piano_C3_mezzo.wav`  — piano note C3, mezzo-forte
- `piano_C3_felt.wav`   — felt mod piano
- `guitar_Em_drone.wav` — E minor sustained chord, guitar archet/bow
- `guitar_A2_pluck.wav` — A2 pinched string
- `strings_Cmaj_sustain.wav` — string section Cmaj long note

## Recommended CC0 sources

### Piano (felt / classical)
- **Salamander Grand Piano** (CC0) — https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
  - Concert grand, multi-vel, stereo
- **University of Iowa MIS** — https://theremin.music.uiowa.edu/MISpiano.html
  - Single-note isolated samples, clean recording
- **Piano Book Felt Piano** — https://piano.book/felt (CC-BY, attribution required)

### Strings
- **VSCO 2 Community Edition** (CC0) — https://vis.versilstudios.com/vsco-community.html
  - Violin, viola, cello, bass — short + sustained
- **Sonatina Symphonic Orchestra** (CC-Sampling+) — https://sso.mattiaswestlund.net/
  - Full orchestral sections, mellotron-adjacent

### Field / ambient
- **Freesound.org** with CC0 filter — e.g., current `firecrack` in `inbox/`
- **Radio Aporee** (CC0 recordings) — https://radio.aporee.org/

## Your own recordings

Drop them into `samples/own/`. Recommendation :
- 48 kHz / 24-bit if possible (but `load_sample` resamples anyway)
- 5-30 seconds per take — long enough for granular, short enough to store
- Record several takes of the same note at different dynamics — variety helps
- Commit only the README update, never the audio

## License / attribution

CC0 samples : no attribution required (but nice to note the source in commits).
CC-BY : attribution required in YouTube description.
Own recordings : self-licensed, free to use.
