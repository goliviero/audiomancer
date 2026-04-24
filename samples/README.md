# Samples Bank

Acoustic sample library used by Audiomancer for enrichment (granular, loop, layer).
**Audio files are gitignored** — only this README is tracked.

> **Backup reminder** : since audio in `samples/cc0/` and `samples/own/`
> never lands on GitHub, back them up to Proton Drive (or another cloud)
> alongside the rest of your production assets. They are NOT reproducible
> if lost.

## Current inventory

| Path | Source / License | Used by |
|---|---|---|
| `samples/cc0/didgeridoo_C2.wav` | LaSonotheque.fr (CC0-equivalent per their ToS, credit encouraged: "Joseph SARDIN - LaSonotheque.org") | `scripts/53_didgeridoo_real_vs_synth.py` |
| `samples/cc0/fireplace_crackling.wav` | Freesound ID 501417 (Visionear, CC0) | V005 mix scripts (25/26/27/29) + `configs/V005.py` |

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

### Ethnic instruments (single-shot samples)

With `audiomancer.sampler`, a single-note WAV can be played at any pitch
via pitch-shift (high-quality scipy polyphase), and transformed into long
ambient pads via `pitched_pad()` (pitch-shift + paulstretch). You DO NOT
need a full multisample library — one good sample per instrument is enough
for ambient use.

Recommended CC0 / freely downloadable sources:

- **Freesound.org** — filter by License = CC0 + search term. Quality varies,
  check for clean recordings without room noise.
  - Search: `handpan`, `hang drum`, `oud`, `sitar`, `darbuka`,
    `didgeridoo`, `tanpura`, `sarangi`, `kalimba`, `singing bowl`.
- **FreePats** — https://freepats.zenvoid.org/ — CC0 / CC-BY instrument
  patches (soundfonts / samples). Good for sitar, sarangi, some world
  percussion.
- **Philharmonia orchestra** — https://philharmonia.co.uk/resources/sound-samples/
  — CC-BY clean single-note WAV, full chromatic range (attribution required
  in YouTube descriptions).

### Naming convention for multisample support

If you have multiple pitches of the same instrument:

    samples/cc0/handpan_D3.wav
    samples/cc0/handpan_A3.wav
    samples/cc0/handpan_F4.wav

Then in Python:

    from audiomancer.sampler import load_multisample, play_note_multisample
    bank = load_multisample("handpan")
    note = play_note_multisample(bank, target_hz=220.0)  # auto-picks closest

Single-sample still works fine: just drop `handpan_D3.wav` and use
`builders.instrument_sampled` with `source_hz=146.83`.

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
