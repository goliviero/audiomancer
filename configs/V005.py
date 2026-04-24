"""V005 config — C major sus2 / 40 Hz gamma study music.

Reproduction of the V005 production setup in the config-driven format.
Used by scripts/render_stem.py and scripts/render_mix.py.

Historical note: the original V005 shipped via per-video scripts
(scripts/21-29_v005_*.py). Those are preserved as archive. This config
shows the new pattern for V006+.
"""

META = {
    "label": "V005 - C major sus2 / 40 Hz gamma focus",
    "target_lufs": -14.0,
    "sample_rate": 48000,
    "duration": 300,  # 5 min
}

STEMS = {
    "warm_pad": {
        "builder": "pad_alive",
        "params": {
            # C2 + C3 + E3 + G4 (deep, no strident G5)
            "chord": [66.0, 132.0, 165.0, 396.0],
            "intensity": "moderate",
        },
    },
    "arpege_bass": {
        "builder": "arpege_bass",
        "params": {
            # C sus2 palindrome 2 octaves: C-D-G-C-D-G-D-C-G-D
            "arpege": [66.0, 74.25, 99.0, 132.0, 148.5, 198.0,
                       148.5, 132.0, 99.0, 74.25],
            "note_dur": 15.0,
            "xfade": 3.0,
            "hp_hz": 50.0,
            "lp_hz": 500.0,
        },
    },
    "grounding_bass": {
        # Alternative bass (pendulum vs arpege) — kept as option
        "builder": "pendulum_bass",
        "params": {
            "pendulum": [66.0, 99.0],  # C2 <-> G2
            "note_dur": 60.0,
            "xfade": 15.0,
            "lp_hz": 300.0,
        },
    },
    "binaural": {
        "builder": "binaural_beat",
        "params": {
            "carrier_hz": 264.0,  # C4 JI
            "beat_hz": 40.0,      # gamma
            "volume_db": -6.0,
        },
    },
}

MIX = {
    # Which stems to include in the mix, and at what level
    "volumes_db": {
        "warm_pad": -6.0,
        "arpege_bass": -9.0,
        "binaural": -14.0,
        "firecrack": -18.0,
    },
    # Micro-events layer
    "micro_events": [
        {"type": "harmonic_bloom", "rate_per_min": 1.0, "volume_db": -24.0,
         "duration_range": [3.0, 8.0]},
        {"type": "grain_burst", "rate_per_min": 0.3, "volume_db": -28.0,
         "duration_range": [1.0, 3.0]},
        {"type": "overtone_whisper", "rate_per_min": 0.5, "volume_db": -30.0,
         "duration_range": [2.0, 5.0]},
    ],
    # Chord frequencies used by harmonic_bloom / overtone_whisper
    "chord_freqs_for_events": [264.0, 330.0, 396.0],  # C major
    # Long-term 5-min arc envelope
    "density_profile": "random_walk",
    # Optional external sample for the firecrack layer
    "firecrack": {
        "path": "samples/cc0/fireplace_crackling.wav",
        "offset_sec": 30.0,
    },
}
