"""V006 config — Muladhara Root Chakra (grounding / sleep meditation).

Brief: 5:00 ambient loop, C minor anchored on C2 (65.41 Hz) and C3 (130.81 Hz).
48kHz / 24-bit WAV. Integrated LUFS -20 (gated, BS.1770), true peak ceiling
-3 dBTP. Pre-fade 1.5s in/out.

Palette: saffron, ochre, burnt sienna, charcoal. Earth (prithvi), standstill.
Reference mood: Stars of the Lid at half speed, Biosphere's Substrata but more
mineral.

Layers (max 3 simultaneous):
    - foundation   : tectonic C2+C3 detuned drone, LP ~500Hz, 21.4s breath
    - ochre_mid    : C3 root + G3 / G4 / C4 hinted voices, LP 4kHz, tape sat
    - didgeridoo   : 2 sparse sample-based events across 5min (18-25s each)
    - subliminal   : 60Hz sine -26dB, 25s tremolo (felt more than heard)

Rendered via the V006+ config-driven pipeline:
    python scripts/render_mix.py --config V006
    python scripts/render_stem.py --config V006 --stem ochre_mid
"""

# Pitch anchors (equal temperament, C minor)
C2 = 65.41
C3 = 130.81
C4 = 261.63
G3 = 196.00
G4 = 392.00

META = {
    "label": "V006 - Muladhara Root Chakra",
    "target_lufs": -20.0,
    "sample_rate": 48000,
    "duration": 300,
    # V006+ ambient-specific options (see audiomancer.mastering.ambient_master_chain)
    "master_mode": "ambient",
    "ceiling_dbtp": -3.0,
    "pre_fade_sec": 1.5,
    "bit_depth": 24,
}

STEMS = {
    "foundation": {
        "builder": "foundation_drone",
        "params": {
            "freqs": [C2, C3],
            "detune_cents": 3.0,
            "lp_hz": 500.0,
            # 14 exact breath cycles across 300s for cleaner loop closure
            "amp_mod_cycle_sec": 300.0 / 14,
            "amp_mod_depth_db": 1.0,
            "reverb_room": 0.55,
            "reverb_wet": 0.18,
        },
    },
    "ochre_mid": {
        "builder": "ochre_pad",
        "params": {
            # Fuller body without breaking "no bright harmonics":
            #   C3 root, G3 fifth, C4 octave, G4 twelfth (all at or below
            #   -20dB). Keeps the brief's monomodal C-minor color; no new
            #   note class, just harmonic reinforcement in upper mids.
            "chord": [
                (C3,  0.0),
                (G3, -20.0),
                (C4, -22.0),
                (G4, -26.0),
            ],
            "voices": 4,
            "detune_cents": 10.0,
            "lp_hz": 4000.0,           # was 3000 — more upper-mid body
            "sat_drive": 1.6,          # was 1.1 — warmer harmonic content
            # 11 exact breath cycles across 300s
            "breath_cycle_sec": 300.0 / 11,
            "breath_depth": 0.04,
            "reverb_room": 0.70,
            "reverb_wet": 0.35,
        },
    },
    "didgeridoo": {
        "builder": "sparse_sample_events",
        "params": {
            "source_path": "samples/cc0/didgeridoo_C2.wav",
            "source_hz": C2,
            "target_hz": C2,
            "event_count": 2,
            "event_dur_range": (18.0, 25.0),
            "fade_in_sec": 18.0,
            "fade_out_sec": 20.0,
            "pitch_drift_cents": 30.0,
            "stereo_width": 0.15,
            "hp_hz": 40.0,
            "lp_hz": 4000.0,
            "reverb_room": 0.6,
            "reverb_wet": 0.30,
        },
    },
    "subliminal": {
        "builder": "subliminal_sine",
        "params": {
            "freq": 60.0,
            # 12 exact tremolo cycles across 300s
            "tremolo_cycle_sec": 300.0 / 12,
            "tremolo_depth_db": 2.0,
        },
    },
}

MIX = {
    "volumes_db": {
        "foundation": -6.0,
        "ochre_mid":  -8.0,   # was -10: ochre carries more of the mid body
        "didgeridoo": -12.0,
        "subliminal": -26.0,
    },
}
