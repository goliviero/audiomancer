"""V006_cavern — Muladhara variant: breathing foundation + cave ambience.

The foundation SWELLS and RECEDES over the 5min (breathing envelope) —
very slow tidal movement, more geological than the tidal variant's
sparse gating.

Adds a constant low-level brown noise wash to evoke cave-wall ambience.
Didgeridoo presence increased to 4 events. Overtone whispers fill upper-mid.
"""

C2 = 65.41
C3 = 130.81
C4 = 261.63
Eb3 = 155.56
Eb4 = 311.13
G3 = 196.00
G4 = 392.00

META = {
    "label": "V006_cavern - Muladhara (breathing foundation + cave air)",
    "target_lufs": -20.0,
    "sample_rate": 48000,
    "duration": 300,
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
            "amp_mod_cycle_sec": 300.0 / 14,
            "amp_mod_depth_db": 1.0,
            "reverb_room": 0.55,
            "reverb_wet": 0.18,
        },
    },
    "ochre_mid": {
        "builder": "ochre_pad",
        "params": {
            "chord": [(C3, 0.0), (G3, -20.0), (C4, -22.0), (G4, -26.0)],
            "voices": 4,
            "detune_cents": 10.0,
            "lp_hz": 4000.0,
            "sat_drive": 1.6,
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
            "event_count": 4,              # 4 events — busier
            "event_dur_range": (15.0, 22.0),
            "fade_in_sec": 12.0,
            "fade_out_sec": 14.0,
            "pitch_drift_cents": 40.0,
            "stereo_width": 0.25,
            "hp_hz": 40.0,
            "lp_hz": 4000.0,
            "reverb_room": 0.7,            # bigger room — cavern
            "reverb_wet": 0.45,
        },
    },
    "cave_air": {
        # Brown noise wash heavily lowpass-filtered — air moving through stone
        "builder": "texture",
        "params": {
            "texture_name": "noise_wash",
            "color": "brown",
        },
    },
    "subliminal": {
        "builder": "subliminal_sine",
        "params": {
            "freq": 60.0,
            "tremolo_cycle_sec": 300.0 / 12,
            "tremolo_depth_db": 2.0,
        },
    },
}

MIX = {
    "volumes_db": {
        "foundation": -6.0,
        "ochre_mid":  -8.0,
        "didgeridoo": -12.0,
        "cave_air":   -22.0,    # constant low bed
        "subliminal": -26.0,
    },
    # Foundation breathes — slow raised cosine from 0.3 to 1.0 to 0.3 across 5min
    "stem_envelopes": {
        "foundation": "breathing",
    },
    # Overtone whispers in upper mids — sparse, felt-not-heard breath
    "micro_events": [
        {"type": "overtone_whisper", "rate_per_min": 1.0, "volume_db": -28.0,
         "duration_range": [4.0, 8.0]},
    ],
    "chord_freqs_for_events": [C3, Eb3, G3, C4, Eb4],
}
