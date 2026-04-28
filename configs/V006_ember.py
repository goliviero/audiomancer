"""V006_ember — Muladhara variant with layered ochre + active mids.

# FROZEN — V006_ember livré en production, ne plus modifier.

Same foundation / didgeridoo / subliminal as V006 but the MID is split
in two voices that fade independently:
    - ochre_mid  (C-major color) with an "arc" envelope (builds then recedes)
    - ochre_minor (Eb color) with a "breathing" envelope (opposite phase)

The two mids never reach peak together: when one recedes the other rises,
creating continuous harmonic movement in the C-minor space.

Plus harmonic_bloom micro-events for air in upper mids.
"""

C2 = 65.41
C3 = 130.81
C4 = 261.63
Eb3 = 155.56
Eb4 = 311.13
Bb3 = 233.08    # minor 7th (Eb's 5th)
G3 = 196.00
G4 = 392.00

META = {
    "label": "V006_ember - Muladhara (layered ochre + active mids)",
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
        # Bright ochre — C color (root + 5th + octave)
        "builder": "ochre_pad",
        "params": {
            "chord": [(C3, 0.0), (G3, -20.0), (C4, -24.0), (G4, -28.0)],
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
    "ochre_minor": {
        # Minor ochre — Eb color (minor 3rd + 5th = full C-minor triad with root)
        "builder": "ochre_pad",
        "params": {
            "chord": [(Eb3, 0.0), (Bb3, -20.0), (Eb4, -24.0)],
            "voices": 3,
            "detune_cents": 12.0,
            "lp_hz": 3500.0,
            "sat_drive": 1.4,
            "breath_cycle_sec": 300.0 / 9,     # slightly different period
            "breath_depth": 0.05,
            "reverb_room": 0.75,
            "reverb_wet": 0.40,
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
            "tremolo_cycle_sec": 300.0 / 12,
            "tremolo_depth_db": 2.0,
        },
    },
}

MIX = {
    "volumes_db": {
        "foundation":  -6.0,
        "ochre_mid":   -9.0,
        "ochre_minor": -11.0,    # quieter — it's a color, not a main voice
        "didgeridoo": -12.0,
        "subliminal": -26.0,
    },
    # Ochres breathe in opposite phases: arc (builds to center, recedes) vs
    # breathing (recedes at center, builds at edges). Never both peak together.
    "stem_envelopes": {
        "ochre_mid":   "arc",
        "ochre_minor": "breathing",
    },
    # Harmonic blooms on full C-minor tones every ~45s
    "micro_events": [
        {"type": "harmonic_bloom", "rate_per_min": 1.3, "volume_db": -24.0,
         "duration_range": [4.0, 9.0]},
        {"type": "overtone_whisper", "rate_per_min": 0.5, "volume_db": -30.0,
         "duration_range": [3.0, 6.0]},
    ],
    "chord_freqs_for_events": [C3, Eb3, G3, Bb3, C4, Eb4],
}
