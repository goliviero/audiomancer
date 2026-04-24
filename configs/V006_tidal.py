"""V006_tidal — Muladhara variant with tidal foundation.

Same palette as V006 but the FOUNDATION is gated by a sparse envelope:
bass present for 20-60s sections, retracts to 30% for 20-60s, then returns.
Creates the "basses qui se coupent et reviennent" dynamic.

Mid body enriched with:
    - 3 didgeridoo events (instead of 2)
    - 1 harmonic_bloom micro-event per minute on C-minor chord tones
"""

C2 = 65.41
C3 = 130.81
C4 = 261.63
Eb3 = 155.56    # minor 3rd above C3 (C-minor color)
Eb4 = 311.13
G3 = 196.00
G4 = 392.00

META = {
    "label": "V006_tidal - Muladhara (tidal bass + mid blooms)",
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
            "event_count": 3,              # was 2 — more presence
            "event_dur_range": (20.0, 30.0),
            "fade_in_sec": 15.0,
            "fade_out_sec": 18.0,
            "pitch_drift_cents": 45.0,     # wider drift for more variation
            "stereo_width": 0.20,
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
        "foundation": -6.0,
        "ochre_mid":  -8.0,
        "didgeridoo": -12.0,
        "subliminal": -26.0,
    },
    # Foundation gated in/out — sections 20-60s with smooth transitions
    # (density_profile "sparse" alternates between ~1.0 and ~0.3).
    "stem_envelopes": {
        "foundation": "sparse",
    },
    # Mid blooms every ~60s on C-minor chord tones
    "micro_events": [
        {"type": "harmonic_bloom", "rate_per_min": 1.0, "volume_db": -26.0,
         "duration_range": [5.0, 10.0]},
    ],
    "chord_freqs_for_events": [C3, Eb3, G3, C4, Eb4],
}
