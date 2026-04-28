"""V006_ember_v2 — fixes the piercing high sines of V006_ember.

# FROZEN — V006_ember_v2 livré en production, ne plus modifier.

The v1 mix used overtone_whisper micro-events, which generate pure sine
waves at 5x/7x/9x the fundamental. With Eb4 / C4 in the event chord pool,
that produced 1300-2800Hz sinusoids that cut through the ambient bed as
ear-stabbing pure tones at ~45s, 1min, 4:45.

Fix:
    - drop overtone_whisper entirely (it's always in the sharp register)
    - restrict the bloom chord pool to [C3, Eb3, G3, Bb3] (no C4/Eb4)
      so that harmonic_bloom multipliers (x2,3,4) land at most at 932 Hz
      (Bb3 x4) — firmly in the mid register, never piercing.

Everything else identical to V006_ember.
"""

C2 = 65.41
C3 = 130.81
C4 = 261.63
Eb3 = 155.56
Eb4 = 311.13
Bb3 = 233.08
G3 = 196.00
G4 = 392.00

META = {
    "label": "V006_ember_v2 - Muladhara ember (piercing highs removed)",
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
        "builder": "ochre_pad",
        "params": {
            "chord": [(Eb3, 0.0), (Bb3, -20.0), (Eb4, -24.0)],
            "voices": 3,
            "detune_cents": 12.0,
            "lp_hz": 3500.0,
            "sat_drive": 1.4,
            "breath_cycle_sec": 300.0 / 9,
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
        "ochre_minor": -11.0,
        "didgeridoo": -12.0,
        "subliminal": -26.0,
    },
    "stem_envelopes": {
        "ochre_mid":   "arc",
        "ochre_minor": "breathing",
    },
    # Only harmonic_bloom (no overtone_whisper). Chord pool caps bloom at
    # Bb3 * 4 = 932 Hz — upper-mid, not piercing.
    "micro_events": [
        {"type": "harmonic_bloom", "rate_per_min": 1.0, "volume_db": -24.0,
         "duration_range": [5.0, 10.0]},
    ],
    "chord_freqs_for_events": [C3, Eb3, G3, Bb3],
}
