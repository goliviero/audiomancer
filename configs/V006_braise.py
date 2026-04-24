"""V006_braise — ember_v2 + double mid body via breathing_pad layer.

Same base as V006_ember_v2. Adds a second pad voice built from a different
algorithm (texture breathing_pad: chord_pad + chorus + reverb) so the mid
register has two DISTINCT pads stacking: one tape-saturated (ochre), one
chorused/verbed (breath_pad). Doubles the mid density without doubling the
timbral character.

Micro-events kept but very sparse (0.5/min) and strictly on the low chord
pool so bloom multipliers cap at 932 Hz.
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
    "label": "V006_braise - Muladhara ember + second pad (chorused)",
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
    "breath_pad": {
        # Different algorithm (chord_pad + chorus + reverb vs tape_sat)
        # for timbral contrast with the ochre voices.
        "builder": "texture",
        "params": {
            "texture_name": "breathing_pad",
            "frequencies": [C3, Eb3, G3],   # C-minor triad
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
        "ochre_minor": -12.0,
        "breath_pad":  -14.0,
        "didgeridoo": -12.0,
        "subliminal": -26.0,
    },
    "stem_envelopes": {
        "ochre_mid":   "arc",
        "ochre_minor": "breathing",
        # breath_pad continuous — it's the unifying glue
    },
    # Very sparse blooms, low-register only (Bb3 * 4 = 932 Hz ceiling)
    "micro_events": [
        {"type": "harmonic_bloom", "rate_per_min": 0.5, "volume_db": -26.0,
         "duration_range": [6.0, 10.0]},
    ],
    "chord_freqs_for_events": [C3, Eb3, G3, Bb3],
}
