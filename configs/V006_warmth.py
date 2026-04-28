"""V006_warmth — ember_v2 + continuous mid warmth via evolving_drone layer.

# FROZEN — V006_warmth livré en production, ne plus modifier.

Same base as V006_ember_v2 (dual ochre C/Eb, opposite-phase envelopes,
foundation + didgeridoo + subliminal). Adds:

    - warm_drone : texture(evolving_drone) at A3 = 220 Hz. Natural harmonic
      roll-off from the drone primitive, LP 2000 Hz internally. Parks the
      mid register permanently in the vocal-ooh range without any pure
      sinusoidal overtones.
    - micro_events REMOVED. The mid body now comes from the warm_drone
      continuity rather than sporadic blooms, so no synthesized high tones.

Sonic result: denser, rounder mids, no air sparkle. Still no brightness.
"""

C2 = 65.41
C3 = 130.81
C4 = 261.63
Eb3 = 155.56
Eb4 = 311.13
Bb3 = 233.08
A3 = 220.00     # warm vocal "ooh" register — between C3 and Eb3
G3 = 196.00
G4 = 392.00

META = {
    "label": "V006_warmth - Muladhara ember + continuous mid drone",
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
    "warm_drone": {
        "builder": "texture",
        "params": {
            "texture_name": "evolving_drone",
            "frequency": A3,
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
        "warm_drone":  -14.0,
        "didgeridoo": -12.0,
        "subliminal": -26.0,
    },
    "stem_envelopes": {
        "ochre_mid":   "arc",
        "ochre_minor": "breathing",
        # warm_drone is continuous — no envelope
    },
    # no micro_events
}
