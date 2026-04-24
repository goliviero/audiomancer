"""V011 config — Manipura Solar Plexus Chakra (528Hz / 10Hz alpha).

Brief: 5:00 ambient loop, E phrygian warm pad anchor + golden mid pad + brass
sustain sparse signature (2 layers async) + ember crackle floor + 10Hz alpha
binaural (différencié de V005 qui était 40Hz gamma focus).

Palette: gold, saffron, rust, dark amber. Fire element (agni), sustained
inner warmth, not aggression.

Layers (5 simultaneous):
    - foundation  : E2+B2 detuned drone, LP 500Hz, 24s breath
    - golden_mid  : E3 + G#3 major third + B3 / E4 voices, LP 4.2kHz, tape sat 1.65
    - binaural    : 264Hz carrier / 10Hz alpha (sustained calm power)
    - brass_a/b   : 2 sparse brass sustain layers async
    - ember_floor : fireplace crackling continuous very low (from existing sample)

Samples:
    - `samples/cc0/fireplace_crackling.wav` EXISTS (V005 used it)
    - `samples/cc0/brass_sustain_E3.wav` — TODO source via Freesound CC0
      (recherche: "brass sustain E", "french horn sustain", "trumpet drone CC0")

Rendered:
    python scripts/render_mix.py --config V011
"""

# Pitch anchors (equal temperament, E-based)
E2  =  82.41
B2  = 123.47
E3  = 164.81
G_3 = 207.65  # G#3 major third pour warmth solaire (pas phrygian minor third)
B3  = 246.94
E4  = 329.63

META = {
    "label": "V011 - Manipura Solar Plexus Chakra",
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
            "freqs": [E2, B2],
            "detune_cents": 3.0,
            "lp_hz": 500.0,
            "amp_mod_cycle_sec": 300.0 / 12,  # ~25s, 12 cycles
            "amp_mod_depth_db": 1.0,
            "reverb_room": 0.60,
            "reverb_wet": 0.22,
        },
    },
    "golden_mid": {
        "builder": "ochre_pad",
        "params": {
            # E3 root + G#3 major third (warmth solaire, pas phrygian minor),
            # B3 fifth, E4 octave. Fire element = lit-up warmth, not dark.
            "chord": [
                (E3,   0.0),
                (G_3,-18.0),
                (B3, -22.0),
                (E4, -24.0),
            ],
            "voices": 4,
            "detune_cents": 10.0,
            "lp_hz": 4200.0,
            "sat_drive": 1.65,
            "breath_cycle_sec": 300.0 / 12,  # 25s, 12 cycles
            "breath_depth": 0.04,
            "reverb_room": 0.72,
            "reverb_wet": 0.38,
        },
    },
    "binaural": {
        "builder": "binaural_beat",
        "params": {
            # 264Hz = 528/2 Solfeggio. 10Hz alpha (calm confidence),
            # PAS 40Hz gamma (celui de V005, on différencie Manipura chakra
            # de "528Hz focus" par le brain state).
            "carrier_hz": 264.0,
            "beat_hz": 10.0,
            "volume_db": -22.0,
        },
    },
    "brass_a": {
        "builder": "sparse_sample_events",
        "params": {
            "source_path": "samples/cc0/brass_sustain_E3.wav",  # TODO: source
            "source_hz": E3,
            "target_hz": E3,
            "event_count": 2,
            "event_dur_range": (18.0, 24.0),
            "fade_in_sec": 20.0,
            "fade_out_sec": 18.0,
            "pitch_drift_cents": 15.0,
            "stereo_width": 0.12,
            "hp_hz": 100.0,
            "lp_hz": 6000.0,
            "reverb_room": 0.75,
            "reverb_wet": 0.40,
        },
    },
    "brass_b": {
        "builder": "sparse_sample_events",
        "params": {
            # Même sample, timing décalé → asynchrone avec brass_a
            "source_path": "samples/cc0/brass_sustain_E3.wav",  # TODO: source
            "source_hz": E3,
            "target_hz": E3,
            "event_count": 1,
            "event_dur_range": (22.0, 30.0),
            "fade_in_sec": 22.0,
            "fade_out_sec": 20.0,
            "pitch_drift_cents": 25.0,
            "stereo_width": 0.16,
            "hp_hz": 100.0,
            "lp_hz": 6000.0,
            "reverb_room": 0.78,
            "reverb_wet": 0.42,
        },
    },
    "ember_floor": {
        "builder": "sparse_sample_events",
        "params": {
            # fireplace_crackling.wav existe (V005). Continuous = 1 event sur
            # toute la durée avec long fade. Niveau très bas (tactile floor).
            "source_path": "samples/cc0/fireplace_crackling.wav",
            "source_hz": 100.0,   # pas de pitch naturel
            "target_hz": 100.0,   # pas de shift
            "event_count": 1,
            "event_dur_range": (295.0, 300.0),
            "fade_in_sec": 10.0,
            "fade_out_sec": 15.0,
            "pitch_drift_cents": 0.0,
            "stereo_width": 0.08,
            "hp_hz": 80.0,
            "lp_hz": 8000.0,
            "reverb_room": 0.40,
            "reverb_wet": 0.15,
        },
    },
}

MIX = {
    "volumes_db": {
        "foundation":   -8.0,
        "golden_mid":   -8.0,
        "binaural":    -22.0,
        "brass_a":     -15.0,
        "brass_b":     -17.0,
        "ember_floor": -22.0,
    },
}
