"""V008 config — Svadhisthana Sacral Chakra (417Hz / 6Hz theta).

Brief: 5:00 ambient loop, D minor flowing pad + warm amber mid + singing bowl
sparse signature (2 layers async) + 6Hz theta binaural. 48kHz / 24-bit WAV,
LUFS -20 gated, ceiling -3dBTP, pre-fade 1.5s.

Palette: vermillion, orange, amber, umber. Water (apas), flowing standstill.
Reference mood: Stars of the Lid meets Hiroshi Yoshimura, singing bowls at
a temple before dawn.

Layers (max 4 simultaneous):
    - foundation   : D2+A2+D3 detuned drone, LP 450Hz, 22.5s breath
    - amber_mid    : D3 root + A3 / F3 / D4 voices, LP 4kHz, tape sat 1.6
    - binaural     : 208Hz carrier / 6Hz theta, -24dB
    - bowl_a/b     : 2 sparse singing bowl layers async (event_count/jitter varied)

TODO: source `samples/cc0/singing_bowl_A3.wav` via Freesound CC0 avant render.

Rendered:
    python scripts/render_mix.py --config V008
"""

# Pitch anchors (equal temperament, D minor)
D2 = 73.42
A2 = 110.00
D3 = 146.83
F3 = 174.61
A3 = 220.00
D4 = 293.66

META = {
    "label": "V008 - Svadhisthana Sacral Chakra",
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
            "freqs": [D2, A2, D3],
            "detune_cents": 3.0,
            "lp_hz": 450.0,
            "amp_mod_cycle_sec": 300.0 / 13,  # ~23s, 13 cycles clean loop closure
            "amp_mod_depth_db": 1.0,
            "reverb_room": 0.60,
            "reverb_wet": 0.20,
        },
    },
    "amber_mid": {
        "builder": "ochre_pad",
        "params": {
            # D minor: D3 root, A3 fifth, F3 minor third (hinted -22dB for flowing
            # softness), D4 octave. Water element = flowing, not grounded stone.
            "chord": [
                (D3,   0.0),
                (A3, -18.0),
                (F3, -22.0),
                (D4, -26.0),
            ],
            "voices": 4,
            "detune_cents": 10.0,
            "lp_hz": 4000.0,
            "sat_drive": 1.6,
            "breath_cycle_sec": 300.0 / 11,  # ~27.3s, 11 cycles
            "breath_depth": 0.04,
            "reverb_room": 0.72,
            "reverb_wet": 0.38,
        },
    },
    "binaural": {
        "builder": "binaural_beat",
        "params": {
            # 208Hz = 417/2 Solfeggio subharmonic, 6Hz theta beat
            # (emotional processing, creative flow)
            "carrier_hz": 208.0,
            "beat_hz": 6.0,
            "volume_db": -24.0,
        },
    },
    "bowl_a": {
        "builder": "sparse_sample_events",
        "params": {
            "source_path": "samples/cc0/singing_bowl_A3.wav",  # TODO: source
            "source_hz": A3,
            "target_hz": A3,
            "event_count": 3,
            "event_dur_range": (14.0, 20.0),
            "fade_in_sec": 8.0,
            "fade_out_sec": 14.0,
            "pitch_drift_cents": 20.0,
            "stereo_width": 0.18,
            "hp_hz": 80.0,
            "lp_hz": 6000.0,
            "reverb_room": 0.78,
            "reverb_wet": 0.42,
        },
    },
    "bowl_b": {
        "builder": "sparse_sample_events",
        "params": {
            # Même sample, timing/drift différents → asynchrone avec bowl_a
            "source_path": "samples/cc0/singing_bowl_A3.wav",  # TODO: source
            "source_hz": A3,
            "target_hz": A3,
            "event_count": 2,
            "event_dur_range": (18.0, 25.0),
            "fade_in_sec": 12.0,
            "fade_out_sec": 18.0,
            "pitch_drift_cents": 35.0,  # drift plus large que bowl_a
            "stereo_width": 0.22,
            "hp_hz": 80.0,
            "lp_hz": 6000.0,
            "reverb_room": 0.80,
            "reverb_wet": 0.45,
        },
    },
}

MIX = {
    "volumes_db": {
        "foundation": -6.0,
        "amber_mid":  -8.0,
        "binaural":  -24.0,
        "bowl_a":    -15.0,
        "bowl_b":    -17.0,
    },
}
