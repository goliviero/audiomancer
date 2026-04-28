"""V007 config — Brown Noise 10h Deep Sleep (Tier S sleep).

# FROZEN — V007 livré en production, ne plus modifier.

Layered stereo brown noise: 4 bandes (LP 500/1550/2500/3500) decorrele L/R,
densite 6 streams par canal par layer. 'Sleepy cacophony of deep rumbling'.
DRY (no reverb), no filter sweep, breath OFF par defaut (anti-fatigue
disponible via breath_cycle_sec/breath_depth_db si besoin).

Validated: scripts/_archive/v007/66_v007_brown_v7.py / 05_deep_dominant.

Rendered:
    python scripts/render_mix.py --config V007
    # Akasha s'occupe du x120 ffmpeg -> 10h en aval.
"""

META = {
    "label": "V007 - Brown Noise 10h Deep Sleep",
    "target_lufs": -16.0,    # Sweet spot brown noise sleep (Akasha standard)
    "sample_rate": 48000,
    "duration": 300,          # 5min loop, Akasha etend a 10h via ffmpeg
    "master_mode": "ambient", # Gated LUFS BS.1770, no maximizer
    "ceiling_dbtp": -1.0,     # Akasha rec: standard YouTube post-AAC
    "pre_fade_sec": 0.0,      # Akasha gere les fades (10s in / 30s out) au mux
    "bit_depth": 24,
    # Pure noise (aperiodic) needs explicit boundary continuity for
    # ffmpeg -stream_loop to be sample-clean. Inaudible DC tilt fix.
    "loop_boundary_continuity": True,
}

STEMS = {
    "brown_bed": {
        "builder": "layered_brown_stereo",
        "params": {
            # 4 layers stereo decorreles. (lp_hz, db_offset).
            # 0 dB sur le sub (LP 500) et le bed (LP 1550 'avion'),
            # body et air en accent doux pour eviter ear-fatigue 10h.
            "layers": [
                [500, 0.0],     # Sub rumble (deep)
                [1550, 0.0],    # Bed 'avion' (corps principal)
                [2500, -10.0],  # Body discret
                [3500, -14.0],  # Air discret
            ],
            "n_streams_per_channel": 6,
            "hp_hz": 40.0,
            # breath OFF par defaut (fidele au v7/05 valide).
            # Pour activer anti-fatigue Akasha-style, ajouter:
            #   "breath_cycle_sec": 25.0, "breath_depth_db": 1.0
        },
    },
}

MIX = {
    "volumes_db": {
        "brown_bed": 0.0,  # Le builder peak-normalise deja, LUFS final gere le niveau.
    },
}
