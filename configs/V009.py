"""V009 config — Rain on Cabin Window 3h (thematic, no chakra).

Rain field recording + subtle warm drone + subliminal delta binaural.
Pure thematic episode, différenciation de la série chakras par absence de
yantra audio et présence de texture naturaliste (pluie).

TODO: source `samples/cc0/heavy_rain_loop_180s.wav` via Freesound CC0 —
recherche: "heavy rain window cabin no thunder loop", CC0, 3min+, stereo.

Rendered:
    python scripts/render_mix.py --config V009
"""

C2 = 65.41
G2 = 98.00

META = {
    "label": "V009 - Rain Cabin 3h",
    "target_lufs": -18.0,    # un peu plus haut: rain bed a besoin de présence
    "sample_rate": 48000,
    "duration": 300,
    "master_mode": "ambient",
    "ceiling_dbtp": -3.0,
    "pre_fade_sec": 1.5,
    "bit_depth": 24,
}

STEMS = {
    "rain_core": {
        # Event continu: 1 event qui couvre toute la durée avec long fade in/out.
        # sparse_sample_events gère les fades + band-limit + reverb.
        "builder": "sparse_sample_events",
        "params": {
            "source_path": "samples/cc0/heavy_rain_loop_180s.wav",  # TODO: source
            "source_hz": 100.0,   # Pas de pitch naturel, target = source (no shift)
            "target_hz": 100.0,
            "event_count": 1,
            "event_dur_range": (290.0, 300.0),  # quasi toute la durée
            "fade_in_sec": 15.0,
            "fade_out_sec": 30.0,
            "pitch_drift_cents": 0.0,  # pas de drift sur la pluie
            "stereo_width": 0.10,      # légère extension stéréo
            "hp_hz": 40.0,             # rumble removal
            "lp_hz": 16000.0,          # preserve texture pluie
            "reverb_room": 0.35,       # pluie a déjà son espace, peu d'ajout
            "reverb_wet": 0.10,
        },
    },
    "warm_drone": {
        "builder": "foundation_drone",
        "params": {
            # C2 + G2 fondamentale + quinte, barely audible sous la pluie.
            # Donne chaleur émotionnelle sans competing avec le rain.
            "freqs": [C2, G2],
            "detune_cents": 3.0,
            "lp_hz": 400.0,
            "amp_mod_cycle_sec": 60.0,   # breath lent, 5 cycles sur 300s
            "amp_mod_depth_db": 1.0,
            "reverb_room": 0.50,
            "reverb_wet": 0.15,
        },
    },
    "binaural_subliminal": {
        "builder": "binaural_beat",
        "params": {
            # Delta 3Hz pour sleep onset. -28dB = subliminal, masqué par pluie
            # pour les listeners qui cherchent "pure rain", mais le cerveau
            # capte le signal delta.
            "carrier_hz": 110.0,
            "beat_hz": 3.0,
            "volume_db": -28.0,
        },
    },
}

MIX = {
    "volumes_db": {
        "rain_core":          -6.0,
        "warm_drone":        -18.0,
        "binaural_subliminal": -28.0,
    },
}
