"""V007 config — Brown Noise 10h (Tier S sleep).

Pure brown noise bed pour sommeil profond. Pas de binaural, pas de signature.
Le filter sweep + reverb baked dans `noise_wash` évite l'ear fatigue sur 10h.
Render 300s (5min), ffmpeg -stream_loop 119 pour 10h final.

Rendered:
    python scripts/render_mix.py --config V007
    ffmpeg -stream_loop 119 -i mix_V007.wav -c copy V007_10h.wav
"""

META = {
    "label": "V007 - Brown Noise 10h Deep Sleep",
    "target_lufs": -16.0,    # Un peu plus haut que -20 ambient: noise tolère densité
    "sample_rate": 48000,
    "duration": 300,          # 5min loop, ffmpeg étend à 10h
    "master_mode": "ambient", # gated LUFS, no maximizer (sommeil)
    "ceiling_dbtp": -3.0,
    "pre_fade_sec": 1.5,
    "bit_depth": 24,
}

STEMS = {
    "brown_bed": {
        "builder": "texture",
        "params": {
            # noise_wash génère brown noise + LP 3000Hz + HP 40Hz +
            # evolving filter sweep 0.04Hz (25s cycle) + reverb 0.7/0.4
            "texture_name": "noise_wash",
            "color": "brown",
        },
    },
}

MIX = {
    "volumes_db": {
        "brown_bed": -6.0,
    },
}
