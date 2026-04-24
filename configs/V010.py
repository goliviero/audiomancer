"""V010 config — Pink Noise 8h (focus + sleep dual-use).

Pink noise pur (tilt -3dB/octave, balanced warmth vs clarity) pour sessions
focus + sommeil standard. Durée 8h = full night sans le buffer 2h de V007
brown noise. Teste le format "focus work session + full sleep".

Rendered:
    python scripts/render_mix.py --config V010
    ffmpeg -stream_loop 95 -i mix_V010.wav -c copy V010_8h.wav
"""

META = {
    "label": "V010 - Pink Noise 8h Focus/Sleep",
    "target_lufs": -16.0,
    "sample_rate": 48000,
    "duration": 300,
    "master_mode": "ambient",
    "ceiling_dbtp": -3.0,
    "pre_fade_sec": 1.5,
    "bit_depth": 24,
}

STEMS = {
    "pink_bed": {
        "builder": "texture",
        "params": {
            # noise_wash color="pink" → pink noise + LP 3kHz + HP 40Hz +
            # evolving filter sweep + reverb. Plus de presence HF que brown
            # (adapté focus) tout en restant non-fatiguant sur 8h.
            "texture_name": "noise_wash",
            "color": "pink",
        },
    },
}

MIX = {
    "volumes_db": {
        "pink_bed": -6.0,
    },
}
