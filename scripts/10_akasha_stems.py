"""Full pipeline — stems progressifs pour Akasha Portal.

Génère plusieurs stems loopables prêts à injecter dans Akasha via ffmpeg.
Chaque stem = 5 min, loopable seamless, progression musicale complète.

Usage:
    python scripts/10_akasha_stems.py              # tous les stems
    python scripts/10_akasha_stems.py om           # seulement Om 136 Hz
    python scripts/10_akasha_stems.py holy sleep   # Holy + Sleep

Output: output/akasha_stems/
    progressive_om_136hz.wav    → Akasha V003 / Om méditation
    progressive_holy_111hz.wav  → Akasha Holy Frequency
    progressive_sleep_delta.wav → Akasha Sleep / Delta
    progressive_focus_alpha.wav → Akasha Focus / Alpha
    progressive_528hz.wav       → Akasha Solfège 528 Hz

Boucle ffmpeg :
    ffmpeg -stream_loop -1 -i progressive_om_136hz.wav -t 10800 output_3h.wav
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

import audiomancer.quick as q
from audiomancer.compose import fade_envelope, tremolo, make_loopable
from audiomancer.modulation import apply_amplitude_mod, apply_filter_sweep
from audiomancer.layers import mix, normalize_lufs
from audiomancer.utils import export_wav

DURATION = 300
SR = 44100
OUT = project_root / "output" / "akasha_stems"


# ---------------------------------------------------------------------------
# Core builder — réutilisé par tous les stems
# ---------------------------------------------------------------------------

def build_progressive_stem(
    freq: float,
    pad_freqs: list[float],
    binaural_preset: str,
    harmonics=None,
    seed: int = 42,
) -> np.ndarray:
    """Build a 5-min progressive loopable stem.

    Arc: Awakening → Deepening → Fullness → Return (loop-safe).
    """
    if harmonics is None:
        harmonics = q.HARMONICS_WARM

    # --- 1. Raw layers ---
    drone_raw = q.drone(freq, DURATION, harmonics=harmonics,
                        cutoff_hz=3000, seed=seed)
    pad_raw   = q.pad(pad_freqs, DURATION, voices=4, detune_cents=10.0, dark=True)
    tex_raw   = q.texture("noise_wash", DURATION, seed=seed + 1)
    bbeat     = q.binaural(binaural_preset, DURATION, volume_db=-14.0)

    # --- 2. Volume envelopes (start = end for seamless loop) ---
    drone_vol = fade_envelope([
        (0,   0.2), (50,  1.0), (240, 1.0), (280, 0.2), (300, 0.2),
    ], DURATION)
    pad_vol = fade_envelope([
        (0,   0.0), (80,  0.0), (100, 0.85), (240, 0.85), (270, 0.0), (300, 0.0),
    ], DURATION)
    tex_vol = fade_envelope([
        (0,   0.0), (165, 0.0), (190, 0.65), (235, 0.65), (260, 0.0), (300, 0.0),
    ], DURATION)

    drone_raw = apply_amplitude_mod(drone_raw, drone_vol)
    pad_raw   = apply_amplitude_mod(pad_raw,   pad_vol)
    tex_raw   = apply_amplitude_mod(tex_raw,   tex_vol)

    # --- 3. Tremolo sur le drone ---
    drone_raw = tremolo(drone_raw, rate_hz=0.15, depth=0.05, seed=seed + 2)

    # --- 4. Filter sweep sur le drone (s'ouvre, se referme) ---
    filter_curve = fade_envelope([
        (0,   800), (90, 2000), (150, 3000), (210, 2500), (270, 1200), (300, 800),
    ], DURATION)
    drone_raw = apply_filter_sweep(drone_raw, filter_curve)

    # --- 5. Mix + loop seal ---
    stem = mix([drone_raw, pad_raw, tex_raw, bbeat],
               volumes_db=[0.0, -4.0, -8.0, 0.0])
    stem = normalize_lufs(stem, target_lufs=-14.0)
    stem = make_loopable(stem, crossfade_sec=5.0)
    return stem


# ---------------------------------------------------------------------------
# Stems catalogue
# ---------------------------------------------------------------------------

STEMS = {
    # Om 136 Hz — méditation profonde, thème V003
    "om": dict(
        freq=136.1,
        pad_freqs=[136.1, 165.0, 220.0],        # A minor
        binaural_preset="om_theta",
        filename="progressive_om_136hz",
        label="Om 136 Hz — Deep Theta Meditation",
    ),

    # Holy Frequency 111 Hz
    "holy": dict(
        freq=111.0,
        pad_freqs=[111.0, 132.0, 166.0],        # A minor-ish
        binaural_preset="om_theta",
        harmonics=q.HARMONICS_WARM,
        filename="progressive_holy_111hz",
        label="Holy Frequency 111 Hz — Sacred Theta",
    ),

    # Sleep / Delta — grave, sombre
    "sleep": dict(
        freq=111.0,
        pad_freqs=[110.0, 130.81, 164.81],      # A2, C3, E3
        binaural_preset="delta_sleep",
        harmonics=q.HARMONICS_DARK,
        filename="progressive_sleep_delta",
        label="Sleep — Delta 2 Hz",
    ),

    # Focus / Alpha — moins grave, plus lumineux
    "focus": dict(
        freq=256.0,
        pad_freqs=[261.63, 329.63, 392.0],      # C major
        binaural_preset="alpha_relax",
        harmonics=q.HARMONICS_BRIGHT,
        filename="progressive_focus_alpha",
        label="Focus — Alpha 10 Hz",
    ),

    # Solfège 528 Hz — love frequency
    "528": dict(
        freq=528.0,
        pad_freqs=[528.0, 660.0, 792.0],
        binaural_preset="solfeggio_528",
        harmonics=q.HARMONICS_WARM,
        filename="progressive_528hz",
        label="Solfège 528 Hz — Love Frequency",
    ),

    # Sacred dual — 432 Hz (grounding, low filter)
    "432": dict(
        freq=432.0,
        pad_freqs=[432.0, 528.0, 648.0],
        binaural_preset="solfeggio_432",
        harmonics=q.HARMONICS_WARM,
        filename="progressive_432hz",
        label="Sacred A=432 Hz",
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(STEMS.keys())
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Akasha Stems — Progressive ===")
    print(f"Output: {OUT}")
    print()

    for key in selected:
        if key not in STEMS:
            print(f"  Unknown stem: {key!r}. Available: {', '.join(STEMS)}")
            continue

        cfg = STEMS[key]
        print(f"[{key}] {cfg['label']}...")

        stem = build_progressive_stem(
            freq=cfg["freq"],
            pad_freqs=cfg["pad_freqs"],
            binaural_preset=cfg["binaural_preset"],
            harmonics=cfg.get("harmonics"),
            seed=hash(key) % 1000,
        )

        path = OUT / f"{cfg['filename']}.wav"
        export_wav(stem, path)
        print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={np.max(np.abs(stem)):.3f})")
        print()

    files = list(OUT.glob("*.wav"))
    print(f"Done — {len(files)} stems in {OUT}/")
    print()
    print("ffmpeg loop command:")
    print("  ffmpeg -stream_loop -1 -i <stem>.wav -t 10800 output_3h.wav")


if __name__ == "__main__":
    main()
