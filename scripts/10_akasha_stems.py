"""Full pipeline — stems progressifs pour Akasha Portal.

Generates loopable 5-min stems with full mastering chain.
Each stem = 5 min, loopable seamless, progressive arc, mastered.

Features:
    - Progressive arc: Awakening -> Deepening -> Fullness -> Return
    - Spatial processing: auto-pan on textures, progressive stereo width
    - Breathing modulation on pad layer
    - Tonal coherence: all stems in A minor / compatible just intonation
    - Mastering chain: highpass 30Hz, mono bass, soft clip, limiter -1dBTP
    - Loop quality verification with cross-correlation check

Usage:
    python scripts/10_akasha_stems.py              # all stems
    python scripts/10_akasha_stems.py om           # only Om 136 Hz
    python scripts/10_akasha_stems.py holy sleep   # Holy + Sleep

Output: output/akasha_stems/
    progressive_om_136hz.wav    -> Akasha V003 / Om meditation
    progressive_holy_111hz.wav  -> Akasha Holy Frequency
    progressive_sleep_delta.wav -> Akasha Sleep / Delta
    progressive_focus_alpha.wav -> Akasha Focus / Alpha
    progressive_528hz.wav       -> Akasha Solfege 528 Hz
    progressive_432hz.wav       -> Akasha Sacred A=432 Hz

ffmpeg loop:
    ffmpeg -stream_loop -1 -i progressive_om_136hz.wav -t 10800 output_3h.wav
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

import audiomancer.quick as q
from audiomancer.compose import fade_envelope, tremolo, make_loopable, verify_loop
from audiomancer.modulation import apply_amplitude_mod, apply_filter_sweep
from audiomancer.layers import mix, normalize_lufs
from audiomancer.spatial import auto_pan, stereo_width
from audiomancer.envelope import breathing
from audiomancer.mastering import master_chain
from audiomancer.utils import export_wav

SR = 44100
OUT = project_root / "output" / "akasha_stems"


# ---------------------------------------------------------------------------
# Core builder — all 6 features integrated
# ---------------------------------------------------------------------------

def build_progressive_stem(
    freq: float,
    pad_freqs: list[float],
    binaural_preset: str,
    harmonics=None,
    seed: int = 42,
    duration: int = 300,
    standalone: bool = False,
) -> np.ndarray:
    """Build a progressive mastered stem.

    Arc: Awakening -> Deepening -> Fullness -> Return (loop-safe).

    Processing chain:
        1. Raw synthesis (drone, pad, texture, binaural)
        2. Volume envelopes (loop-safe start=end)
        3. Breathing modulation on pad
        4. Tremolo + filter sweep on drone
        5. Spatial: auto-pan on texture, progressive stereo width
        6. Mix + LUFS normalization
        7. Mastering chain (highpass, mono bass, soft clip, limiter)
        8. Loop seal (crossfade) or fade in/out (standalone)
    """
    if harmonics is None:
        harmonics = q.HARMONICS_WARM

    D = duration  # local alias

    # --- 1. Raw layers ---
    drone_raw = q.drone(freq, D, harmonics=harmonics,
                        cutoff_hz=3000, seed=seed)
    pad_raw   = q.pad(pad_freqs, D, voices=4, detune_cents=10.0, dark=True)
    tex_raw   = q.texture("noise_wash", D, seed=seed + 1)
    bbeat     = q.binaural(binaural_preset, D, volume_db=-14.0)

    # --- 2. Volume envelopes (proportional to duration) ---
    # Ratios: 0/300, 50/300, 240/300, etc.
    if standalone:
        # Standalone: fade in 10s, fade out 10s
        drone_vol = fade_envelope([
            (0, 0.0), (10, 1.0), (D - 10, 1.0), (D, 0.0),
        ], D)
        pad_vol = fade_envelope([
            (0, 0.0), (15, 0.85), (D - 15, 0.85), (D, 0.0),
        ], D)
        tex_vol = fade_envelope([
            (0, 0.0), (20, 0.65), (D - 20, 0.65), (D, 0.0),
        ], D)
    else:
        # Loop-safe: start = end
        r = D / 300  # scale factor
        drone_vol = fade_envelope([
            (0, 0.2), (50*r, 1.0), (240*r, 1.0), (280*r, 0.2), (D, 0.2),
        ], D)
        pad_vol = fade_envelope([
            (0, 0.0), (80*r, 0.0), (100*r, 0.85), (240*r, 0.85), (270*r, 0.0), (D, 0.0),
        ], D)
        tex_vol = fade_envelope([
            (0, 0.0), (165*r, 0.0), (190*r, 0.65), (235*r, 0.65), (260*r, 0.0), (D, 0.0),
        ], D)

    drone_raw = apply_amplitude_mod(drone_raw, drone_vol)
    pad_raw   = apply_amplitude_mod(pad_raw,   pad_vol)
    tex_raw   = apply_amplitude_mod(tex_raw,   tex_vol)

    # --- 3. Breathing modulation on pad (organic inhale/exhale) ---
    breath_env = breathing(D, breath_rate=0.08, depth=0.2, floor=0.8)
    pad_raw = apply_amplitude_mod(pad_raw, breath_env)

    # --- 4. Tremolo + filter sweep on drone ---
    drone_raw = tremolo(drone_raw, rate_hz=0.15, depth=0.05, seed=seed + 2)

    filter_curve = fade_envelope([
        (0, 800), (90*D/300, 2000), (150*D/300, 3000),
        (210*D/300, 2500), (270*D/300, 1200), (D, 800),
    ], D)
    drone_raw = apply_filter_sweep(drone_raw, filter_curve)

    # --- 5. Spatial processing ---
    tex_raw = auto_pan(tex_raw, rate_hz=0.04, depth=0.6, center=0.0)

    width_curve = fade_envelope([
        (0, 0.6), (50*D/300, 0.8), (150*D/300, 1.4),
        (210*D/300, 1.4), (270*D/300, 0.8), (D, 0.6),
    ], D)
    for i_layer in [drone_raw, pad_raw]:
        if i_layer.ndim == 2:
            mid = (i_layer[:, 0] + i_layer[:, 1]) * 0.5
            side = (i_layer[:, 0] - i_layer[:, 1]) * 0.5
            i_layer[:, 0] = mid + side * width_curve
            i_layer[:, 1] = mid - side * width_curve

    # --- 6. Mix + LUFS normalization ---
    stem = mix([drone_raw, pad_raw, tex_raw, bbeat],
               volumes_db=[0.0, -4.0, -8.0, 0.0])
    stem = normalize_lufs(stem, target_lufs=-14.0)

    # --- 7. Mastering chain ---
    stem = master_chain(stem)

    # --- 8. Loop seal or standalone fade ---
    if standalone:
        # Already faded in/out via envelopes
        pass
    else:
        stem = make_loopable(stem, crossfade_sec=5.0)

    return stem


# ---------------------------------------------------------------------------
# Stems catalogue — tonal coherence: A minor center, just intonation
# ---------------------------------------------------------------------------

# All pad chords use just intonation ratios from the root:
#   Minor triad: root, minor 3rd (6/5), perfect 5th (3/2)
#   Major triad: root, major 3rd (5/4), perfect 5th (3/2)
# A minor and C major are relative keys — all stems are harmonically compatible.

STEMS = {
    # Om 136.1 Hz (Earth year, Cousto) — A minor just intonation
    "om": dict(
        freq=136.1,
        pad_freqs=[136.1, 136.1 * 6/5, 136.1 * 3/2],  # A minor: 136.1, 163.3, 204.2
        binaural_preset="om_theta",
        filename="progressive_om_136hz",
        label="Om 136 Hz — Deep Theta Meditation",
    ),

    # Holy Frequency 111 Hz — A minor just intonation
    "holy": dict(
        freq=111.0,
        pad_freqs=[111.0, 111.0 * 6/5, 111.0 * 3/2],  # A minor: 111.0, 133.2, 166.5
        binaural_preset="om_theta",
        harmonics=q.HARMONICS_WARM,
        filename="progressive_holy_111hz",
        label="Holy Frequency 111 Hz — Sacred Theta",
    ),

    # Sleep / Delta — A minor, low register
    "sleep": dict(
        freq=110.0,
        pad_freqs=[110.0, 110.0 * 6/5, 110.0 * 3/2],  # A minor: 110.0, 132.0, 165.0
        binaural_preset="delta_sleep",
        harmonics=q.HARMONICS_DARK,
        filename="progressive_sleep_delta",
        label="Sleep — Delta 2 Hz",
    ),

    # Focus / Alpha — C major (relative major of A minor), just intonation
    "focus": dict(
        freq=261.63,
        pad_freqs=[261.63, 261.63 * 5/4, 261.63 * 3/2],  # C major: 261.6, 327.0, 392.4
        binaural_preset="alpha_relax",
        harmonics=q.HARMONICS_BRIGHT,
        filename="progressive_focus_alpha",
        label="Focus — Alpha 10 Hz",
    ),

    # Solfege 528 Hz — C major just intonation (528 = C5 in solfege)
    "528": dict(
        freq=528.0,
        pad_freqs=[528.0, 528.0 * 5/4, 528.0 * 3/2],  # C major: 528.0, 660.0, 792.0
        binaural_preset="solfeggio_528",
        harmonics=q.HARMONICS_WARM,
        filename="progressive_528hz",
        label="Solfege 528 Hz — Love Frequency",
    ),

    # Sacred A=432 Hz — A minor just intonation
    "432": dict(
        freq=432.0,
        pad_freqs=[432.0, 432.0 * 6/5, 432.0 * 3/2],  # A minor: 432.0, 518.4, 648.0
        binaural_preset="solfeggio_432",
        harmonics=q.HARMONICS_WARM,
        filename="progressive_432hz",
        label="Sacred A=432 Hz",
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Akasha Portal — progressive ambient stems generator."
    )
    parser.add_argument("stems", nargs="*", default=list(STEMS.keys()),
                        help="Stem keys to render (default: all)")
    parser.add_argument("--vary", action="store_true",
                        help="Use random seed instead of deterministic hash")
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview render")
    parser.add_argument("--standalone", action="store_true",
                        help="Fade in/out instead of loop-seal (non-looping export)")
    return parser.parse_args()


def main():
    args = parse_args()
    duration = 30 if args.preview else 300

    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 30s" if args.preview else "5 min"
    loop_mode = "standalone (fade)" if args.standalone else "loopable"
    seed_mode = "random" if args.vary else "deterministic"
    print("=== Akasha Stems — Progressive + Mastered ===")
    print(f"Output: {OUT}")
    print(f"Mode: {mode} | {loop_mode} | seed: {seed_mode}")
    print()

    for key in args.stems:
        if key not in STEMS:
            print(f"  Unknown stem: {key!r}. Available: {', '.join(STEMS)}")
            continue

        cfg = STEMS[key]
        seed = np.random.default_rng().integers(0, 100000) if args.vary else hash(key) % 1000
        print(f"[{key}] {cfg['label']}  (seed={seed})...")
        print(f"  Pad chord: {[round(f, 1) for f in cfg['pad_freqs']]} Hz")

        stem = build_progressive_stem(
            freq=cfg["freq"],
            pad_freqs=cfg["pad_freqs"],
            binaural_preset=cfg["binaural_preset"],
            harmonics=cfg.get("harmonics"),
            seed=seed,
            duration=duration,
            standalone=args.standalone,
        )

        # Loop quality check (skip for standalone)
        if not args.standalone:
            score, report = verify_loop(stem, crossfade_sec=5.0)
            quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
            print(f"  Loop: {quality} ({score:.3f}) | jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

        suffix = "_preview" if args.preview else ""
        suffix += "_standalone" if args.standalone else ""
        path = OUT / f"{cfg['filename']}{suffix}.wav"
        export_wav(stem, path)
        peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
        print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")
        print()

    files = list(OUT.glob("*.wav"))
    print(f"Done — {len(files)} stems in {OUT}/")
    if not args.standalone:
        print()
        print("ffmpeg loop command:")
        print("  ffmpeg -stream_loop -1 -i <stem>.wav -t 10800 output_3h.wav")


if __name__ == "__main__":
    main()
