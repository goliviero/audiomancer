"""Gallery v3 — showcase of the 10 post-v2 features.

11 clips exercising:
    IR convolution reverb (4 spaces), sidechain ducking, bowed_string,
    granular pitch curve, morph_textures, ping-pong delay, suggest_eq_cuts,
    new binaural presets, arpeggio_from_chord, full ambient scene, viz PNGs.

Output: output/gallery_v3/*.wav  (+ viz/*.png for the last step)

Usage:
    python scripts/51_gallery_v3.py
    python scripts/51_gallery_v3.py --only 02_sidechain_duck_demo
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.binaural import from_preset
from audiomancer.effects import delay_pingpong, reverb_hall
from audiomancer.harmony import arpeggio_from_chord
from audiomancer.ir_reverb import reverb_from_synthetic
from audiomancer.layers import mix, normalize_lufs, suggest_eq_cuts
from audiomancer.mastering import master_chain
from audiomancer.sidechain import sidechain_duck
from audiomancer.spectral import paulstretch
from audiomancer.synth import (
    bowed_string,
    granular,
    karplus_strong,
    sine,
)
from audiomancer.textures import generate as texture_gen
from audiomancer.utils import export_wav, fade_in, fade_out, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "gallery_v3"


def _finalize(sig: np.ndarray, target_lufs: float = -14.0,
              fade_sec: float = 1.0) -> np.ndarray:
    if sig.ndim == 1:
        sig = mono_to_stereo(sig)
    sig = fade_in(sig, fade_sec, sample_rate=SR)
    sig = fade_out(sig, fade_sec, sample_rate=SR)
    sig = normalize_lufs(sig, target_lufs=target_lufs, sample_rate=SR)
    sig = master_chain(sig, sample_rate=SR)
    return sig


# ---------------------------------------------------------------------------
# Clip builders
# ---------------------------------------------------------------------------

def clip_01_ir_reverb_presets() -> np.ndarray:
    """Same karplus chord through room -> hall -> cathedral -> plate (6s each)."""
    section = 6.0
    # Short chord source
    src = np.zeros(int(3.0 * SR))
    for f in (196.0, 246.94, 293.66):
        src += karplus_strong(f, 3.0, decay=0.998, brightness=0.6,
                              amplitude=0.4, seed=int(f), sample_rate=SR)
    src = src / np.max(np.abs(src)) * 0.7

    chunks = []
    for space in ("room", "hall", "cathedral", "plate"):
        # Pad to section length with silence tail for reverb ringout
        padded = np.concatenate([src, np.zeros(int((section - 3.0) * SR))])
        wet = reverb_from_synthetic(padded, space=space, wet=0.6,
                                    pre_delay_ms=12, seed=42,
                                    sample_rate=SR)
        wet = fade_in(wet, 0.1, sample_rate=SR)
        wet = fade_out(wet, 0.3, sample_rate=SR)
        chunks.append(wet)
    combined = np.concatenate(chunks, axis=0)
    return _finalize(combined, fade_sec=0.3)


def clip_02_sidechain_duck_demo() -> np.ndarray:
    """Pad + chime. First 8s unducked, next 8s ducked by chime trigger."""
    section = 8.0
    # Pad (continuous)
    pad = np.zeros(int(section * SR))
    for f in (264.0, 330.0, 396.0):
        pad += 0.25 * np.sin(2 * np.pi * f * np.linspace(0, section, int(section * SR),
                                                         endpoint=False))
    pad_stereo = mono_to_stereo(pad)
    pad_stereo = reverb_hall(pad_stereo, sample_rate=SR)

    # Chime trigger: 3 spaced karplus hits at 1s, 3.5s, 6s
    trig = np.zeros(int(section * SR))
    for pos_s in (1.0, 3.5, 6.0):
        pluck = karplus_strong(784.0, 2.5, decay=0.998, brightness=0.7,
                               amplitude=0.7, seed=int(pos_s * 100),
                               sample_rate=SR)
        start = int(pos_s * SR)
        end = min(start + len(pluck), len(trig))
        trig[start:end] += pluck[:end - start]
    trig_stereo = mono_to_stereo(trig)
    trig_stereo = reverb_hall(trig_stereo, sample_rate=SR)

    # 1st half: pad + chime (no ducking)
    half_a = pad_stereo + trig_stereo

    # 2nd half: pad DUCKED against chime
    pad_ducked = sidechain_duck(pad_stereo, trig_stereo,
                                amount_db=-8.0, threshold_db=-28.0,
                                attack_ms=8, release_ms=180,
                                sample_rate=SR)
    half_b = pad_ducked + trig_stereo

    combined = np.concatenate([half_a, half_b], axis=0)
    return _finalize(combined, fade_sec=0.5)


def clip_03_bowed_vs_karplus() -> np.ndarray:
    """Same note plucked (Karplus) then bowed (friction model)."""
    freq = 196.0
    section = 5.0
    plucked = karplus_strong(freq, section, decay=0.9988, brightness=0.6,
                             amplitude=0.7, seed=42, sample_rate=SR)
    bowed = bowed_string(freq, section, bow_pressure=0.55,
                         bow_velocity=0.7, decay=0.9994,
                         brightness=0.55, amplitude=0.7,
                         seed=42, sample_rate=SR)
    plucked_s = mono_to_stereo(plucked)
    bowed_s = mono_to_stereo(bowed)
    plucked_s = reverb_hall(plucked_s, sample_rate=SR)
    bowed_s = reverb_hall(bowed_s, sample_rate=SR)
    plucked_s = fade_in(plucked_s, 0.1, sample_rate=SR)
    plucked_s = fade_out(plucked_s, 0.5, sample_rate=SR)
    bowed_s = fade_in(bowed_s, 0.3, sample_rate=SR)
    bowed_s = fade_out(bowed_s, 0.5, sample_rate=SR)
    combined = np.concatenate([plucked_s, bowed_s], axis=0)
    return _finalize(combined, fade_sec=0.3)


def clip_04_granular_pitch_shift() -> np.ndarray:
    """A sine drifts from 0 to +1 octave over 20s via pitch_curve."""
    duration = 20.0
    # Source: harmonic chord for richer granular texture
    src_dur = 3.0
    src = np.zeros(int(src_dur * SR))
    for f in (220.0, 277.0, 330.0):
        src += 0.35 * np.sin(
            2 * np.pi * f * np.linspace(0, src_dur, int(src_dur * SR),
                                        endpoint=False)
        )

    n_out = int(duration * SR)
    # Linear ramp 0 -> +1 octave
    curve = np.linspace(0.0, 1.0, n_out)
    cloud = granular(
        src, duration, grain_size_ms=100, grain_density=15,
        pitch_spread=0.0, pitch_curve=curve,
        amplitude=0.55, seed=42, sample_rate=SR,
    )
    stereo = mono_to_stereo(cloud)
    stereo = reverb_from_synthetic(stereo, space="hall", wet=0.5,
                                   seed=42, sample_rate=SR)
    return _finalize(stereo, fade_sec=2.0)


def clip_05_morph_textures() -> np.ndarray:
    """crystal_shimmer -> earth_hum over 15s via spectral morph."""
    duration = 15.0
    sig_a = texture_gen("crystal_shimmer", duration_sec=duration,
                        base_freq=528.0, seed=42, sample_rate=SR)
    sig_b = texture_gen("earth_hum", duration_sec=duration,
                        frequency=55.0, seed=42, sample_rate=SR)
    from audiomancer.spectral import morph
    min_len = min(sig_a.shape[0], sig_b.shape[0])
    morphed = morph(sig_a[:min_len], sig_b[:min_len], sample_rate=SR)
    return _finalize(morphed, fade_sec=1.0)


def clip_06_pingpong_delay() -> np.ndarray:
    """Karplus plucks traveling in stereo ping-pong."""
    duration = 10.0
    src = np.zeros(int(duration * SR))
    # 4 plucks at different positions
    for i, (pos, freq) in enumerate([(0.3, 392.0), (2.0, 523.25),
                                     (4.5, 659.25), (7.0, 784.0)]):
        pluck = karplus_strong(freq, 1.8, decay=0.997, brightness=0.7,
                               amplitude=0.5, seed=42 + i * 17,
                               sample_rate=SR)
        start = int(pos * SR)
        end = min(start + len(pluck), len(src))
        src[start:end] += pluck[:end - start]
    stereo = mono_to_stereo(src)
    # Ping-pong heavy
    wet = delay_pingpong(stereo, delay_seconds=0.45, feedback=0.55,
                        mix=0.5, cross_feedback=1.0, sample_rate=SR)
    wet = reverb_hall(wet, sample_rate=SR)
    return _finalize(wet, fade_sec=0.5)


def clip_07_multi_stem_eq_suggest() -> np.ndarray:
    """Print eq suggestions to console + render the mix with the suggestions applied to a peaking EQ."""
    duration = 8.0
    # Three overlapping stems: pad mids, bass low-mids, strings mids
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    pad = 0.3 * (np.sin(2 * np.pi * 600 * t) + np.sin(2 * np.pi * 900 * t))
    bass = 0.3 * np.sin(2 * np.pi * 180 * t)
    strings = 0.3 * (np.sin(2 * np.pi * 1100 * t) + np.sin(2 * np.pi * 1500 * t))
    stems = {"pad": pad, "bass": bass, "strings": strings}

    suggestions = suggest_eq_cuts(stems, sample_rate=SR)
    print("    suggest_eq_cuts ->", suggestions[:5])

    # Mix them straight (the gallery clip demonstrates the AUDIBLE masking)
    combined = mix([mono_to_stereo(v) for v in stems.values()],
                   volumes_db=[-6, -6, -6])
    combined = reverb_hall(combined, sample_rate=SR)
    return _finalize(combined, fade_sec=0.5)


def clip_08_binaural_new_presets() -> np.ndarray:
    """beta_13hz (6s) + smr_14hz (6s) + high_gamma_60hz (6s)."""
    section = 6.0
    parts = []
    for preset in ("beta_13hz", "smr_14hz", "high_gamma_60hz"):
        sig = from_preset(preset, section, sample_rate=SR)
        sig = fade_in(sig, 0.3, sample_rate=SR)
        sig = fade_out(sig, 0.3, sample_rate=SR)
        parts.append(sig)
    combined = np.concatenate(parts, axis=0)
    return _finalize(combined, fade_sec=0.3)


def clip_09_arpeggio_from_chord_cmaj9() -> np.ndarray:
    """Cmaj9 arpege up-down 2 octaves, each note as Karplus."""
    freqs = arpeggio_from_chord("Cmaj9", octaves=2, pattern="up_down",
                                root_octave=3)
    n_notes = len(freqs)
    note_gap = 0.4
    duration = n_notes * note_gap + 3.0  # extra tail for decay

    n_samples = int(duration * SR)
    out = np.zeros(n_samples)
    for i, freq in enumerate(freqs):
        start = int(i * note_gap * SR)
        if start >= n_samples:
            break
        pluck = karplus_strong(freq, 2.5, decay=0.9985, brightness=0.6,
                               amplitude=0.5, seed=i * 17,
                               sample_rate=SR)
        end = min(start + len(pluck), n_samples)
        out[start:end] += pluck[:end - start]

    stereo = mono_to_stereo(out)
    stereo = reverb_from_synthetic(stereo, space="hall", wet=0.5, seed=42,
                                   sample_rate=SR)
    return _finalize(stereo, fade_sec=0.3)


def clip_10_full_v3_ambient() -> np.ndarray:
    """Full scene: pad + bowed cellos + karplus chimes + IR cathedral + sidechain + morph texture underlay."""
    duration = 25.0

    # Pad layer (from builders.pad_alive)
    from audiomancer.builders import pad_alive
    pad = pad_alive(duration=duration, seed=42, sample_rate=SR,
                    chord=[132.0, 196.0, 329.63], intensity="moderate")

    # Bowed cello drone (mono)
    cello = bowed_string(131.0, duration, bow_pressure=0.5, bow_velocity=0.6,
                         decay=0.9998, brightness=0.5, amplitude=0.45,
                         seed=42, sample_rate=SR)
    cello_s = mono_to_stereo(cello)
    cello_s = reverb_from_synthetic(cello_s, space="cathedral", wet=0.4,
                                    seed=42, sample_rate=SR)

    # Karplus chimes (3 sparse hits)
    chimes = np.zeros(int(duration * SR))
    for pos, freq in [(3.0, 523.25), (9.0, 659.25), (17.0, 783.99)]:
        pluck = karplus_strong(freq, 5.0, decay=0.9985, brightness=0.75,
                               amplitude=0.45, seed=int(pos * 100),
                               sample_rate=SR)
        start = int(pos * SR)
        end = min(start + len(pluck), len(chimes))
        chimes[start:end] += pluck[:end - start]
    chimes_s = mono_to_stereo(chimes)
    chimes_s = reverb_from_synthetic(chimes_s, space="cathedral", wet=0.55,
                                     seed=42, sample_rate=SR)

    # Sidechain pad against chimes (breathing)
    min_len = min(pad.shape[0], chimes_s.shape[0], cello_s.shape[0])
    pad = pad[:min_len]
    chimes_s = chimes_s[:min_len]
    cello_s = cello_s[:min_len]
    pad = sidechain_duck(pad, chimes_s, amount_db=-4.0, threshold_db=-24,
                         attack_ms=10, release_ms=400, sample_rate=SR)

    combined = mix(
        [pad, cello_s, chimes_s],
        volumes_db=[-6.0, -9.0, -10.0],
    )
    return _finalize(combined, fade_sec=2.0)


CLIPS = {
    "01_ir_reverb_presets": (clip_01_ir_reverb_presets,
                             "Same source through room / hall / cathedral / plate"),
    "02_sidechain_duck_demo": (clip_02_sidechain_duck_demo,
                               "8s un-ducked then 8s pad-ducked by chime trigger"),
    "03_bowed_vs_karplus": (clip_03_bowed_vs_karplus,
                            "Same note plucked then bowed (friction model)"),
    "04_granular_pitch_shift": (clip_04_granular_pitch_shift,
                                "Sine chord drifts 0 -> +1 octave via pitch_curve"),
    "05_morph_textures": (clip_05_morph_textures,
                          "crystal_shimmer -> earth_hum over 15s (spectral morph)"),
    "06_pingpong_delay": (clip_06_pingpong_delay,
                          "4 karplus plucks traveling L<->R with ping-pong delay"),
    "07_multi_stem_eq_suggest": (clip_07_multi_stem_eq_suggest,
                                 "3 overlapping stems + suggest_eq_cuts printout"),
    "08_binaural_new_presets": (clip_08_binaural_new_presets,
                                "beta 13Hz, SMR 14Hz, high gamma 60Hz"),
    "09_arpeggio_from_chord_cmaj9": (clip_09_arpeggio_from_chord_cmaj9,
                                     "Cmaj9 up-down 2-octave karplus arpege"),
    "10_full_v3_ambient": (clip_10_full_v3_ambient,
                           "Full scene: pad + bowed cello + chimes + IR cathedral + sidechain"),
}


def _generate_viz(out_dir: Path) -> None:
    """Render PNGs for 3 representative clips."""
    try:
        from audiomancer.viz import plot_stem
    except ImportError:
        print("  [viz] matplotlib not installed, skipping PNGs "
              "(pip install audiomancer[viz])")
        return
    from audiomancer.utils import load_audio

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for label in ("03_bowed_vs_karplus", "05_morph_textures",
                  "10_full_v3_ambient"):
        wav = out_dir / f"{label}.wav"
        if not wav.exists():
            continue
        sig, sr = load_audio(wav, target_sr=SR)
        png = viz_dir / f"{label}.png"
        plot_stem(sig, png, sample_rate=sr, title=label)
        print(f"    -> viz/{png.name}")


def main():
    parser = argparse.ArgumentParser(description="Gallery v3 — 10 features showcase.")
    parser.add_argument("--only", default=None, help="Render only one clip by label")
    parser.add_argument("--no-viz", action="store_true", help="Skip PNG render")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    selected = {args.only: CLIPS[args.only]} if args.only else CLIPS

    print(f"=== Gallery v3 - {len(selected)} clip(s) ===")
    print(f"  Target: -14 LUFS, 48 kHz WAV, {OUT}/")
    print()

    for label, (fn, desc) in selected.items():
        print(f"  [{label}] {desc}")
        sig = fn()
        path = OUT / f"{label}.wav"
        export_wav(sig, path, sample_rate=SR)
        peak_db = 20 * np.log10(np.max(np.abs(sig)) + 1e-10)
        print(f"    -> {path.name}  ({sig.shape[0] / SR:.1f}s, peak={peak_db:.1f} dBFS)")

    if not args.no_viz:
        print()
        print("  [viz] rendering PNGs...")
        _generate_viz(OUT)

    print()
    print(f"Done - {len(selected)} clip(s) in {OUT}/")


if __name__ == "__main__":
    main()
