"""V005 Stem 2 — Grounding bass pendulum C2 <-> G2.

Slow evolving bass: C2(66Hz) <-> G2(99Hz).
60s per note, 15s crossfade glide. Simple two-note pendulum.
Pure sine + subtle triangle at -18 dB.
LP 300 Hz (sub-focused, grounding).
No modulation, no tremolo — just breathe slowly.
Mastered -14 LUFS, 48 kHz WAV.

Usage:
    python scripts/22_v005_grounding_bass.py
    python scripts/22_v005_grounding_bass.py --preview   # 15s preview
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable, verify_loop
from audiomancer.effects import lowpass, reverb
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import apply_amplitude_mod, random_walk
from audiomancer.synth import sine, triangle
from audiomancer.utils import export_wav, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "V005"

# Two-note pendulum: C2 <-> G2 (Just Intonation)
PENDULUM = [66.0, 99.0]
NOTE_DUR = 60.0
XFADE = 15.0

# Triangle harmonic at -18 dB (~0.126 linear)
TRIANGLE_GAIN = 10 ** (-18 / 20)


def _build_note(freq: float, dur: float) -> np.ndarray:
    """Pure sine + subtle triangle at -18 dB."""
    main = sine(freq, dur, amplitude=0.7, sample_rate=SR)
    tri = triangle(freq, dur, amplitude=0.7 * TRIANGLE_GAIN, sample_rate=SR)
    return main + tri


def build_stem(duration: int = 300, seed: int = 42) -> np.ndarray:
    """Build the grounding bass pendulum.

    Args:
        duration: Duration in seconds.
        seed: Random seed for direction flip + per-note micro-detune.
    """
    rng = np.random.default_rng(seed)

    # Random pendulum direction: G2 first instead of C2?
    pendulum = list(PENDULUM)
    if rng.random() > 0.5:
        pendulum = list(reversed(pendulum))

    n_samples = int(duration * SR)
    xfade_samples = int(XFADE * SR)

    output = np.zeros(n_samples)
    pos = 0
    note_idx = 0

    while pos < n_samples:
        # Micro-detune +-1 cent per note
        detune_cents = rng.uniform(-1.0, 1.0)
        freq = pendulum[note_idx % len(pendulum)] * 2 ** (detune_cents / 1200)
        note_idx += 1

        # Phase B3: per-note duration jitter +-2s (break pendulum rigidity)
        note_dur = NOTE_DUR + rng.uniform(-2.0, 2.0)
        note_samples_var = int(note_dur * SR)
        note_len_sec = note_dur + XFADE
        note = _build_note(freq, note_len_sec)

        # Smooth crossfade envelope sized to this note's hold
        hold_samples = max(0, note_samples_var - xfade_samples)
        fade_in_env = np.linspace(0, 1, xfade_samples)
        fade_out_env = np.linspace(1, 0, xfade_samples)
        env = np.concatenate([fade_in_env, np.ones(hold_samples), fade_out_env])
        env = env[:len(note)]
        if len(env) < len(note):
            note = note[:len(env)]
        note = note * env

        end = min(pos + len(note), n_samples)
        chunk = end - pos
        output[pos:end] += note[:chunk]
        pos += note_samples_var

    # LP 300 Hz — sub-focused, grounding
    output = lowpass(output, cutoff_hz=300, sample_rate=SR)

    # Mono to stereo (centered, no Haas — no modulation)
    stereo = mono_to_stereo(output)

    # Very subtle reverb for presence (not modulating, just space)
    stereo = reverb(stereo, room_size=0.4, damping=0.5, wet_level=0.2,
                    sample_rate=SR)

    # Phase D6: random_walk amplitude drift (non-periodic, ~5% variation)
    # tau=30s -> slow enough to be felt, not heard
    walk = random_walk(duration, sigma=0.05, tau=30.0,
                       seed=seed, sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, walk)

    # LUFS + mastering + loop seal
    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V005 Stem 2 - Grounding bass pendulum C2<->G2."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 15s preview")
    parser.add_argument("--vary", action="store_true",
                        help="Random seed (different direction + detune each render)")
    args = parser.parse_args()

    duration = 15 if args.preview else 300
    seed = int(np.random.default_rng().integers(0, 100000)) if args.vary else 42
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 15s" if args.preview else "5 min"
    print(f"=== V005 Grounding Bass C2<->G2 - {mode} ===")
    print(f"  Pendulum: C2 (66 Hz) <-> G2 (99 Hz)")
    print(f"  Notes: {NOTE_DUR}s each, {XFADE}s crossfade glide")
    print(f"  Waveform: sine + triangle (-18 dB)")
    print(f"  Filter: lowpass 300 Hz (sub-focused)")
    print(f"  No modulation, no tremolo")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print(f"  Seed: {seed}{' (random)' if args.vary else ' (deterministic)'}")
    if args.preview:
        print(f"  NOTE: 15s preview shows only first note. Full render needed to hear pendulum.")
    print()

    stem = build_stem(duration, seed=seed)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V005_grounding_bass{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
