"""V005 Stem 2 bis — Arpege bass C sus2 palindrome (alternative to pendulum).

C sus2 Just Intonation, palindrome across 2 octaves:
    C2 -> D2 -> G2 -> C3 -> D3 -> G3 -> D3 -> C3 -> G2 -> D2 (cycle)
Each note 15s, 3s crossfade glide. Cycle ~150s (2 cycles per 5 min).
Sine + subtle triangle at -18 dB. HP 50 Hz, LP 500 Hz (mid clarity).
Light reverb, Haas 3ms. random_walk amplitude drift (Phase D).
Mastered -14 LUFS, 48 kHz WAV.

Usage:
    python scripts/28_v005_arpege_bass.py
    python scripts/28_v005_arpege_bass.py --preview
    python scripts/28_v005_arpege_bass.py --vary
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable, verify_loop
from audiomancer.effects import chorus_subtle, highpass, lowpass, reverb
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import apply_amplitude_mod, random_walk
from audiomancer.spatial import haas_width
from audiomancer.synth import sine, triangle
from audiomancer.utils import export_wav, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "V005"

# C sus2 JI palindrome: C-D-G across two octaves, up then down
# Root C2 = 66 Hz, D2 = 66 * 9/8, G2 = 66 * 3/2, etc.
ARPEGE = [
    66.0,       # C2
    74.25,      # D2 (9/8)
    99.0,       # G2 (3/2)
    132.0,      # C3 (2)
    148.5,      # D3 (2 * 9/8)
    198.0,      # G3 (2 * 3/2)
    148.5,      # D3 (descent)
    132.0,      # C3
    99.0,       # G2
    74.25,      # D2 (loops back to C2)
]
NOTE_DUR = 15.0
XFADE = 3.0
TRIANGLE_GAIN = 10 ** (-18 / 20)


def _build_note(freq: float, dur: float) -> np.ndarray:
    main = sine(freq, dur, amplitude=0.7, sample_rate=SR)
    tri = triangle(freq, dur, amplitude=0.7 * TRIANGLE_GAIN, sample_rate=SR)
    return main + tri


def build_stem(duration: int = 300, seed: int = 42) -> np.ndarray:
    """Build the arpege bass palindrome stem."""
    rng = np.random.default_rng(seed)

    n_samples = int(duration * SR)
    xfade_samples = int(XFADE * SR)

    output = np.zeros(n_samples)
    pos = 0
    note_idx = 0

    while pos < n_samples:
        # Micro-detune +-1 cent per note (seeded)
        detune_cents = rng.uniform(-1.0, 1.0)
        freq = ARPEGE[note_idx % len(ARPEGE)] * 2 ** (detune_cents / 1200)
        note_idx += 1

        # Note duration jitter +-1s (small since notes are already short)
        note_dur = NOTE_DUR + rng.uniform(-1.0, 1.0)
        note_samples_var = int(note_dur * SR)
        note_len_sec = note_dur + XFADE
        note = _build_note(freq, note_len_sec)

        # Smooth crossfade envelope
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

    # Filter chain: HP 50 Hz (no sub) + LP 500 Hz (mid clarity)
    output = highpass(output, cutoff_hz=50, sample_rate=SR)
    output = lowpass(output, cutoff_hz=500, sample_rate=SR)

    # Stereo + Haas 3ms (subtle width)
    stereo = mono_to_stereo(output)
    stereo = haas_width(stereo, delay_ms=3.0, sample_rate=SR)

    # Subtle chorus
    stereo = chorus_subtle(stereo, sample_rate=SR)

    # Light reverb
    stereo = reverb(stereo, room_size=0.5, damping=0.5, wet_level=0.35,
                    sample_rate=SR)

    # Phase D: random_walk amplitude drift (non-periodic, ~5% variation)
    walk = random_walk(duration, sigma=0.05, tau=30.0,
                       seed=seed, sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, walk)

    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V005 Arpege bass C sus2 palindrome (2 octaves)."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    parser.add_argument("--vary", action="store_true",
                        help="Random seed")
    args = parser.parse_args()

    duration = 30 if args.preview else 300
    seed = int(np.random.default_rng().integers(0, 100000)) if args.vary else 42
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 30s" if args.preview else "5 min"
    print(f"=== V005 Arpege Bass C sus2 - {mode} ===")
    print(f"  Palindrome: C2->D2->G2->C3->D3->G3->D3->C3->G2->D2")
    print(f"  Notes: {NOTE_DUR}s +-1s jitter, {XFADE}s crossfades")
    print(f"  Waveform: sine + triangle (-18 dB)")
    print(f"  Filter: HP 50 Hz + LP 500 Hz")
    print(f"  Modulation: random_walk amplitude (Phase D)")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print(f"  Seed: {seed}{' (random)' if args.vary else ' (deterministic)'}")
    print()

    stem = build_stem(duration, seed=seed)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V005_arpege_bass{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
