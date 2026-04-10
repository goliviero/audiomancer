"""V004 Stem 1 — Bass arpeggio G minor, palindrome cycling.

Arpeggio palindrome: G2(98) -> Bb2(117.6) -> D3(147) -> G3(196) -> D3 -> Bb2
Each note 40s with 5s crossfade glides. Cycle = 240s.
5 min = 300s, loop-sealed with 5s crossfade at junction.
Warm sine + subtle saw undertone, sub-bass emphasis.
Breathing +-3dB (30s cycle). Warm reverb. Haas 5ms.
Mastered -14 LUFS, 48 kHz WAV.

Usage:
    python scripts/16_v004_bass_arpeggio.py
    python scripts/16_v004_bass_arpeggio.py --preview
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable, verify_loop
from audiomancer.effects import lowpass, reverb
from audiomancer.envelope import breathing
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import apply_amplitude_mod
from audiomancer.spatial import haas_width
from audiomancer.synth import drone, sawtooth
from audiomancer.utils import export_wav, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "V004"

# Palindrome: G2 -> Bb2 -> D3 -> G3 -> D3 -> Bb2 (no duplicates at extremes)
ARPEGGIO = [98.0, 117.6, 147.0, 196.0, 147.0, 117.6]
NOTE_DUR = 40.0
XFADE = 5.0
HARMONICS_BASS = [(1, 1.0), (2, 0.35), (3, 0.12), (4, 0.05)]


def build_stem(duration: int = 300) -> np.ndarray:
    """Build the bass arpeggio palindrome stem."""
    n_samples = int(duration * SR)
    xfade_samples = int(XFADE * SR)
    note_samples = int(NOTE_DUR * SR)

    output = np.zeros(n_samples)
    pos = 0
    note_idx = 0

    while pos < n_samples:
        freq = ARPEGGIO[note_idx % len(ARPEGGIO)]
        note_idx += 1

        note_len_sec = NOTE_DUR + XFADE
        main = drone(freq, note_len_sec, harmonics=HARMONICS_BASS,
                     amplitude=0.7, sample_rate=SR)
        saw = sawtooth(freq, note_len_sec, amplitude=0.15, sample_rate=SR)
        saw = lowpass(saw, cutoff_hz=300, sample_rate=SR)
        note = main + saw
        note = lowpass(note, cutoff_hz=250, sample_rate=SR)

        # Smooth crossfade envelope
        fade_in_env = np.linspace(0, 1, xfade_samples)
        fade_out_env = np.linspace(1, 0, xfade_samples)
        hold_samples = note_samples - xfade_samples
        env = np.concatenate([fade_in_env, np.ones(hold_samples), fade_out_env])
        env = env[:len(note)]
        if len(env) < len(note):
            note = note[:len(env)]
        note = note * env

        end = min(pos + len(note), n_samples)
        chunk = end - pos
        output[pos:end] += note[:chunk]
        pos += note_samples

    # Stereo + Haas 5ms
    stereo = mono_to_stereo(output)
    stereo = haas_width(stereo, delay_ms=5.0, sample_rate=SR)

    # Warm reverb
    stereo = reverb(stereo, room_size=0.8, damping=0.5, wet_level=0.45,
                    sample_rate=SR)

    # Breathing +-3dB, 30s cycle
    breath_env = breathing(duration, breath_rate=1.0 / 30.0, depth=0.29,
                           floor=0.71, sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, breath_env)

    # LUFS + mastering + loop seal
    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=5.0, sample_rate=SR)

    return stereo


def main():
    parser = argparse.ArgumentParser(
        description="V004 Stem 1 — Bass arpeggio G minor palindrome."
    )
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    args = parser.parse_args()

    duration = 30 if args.preview else 300
    OUT.mkdir(parents=True, exist_ok=True)

    mode = "preview 30s" if args.preview else "5 min"
    print(f"=== V004 Bass Arpeggio G minor — {mode} ===")
    print(f"  Palindrome: G2->Bb2->D3->G3->D3->Bb2")
    print(f"  Notes: {NOTE_DUR}s each, {XFADE}s crossfades")
    print(f"  Breathing: 30s cycle, +-3dB")
    print(f"  Reverb: warm (room=0.8), Haas 5ms")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print()

    stem = build_stem(duration)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    suffix = "_preview" if args.preview else ""
    path = OUT / f"V004_bass_arpeggio{suffix}.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
