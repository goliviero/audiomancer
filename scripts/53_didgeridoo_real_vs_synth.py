"""Didgeridoo A/B — synth vs real sample + pitch-shift demo.

Uses the CC0 didgeridoo recording at samples/cc0/didgeridoo_C2.wav
(LaSonotheque.fr, license equivalent to CC0 — credit encouraged:
"Joseph SARDIN - LaSonotheque.org").

4 clips:
    01 Synthesized didgeridoo (the best my instruments.py can do, reference)
    02 Real sample at native pitch (C2 ~65.4 Hz)
    03 Real sample pitch-shifted up a 5th (G2 ~98 Hz)
    04 Real sample pitch-shifted to an octave (C3 ~131 Hz)

Earlier versions of the script also generated a 30s paulstretched pad
and a mix with synth handpan/derbouka, but user validation showed those
synths aren't production-ready yet — the plain pitched real sample is
the real win. Once we have CC0 handpan/derbouka samples in samples/cc0/,
the sampler workflow will give the same quality bump for them.

Usage:
    python scripts/53_didgeridoo_real_vs_synth.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.instruments import didgeridoo
from audiomancer.ir_reverb import reverb_from_synthetic
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.sampler import play_note
from audiomancer.utils import (
    export_wav,
    fade_in,
    fade_out,
    load_audio,
    mono_to_stereo,
    normalize,
)

SR = 48000
OUT = project_root / "output" / "didgeridoo_ab"
SAMPLE_PATH = project_root / "samples" / "cc0" / "didgeridoo_C2.wav"
SOURCE_HZ = 65.41  # C2


def _finalize(sig: np.ndarray, target_lufs: float = -14.0,
              fade_sec: float = 1.0) -> np.ndarray:
    if sig.ndim == 1:
        sig = mono_to_stereo(sig)
    sig = fade_in(sig, fade_sec, sample_rate=SR)
    sig = fade_out(sig, fade_sec, sample_rate=SR)
    sig = normalize_lufs(sig, target_lufs=target_lufs, sample_rate=SR)
    sig = master_chain(sig, sample_rate=SR)
    return sig


def main():
    if not SAMPLE_PATH.exists():
        print(f"[!] Sample missing: {SAMPLE_PATH}")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== Didgeridoo A/B — synth vs real sample ===")
    print(f"  Source: {SAMPLE_PATH.name}")
    print(f"  Output: {OUT}/")
    print()

    # Load the real sample (resample to 48k)
    real, _ = load_audio(SAMPLE_PATH, target_sr=SR)
    real = normalize(real, target_db=-3.0)  # normalize peak for predictable mix

    # --- Clip 01: Synthesized didgeridoo ---
    print("  [01_synth] Synthesized didgeridoo at 65.4 Hz + cathedral IR...")
    synth = didgeridoo(65.41, 10.0, breath_rate=0.35, formant_shift=0.0,
                       seed=42, sample_rate=SR)
    synth_s = mono_to_stereo(synth)
    synth_s = reverb_from_synthetic(synth_s, space="cathedral", wet=0.25,
                                    seed=42, sample_rate=SR)
    export_wav(_finalize(synth_s), OUT / "01_synth.wav", sample_rate=SR)
    print("    -> 01_synth.wav")

    # --- Clip 02: Real sample at native pitch ---
    print("  [02_real_native] Real sample at C2 (native) + cathedral IR...")
    real_s = mono_to_stereo(real)
    real_s = reverb_from_synthetic(real_s, space="cathedral", wet=0.25,
                                   seed=42, sample_rate=SR)
    export_wav(_finalize(real_s), OUT / "02_real_native.wav", sample_rate=SR)
    print("    -> 02_real_native.wav")

    # --- Clip 03: Real sample pitch-shifted to G2 (+5th = +7 semitones) ---
    print("  [03_real_G2] Real sample shifted to G2 (~98 Hz)...")
    g2 = play_note(real, source_hz=SOURCE_HZ, target_hz=98.0,
                   amplitude=0.9, sample_rate=SR)
    g2_s = mono_to_stereo(g2)
    g2_s = reverb_from_synthetic(g2_s, space="cathedral", wet=0.25,
                                 seed=42, sample_rate=SR)
    export_wav(_finalize(g2_s), OUT / "03_real_G2.wav", sample_rate=SR)
    print("    -> 03_real_G2.wav")

    # --- Clip 04: Real sample pitch-shifted up an octave (C3) ---
    print("  [04_real_C3] Real sample shifted to C3 (octave up)...")
    c3 = play_note(real, source_hz=SOURCE_HZ, target_hz=130.81,
                   amplitude=0.9, sample_rate=SR)
    c3_s = mono_to_stereo(c3)
    c3_s = reverb_from_synthetic(c3_s, space="cathedral", wet=0.25,
                                 seed=42, sample_rate=SR)
    export_wav(_finalize(c3_s), OUT / "04_real_C3.wav", sample_rate=SR)
    print("    -> 04_real_C3.wav  (note: +12 st is the edge before timbre shifts)")

    print()
    print(f"Done — 4 clips in {OUT}/")
    print()
    print("Listening tip:")
    print("  01 vs 02 : hear the gap between synth and real")
    print("  03 & 04  : free-pitch playback (proof that any note works)")


if __name__ == "__main__":
    main()
