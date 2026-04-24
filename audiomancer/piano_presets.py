"""Piano processing presets — library module.

Extracted from scripts/piano/process_piano.py so they can be reused by
builders.piano_processed + other scripts without importing the CLI wrapper.

3 presets transform a raw piano .wav into an ambient stem:
    bass_drone    LP 500 + cathedral reverb + slow compression + loop
    mid_pad       LP 3k + chorus + hall reverb + gentle compression + loop
    sparse_notes  long delay + cathedral + very gentle compression, no forced loop

Each preset returns a stereo ndarray. LUFS normalization is NOT applied
(caller decides target based on context).
"""

import numpy as np

from audiomancer.compose import make_loopable
from audiomancer.effects import (
    chorus_subtle,
    compress,
    delay,
    lowpass,
    reverb_cathedral,
    reverb_hall,
)
from audiomancer.utils import fade_in, fade_out, mono_to_stereo


def _ensure_stereo(sig: np.ndarray) -> np.ndarray:
    return mono_to_stereo(sig) if sig.ndim == 1 else sig


def _pad_or_trim_to_duration(sig: np.ndarray, duration: float,
                             sample_rate: int) -> np.ndarray:
    """Extend (loop-tile) or trim signal to match target duration."""
    target_n = int(duration * sample_rate)
    n = sig.shape[0]
    if n == target_n:
        return sig
    if n > target_n:
        return sig[:target_n]
    repeats = (target_n + n - 1) // n
    if sig.ndim == 2:
        tiled = np.tile(sig, (repeats, 1))
    else:
        tiled = np.tile(sig, repeats)
    return tiled[:target_n]


def preset_bass_drone(sig: np.ndarray, duration: float,
                      sample_rate: int) -> np.ndarray:
    """piano -> deep sub drone: LP 500 + cathedral + slow compression + loop."""
    sig = lowpass(sig, cutoff_hz=500, sample_rate=sample_rate)
    sig = _ensure_stereo(sig)
    sig = reverb_cathedral(sig, sample_rate=sample_rate)
    sig = compress(sig, threshold_db=-18.0, ratio=3.0,
                   attack_ms=50.0, release_ms=500.0, sample_rate=sample_rate)
    sig = _pad_or_trim_to_duration(sig, duration, sample_rate)
    sig = make_loopable(sig, crossfade_sec=3.0, sample_rate=sample_rate)
    sig = fade_in(sig, 2.0, sample_rate=sample_rate)
    sig = fade_out(sig, 2.0, sample_rate=sample_rate)
    return sig


def preset_mid_pad(sig: np.ndarray, duration: float,
                   sample_rate: int) -> np.ndarray:
    """piano -> warm pad: LP 3k + subtle chorus + hall reverb + soft compression + loop."""
    sig = lowpass(sig, cutoff_hz=3000, sample_rate=sample_rate)
    sig = _ensure_stereo(sig)
    sig = chorus_subtle(sig, sample_rate=sample_rate)
    sig = reverb_hall(sig, sample_rate=sample_rate)
    sig = compress(sig, threshold_db=-20.0, ratio=2.0,
                   attack_ms=20.0, release_ms=300.0, sample_rate=sample_rate)
    sig = _pad_or_trim_to_duration(sig, duration, sample_rate)
    sig = make_loopable(sig, crossfade_sec=2.0, sample_rate=sample_rate)
    sig = fade_in(sig, 1.5, sample_rate=sample_rate)
    sig = fade_out(sig, 1.5, sample_rate=sample_rate)
    return sig


def preset_sparse_notes(sig: np.ndarray, duration: float,
                        sample_rate: int) -> np.ndarray:
    """piano -> sparse notes with ultra-long decay.

    No LP (keep piano identity). Shimmer-style via long delay + cathedral.
    `duration` argument preserved for signature compatibility but not enforced
    (natural decay is the content).
    """
    del duration  # unused for sparse_notes
    sig = _ensure_stereo(sig)
    sig = delay(sig, delay_seconds=1.2, feedback=0.4, mix=0.35,
                sample_rate=sample_rate)
    sig = reverb_cathedral(sig, sample_rate=sample_rate)
    sig = compress(sig, threshold_db=-24.0, ratio=1.5,
                   attack_ms=50.0, release_ms=1000.0, sample_rate=sample_rate)
    sig = fade_in(sig, 3.0, sample_rate=sample_rate)
    sig = fade_out(sig, 5.0, sample_rate=sample_rate)
    return sig


# Registry: (preset_function, default_target_lufs)
PRESETS = {
    "bass_drone": (preset_bass_drone, -18.0),
    "mid_pad": (preset_mid_pad, -18.0),
    "sparse_notes": (preset_sparse_notes, -22.0),
}
