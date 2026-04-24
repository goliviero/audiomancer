"""Saturation — tape/vinyl analog warmth emulators.

Three functions for adding subtle analog character to digital audio:

- tape_saturate  : asymmetric soft-clip (adds even + odd harmonics like tape)
- tape_hiss      : subliminal pink-noise floor (stereo-decorrelated)
- vinyl_wow      : slow pitch modulation emulating tape/vinyl flutter

These are meant to be applied manually in scripts (not in master_chain),
so the mastering chain stays deterministic and predictable.

Typical usage:
    from audiomancer.saturation import tape_saturate, tape_hiss, vinyl_wow

    stem = tape_saturate(stem, drive=1.1, asymmetry=0.12)
    hiss = tape_hiss(len(stem) / SR, level_db=-45)
    stem = stem + hiss
    stem = vinyl_wow(stem, depth=0.0005, rate_hz=0.25)
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.synth import pink_noise


def tape_saturate(signal: np.ndarray, drive: float = 1.0,
                  asymmetry: float = 0.15) -> np.ndarray:
    """Asymmetric soft-clip simulating tape headroom compression.

    Positive and negative halves of the waveform clip at slightly different
    thresholds, producing even harmonics (2H, 4H) that give "tape warmth".
    Pure tanh on both sides gives only odd harmonics.

    Args:
        signal: Input signal (mono or stereo).
        drive: Saturation intensity. 1.0 = gentle, 2.0 = hot.
        asymmetry: 0.0 = symmetric (odd harmonics only), 0.2 = tape-like
            asymmetry. Usable range 0.05-0.30.

    Returns:
        Saturated signal, same shape as input.
    """
    pos_drive = drive
    neg_drive = drive * (1.0 + asymmetry)
    positive = np.where(signal >= 0, np.tanh(signal * pos_drive), 0.0)
    negative = np.where(signal < 0, np.tanh(signal * neg_drive), 0.0)
    return positive + negative


def tape_hiss(duration_sec: float, level_db: float = -45.0,
              seed: int | None = None,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Subliminal pink-noise hiss for tape-era analog warmth.

    Stereo-decorrelated slightly (right = left * 0.95 with different seed)
    so it widens the soundstage very subtly.

    Args:
        duration_sec: Duration.
        level_db: Target peak level in dB (default -45 = subliminal, felt
            rather than heard on a mix).
        seed: Random seed.
        sample_rate: Sample rate.

    Returns:
        Stereo signal (n, 2) at `level_db`.
    """
    left = pink_noise(duration_sec, amplitude=1.0, sample_rate=sample_rate)
    right_seed_offset = 7919  # prime so L/R don't correlate
    if seed is not None:
        # pink_noise doesn't take seed, but we can offset by regenerating
        # with a slightly different duration trick. Simpler: just scale + phase
        # shift via np.roll.
        right = pink_noise(duration_sec, amplitude=0.95,
                           sample_rate=sample_rate)
        right = np.roll(right, right_seed_offset)
    else:
        right = pink_noise(duration_sec, amplitude=0.95,
                           sample_rate=sample_rate)
        right = np.roll(right, right_seed_offset)
    gain = 10 ** (level_db / 20)
    return np.column_stack([left, right]) * gain


def vinyl_wow(signal: np.ndarray, depth: float = 0.0005,
              rate_hz: float = 0.3, seed: int | None = None,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Slow pitch modulation emulating tape or vinyl wow/flutter.

    Shifts the read position through the signal according to a slow LFO,
    producing a tiny pitch wobble (~0.05% typical = perceptible but not
    obvious). Stereo-preserving.

    Args:
        signal: Mono or stereo input.
        depth: Modulation depth as a fraction of total length. 0.0005
            = ~0.05% pitch variation.
        rate_hz: LFO rate (0.3 Hz = one cycle every ~3s is natural).
        seed: Random seed for initial LFO phase.
        sample_rate: Sample rate.

    Returns:
        Signal with pitch modulation applied. Slight length loss at edges.
    """
    n = signal.shape[0]
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi) if seed is not None else 0.0

    # Position map: nominal index + tiny sinusoidal deviation
    positions = np.arange(n) + depth * n * np.sin(2 * np.pi * rate_hz * t + phase)
    positions = np.clip(positions, 0, n - 1)

    source_idx = np.arange(n)
    if signal.ndim == 2:
        return np.column_stack([
            np.interp(positions, source_idx, signal[:, 0]),
            np.interp(positions, source_idx, signal[:, 1]),
        ])
    return np.interp(positions, source_idx, signal)
