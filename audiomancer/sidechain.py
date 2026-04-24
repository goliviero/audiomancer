"""Sidechain ducking — classic ambient technique for "breathing" mixes.

When a trigger signal (e.g. a chime, drum, or vocal) exceeds a threshold,
the target signal (e.g. a pad) is ducked by `amount_db` with smooth attack
and release, creating space for the trigger.

Typical use: sidechain a pad against chime hits in an ambient mix.
"""

import numpy as np

from audiomancer import SAMPLE_RATE


def _as_mono(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 2:
        return np.mean(signal, axis=1)
    return signal


def envelope_follower(signal: np.ndarray,
                      attack_ms: float = 10.0,
                      release_ms: float = 200.0,
                      sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Follow the envelope of a signal with separate attack and release.

    Classic one-pole envelope follower used by audio compressors. Given a
    trigger signal, returns a smooth amplitude envelope (mono).

    Args:
        signal: Input signal (mono or stereo — stereo is mono-summed first).
        attack_ms: Time to rise when signal grows.
        release_ms: Time to fall when signal decays.
        sample_rate: Sample rate.

    Returns:
        Mono envelope in [0, peak] range, same length as input.
    """
    mono = np.abs(_as_mono(signal))

    # Time constants: smoothing coefficients per sample
    attack_samples = max(1, int(attack_ms * sample_rate / 1000.0))
    release_samples = max(1, int(release_ms * sample_rate / 1000.0))
    attack_coef = np.exp(-1.0 / attack_samples)
    release_coef = np.exp(-1.0 / release_samples)

    env = np.zeros_like(mono)
    env_val = 0.0
    for i, x in enumerate(mono):
        if x > env_val:
            env_val = attack_coef * env_val + (1 - attack_coef) * x
        else:
            env_val = release_coef * env_val + (1 - release_coef) * x
        env[i] = env_val
    return env


def sidechain_duck(target: np.ndarray, trigger: np.ndarray,
                   amount_db: float = -6.0,
                   threshold_db: float = -24.0,
                   attack_ms: float = 10.0,
                   release_ms: float = 200.0,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Duck `target` when `trigger` exceeds threshold.

    Args:
        target: Signal to be ducked (pad, bass, etc.). Mono or stereo.
        trigger: Signal driving the ducking (chime, drum). Mono or stereo.
        amount_db: Maximum gain reduction when trigger is at peak (-6 = -6 dB).
        threshold_db: Trigger level above which ducking engages (dB relative to 1.0).
        attack_ms: How fast the duck engages.
        release_ms: How fast the duck recovers.
        sample_rate: Sample rate.

    Returns:
        Target signal with time-varying gain reduction applied. Same shape
        as `target`.
    """
    # Match lengths
    n = min(len(target), len(trigger))
    target = target[:n]
    trigger = trigger[:n]

    env = envelope_follower(trigger, attack_ms=attack_ms,
                            release_ms=release_ms, sample_rate=sample_rate)

    # Convert env to dB (avoid log(0))
    env_db = 20 * np.log10(np.maximum(env, 1e-10))

    # Amount of reduction: 0 below threshold, -amount_db above peak
    # Linear ramp between threshold and threshold+20 dB
    above = np.clip((env_db - threshold_db) / 20.0, 0.0, 1.0)
    gain_reduction_db = above * amount_db  # amount_db is negative
    gain_linear = 10 ** (gain_reduction_db / 20.0)

    if target.ndim == 2:
        return target * gain_linear[:, np.newaxis]
    return target * gain_linear
