"""Envelope — amplitude shaping for audio signals.

ADSR, multi-segment, exponential, and specialty envelopes
for ambient sound design. All envelopes return modulation arrays
centered at 1.0 (compatible with apply_amplitude_mod).
"""

import numpy as np

from audiomancer import SAMPLE_RATE


# ---------------------------------------------------------------------------
# ADSR envelope
# ---------------------------------------------------------------------------

def adsr(duration_sec: float, attack: float = 0.1, decay: float = 0.2,
         sustain: float = 0.7, release: float = 0.5,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Classic ADSR envelope.

    Args:
        duration_sec: Total duration in seconds.
        attack: Attack time in seconds (0 → peak).
        decay: Decay time in seconds (peak → sustain level).
        sustain: Sustain level (0.0 to 1.0).
        release: Release time in seconds (sustain → 0).
        sample_rate: Sample rate.

    Returns:
        Envelope array (n_samples,) with values in [0, 1].
    """
    n = int(sample_rate * duration_sec)
    n_attack = int(sample_rate * attack)
    n_decay = int(sample_rate * decay)
    n_release = int(sample_rate * release)
    n_sustain = max(0, n - n_attack - n_decay - n_release)

    parts = []

    # Attack: 0 → 1
    if n_attack > 0:
        parts.append(np.linspace(0, 1, n_attack, endpoint=False))

    # Decay: 1 → sustain
    if n_decay > 0:
        parts.append(np.linspace(1, sustain, n_decay, endpoint=False))

    # Sustain: hold at sustain level
    if n_sustain > 0:
        parts.append(np.full(n_sustain, sustain))

    # Release: sustain → 0
    if n_release > 0:
        parts.append(np.linspace(sustain, 0, n_release, endpoint=True))

    if not parts:
        return np.zeros(n)

    env = np.concatenate(parts)
    # Trim or pad to exact length
    if len(env) > n:
        env = env[:n]
    elif len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env


# ---------------------------------------------------------------------------
# Exponential ADSR
# ---------------------------------------------------------------------------

def adsr_exp(duration_sec: float, attack: float = 0.1, decay: float = 0.2,
             sustain: float = 0.7, release: float = 0.5,
             curve: float = 3.0,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Exponential ADSR — more natural-sounding than linear.

    Args:
        duration_sec: Total duration.
        attack: Attack time.
        decay: Decay time.
        sustain: Sustain level.
        release: Release time.
        curve: Exponential curve factor (1.0 = linear, higher = steeper).
        sample_rate: Sample rate.

    Returns:
        Envelope array (n_samples,) with values in [0, 1].
    """
    n = int(sample_rate * duration_sec)
    n_attack = int(sample_rate * attack)
    n_decay = int(sample_rate * decay)
    n_release = int(sample_rate * release)
    n_sustain = max(0, n - n_attack - n_decay - n_release)

    parts = []

    # Attack: exponential rise 0 → 1
    if n_attack > 0:
        t = np.linspace(0, 1, n_attack, endpoint=False)
        parts.append(t ** (1.0 / curve))  # concave up

    # Decay: exponential fall 1 → sustain
    if n_decay > 0:
        t = np.linspace(0, 1, n_decay, endpoint=False)
        parts.append(1.0 - (1.0 - sustain) * (t ** (1.0 / curve)))

    # Sustain
    if n_sustain > 0:
        parts.append(np.full(n_sustain, sustain))

    # Release: exponential fall sustain → 0
    if n_release > 0:
        t = np.linspace(0, 1, n_release, endpoint=True)
        parts.append(sustain * (1.0 - t ** (1.0 / curve)))

    if not parts:
        return np.zeros(n)

    env = np.concatenate(parts)
    if len(env) > n:
        env = env[:n]
    elif len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env


# ---------------------------------------------------------------------------
# AR envelope (attack-release, no sustain)
# ---------------------------------------------------------------------------

def ar(duration_sec: float, attack: float = 0.5,
       curve: float = 2.0,
       sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Attack-Release envelope — simple swell shape.

    Args:
        duration_sec: Total duration.
        attack: Attack fraction of total (0.0 to 1.0).
            0.5 = symmetric rise and fall.
        curve: Exponential curve (1.0 = linear, higher = rounder).
        sample_rate: Sample rate.

    Returns:
        Envelope array.
    """
    n = int(sample_rate * duration_sec)
    n_attack = int(n * np.clip(attack, 0.01, 0.99))
    n_release = n - n_attack

    t_a = np.linspace(0, 1, n_attack, endpoint=False)
    t_r = np.linspace(0, 1, n_release, endpoint=True)

    rise = t_a ** (1.0 / curve)
    fall = 1.0 - t_r ** (1.0 / curve)

    return np.concatenate([rise, fall])


# ---------------------------------------------------------------------------
# Multi-segment envelope
# ---------------------------------------------------------------------------

def segments(points: list[tuple[float, float]], duration_sec: float,
             curve: float = 1.0,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Multi-segment envelope from breakpoints.

    More flexible than fade_envelope in compose.py — supports
    exponential curves between points.

    Args:
        points: List of (time_sec, value) breakpoints.
            Must start at time 0.0. Values typically 0.0 to 1.0.
        duration_sec: Total duration.
        curve: Curve factor (1.0 = linear, >1 = exponential).
        sample_rate: Sample rate.

    Returns:
        Envelope array.
    """
    n = int(sample_rate * duration_sec)
    env = np.zeros(n)

    # Sort by time
    pts = sorted(points, key=lambda p: p[0])

    for i in range(len(pts) - 1):
        t_start, v_start = pts[i]
        t_end, v_end = pts[i + 1]
        s_start = int(t_start * sample_rate)
        s_end = int(t_end * sample_rate)
        s_start = max(0, min(s_start, n))
        s_end = max(0, min(s_end, n))
        seg_len = s_end - s_start
        if seg_len <= 0:
            continue
        t = np.linspace(0, 1, seg_len, endpoint=False)
        if curve == 1.0:
            interp = t
        else:
            interp = t ** curve if v_end > v_start else 1.0 - (1.0 - t) ** curve
        env[s_start:s_end] = v_start + (v_end - v_start) * interp

    # Hold last value to end
    if pts:
        last_sample = int(pts[-1][0] * sample_rate)
        if last_sample < n:
            env[last_sample:] = pts[-1][1]

    return env


# ---------------------------------------------------------------------------
# Specialty envelopes for ambient
# ---------------------------------------------------------------------------

def breathing(duration_sec: float, breath_rate: float = 0.1,
              depth: float = 0.3, floor: float = 0.7,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Breathing envelope — slow swell like inhale/exhale.

    Perfect for pads and drones that need gentle, organic movement.

    Args:
        duration_sec: Total duration.
        breath_rate: Breaths per second (0.1 = 1 breath every 10 seconds).
        depth: Breath depth (0.0 = no movement, 1.0 = full 0-to-1).
        floor: Minimum level (envelope = floor ± depth).
        sample_rate: Sample rate.

    Returns:
        Envelope array with values in [floor, floor + depth], clipped to [0, 1].
    """
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # Raised cosine = smooth breathing shape
    env = floor + depth * 0.5 * (1 - np.cos(2 * np.pi * breath_rate * t))
    return np.clip(env, 0.0, 1.0)


def swell(duration_sec: float, peak_time: float = 0.5,
          hold: float = 0.0, curve: float = 2.0,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Swell envelope — gradual build to a peak, then fade.

    Args:
        duration_sec: Total duration.
        peak_time: Where the peak occurs (fraction, 0.0 to 1.0).
        hold: Hold time at peak (fraction of total duration).
        curve: Steepness (1.0 = linear, higher = more dramatic).
        sample_rate: Sample rate.

    Returns:
        Envelope array.
    """
    n = int(sample_rate * duration_sec)
    peak_frac = np.clip(peak_time, 0.01, 0.99)
    hold_frac = np.clip(hold, 0.0, 1.0 - peak_frac)

    n_rise = int(n * peak_frac)
    n_hold = int(n * hold_frac)
    n_fall = n - n_rise - n_hold

    parts = []
    if n_rise > 0:
        t = np.linspace(0, 1, n_rise, endpoint=False)
        parts.append(t ** (1.0 / curve))
    if n_hold > 0:
        parts.append(np.ones(n_hold))
    if n_fall > 0:
        t = np.linspace(0, 1, n_fall, endpoint=True)
        parts.append(1.0 - t ** (1.0 / curve))

    env = np.concatenate(parts) if parts else np.zeros(n)
    if len(env) > n:
        env = env[:n]
    elif len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env


def gate_pattern(duration_sec: float, pattern: list[float],
                 step_sec: float = 0.5, smoothing_ms: float = 5.0,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Rhythmic gate envelope from a step pattern.

    Creates rhythmic chopping effects. Use with long pads
    for pseudo-rhythmic ambient textures.

    Args:
        duration_sec: Total duration.
        pattern: List of amplitude values (0.0 to 1.0) for each step.
            Pattern repeats to fill duration.
        step_sec: Duration of each step.
        smoothing_ms: Smoothing time to avoid clicks.
        sample_rate: Sample rate.

    Returns:
        Envelope array.
    """
    n = int(sample_rate * duration_sec)
    step_samples = int(sample_rate * step_sec)
    smooth_samples = max(1, int(sample_rate * smoothing_ms / 1000))

    env = np.zeros(n)
    for i in range(0, n, step_samples):
        step_idx = (i // step_samples) % len(pattern)
        end = min(i + step_samples, n)
        env[i:end] = pattern[step_idx]

    # Smooth transitions with a small moving average
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        env = np.convolve(env, kernel, mode="same")

    return np.clip(env, 0.0, 1.0)
