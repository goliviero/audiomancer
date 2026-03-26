"""Layers — mixing, layering, and superposition of audio stems.

Combines multiple audio signals with volume control, fades, and alignment.
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.utils import mono_to_stereo, pad_to_length


def mix(signals: list[np.ndarray],
        volumes_db: list[float] | None = None) -> np.ndarray:
    """Mix (sum) multiple signals at given volume levels.

    All signals are padded to the longest. If any is stereo, result is stereo.

    Args:
        signals: List of signals to mix.
        volumes_db: Per-signal volume in dB. Defaults to 0dB for all.

    Returns:
        Mixed signal.
    """
    if not signals:
        return np.array([], dtype=np.float64)

    if volumes_db is None:
        volumes_db = [0.0] * len(signals)

    max_len = max(s.shape[0] for s in signals)
    any_stereo = any(s.ndim == 2 for s in signals)

    if any_stereo:
        result = np.zeros((max_len, 2), dtype=np.float64)
    else:
        result = np.zeros(max_len, dtype=np.float64)

    for sig, vol_db in zip(signals, volumes_db):
        gain = 10 ** (vol_db / 20)
        if any_stereo and sig.ndim == 1:
            sig = mono_to_stereo(sig)
        padded = pad_to_length(sig, max_len)
        result = result + padded * gain

    return result


def layer_at_offset(base: np.ndarray, overlay: np.ndarray,
                    offset_sec: float, volume_db: float = 0.0,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Layer an overlay signal onto a base at a given time offset.

    Args:
        base: Base signal.
        overlay: Signal to layer on top.
        offset_sec: Time offset in seconds where overlay starts.
        volume_db: Volume of overlay in dB.
        sample_rate: Sample rate.

    Returns:
        Combined signal (may be longer than base).
    """
    offset_samples = int(sample_rate * offset_sec)
    gain = 10 ** (volume_db / 20)

    overlay_scaled = overlay * gain
    total_len = max(base.shape[0], offset_samples + overlay.shape[0])

    result = pad_to_length(base.copy(), total_len)
    overlay_padded = np.zeros_like(result)

    end = offset_samples + overlay_scaled.shape[0]
    if result.ndim == 1 and overlay_scaled.ndim == 1:
        overlay_padded[offset_samples:end] = overlay_scaled
    elif result.ndim == 2:
        if overlay_scaled.ndim == 1:
            overlay_scaled = mono_to_stereo(overlay_scaled)
        overlay_padded[offset_samples:end] = overlay_scaled
    else:
        # base is mono, overlay is stereo — promote base
        result = mono_to_stereo(result)
        overlay_padded = np.zeros_like(result)
        overlay_padded[offset_samples:end] = overlay_scaled

    return result + overlay_padded


def crossfade(signal_a: np.ndarray, signal_b: np.ndarray,
              crossfade_sec: float,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Crossfade from signal_a to signal_b.

    The last crossfade_sec of signal_a overlap with the first
    crossfade_sec of signal_b.

    Args:
        signal_a: First signal.
        signal_b: Second signal.
        crossfade_sec: Duration of the crossfade in seconds.
        sample_rate: Sample rate.

    Returns:
        Combined signal.
    """
    xf_samples = int(sample_rate * crossfade_sec)
    xf_samples = min(xf_samples, len(signal_a), len(signal_b))

    fade_out_ramp = np.linspace(1.0, 0.0, xf_samples)
    fade_in_ramp = np.linspace(0.0, 1.0, xf_samples)

    # Parts: A before crossfade | crossfade zone | B after crossfade
    a_pre = signal_a[:-xf_samples]
    a_xf = signal_a[-xf_samples:]
    b_xf = signal_b[:xf_samples]
    b_post = signal_b[xf_samples:]

    if a_xf.ndim == 2:
        fade_out_ramp = fade_out_ramp[:, np.newaxis]
        fade_in_ramp = fade_in_ramp[:, np.newaxis]

    crossfade_zone = a_xf * fade_out_ramp + b_xf * fade_in_ramp

    return np.concatenate([a_pre, crossfade_zone, b_post])


def layer(stems: list[np.ndarray], volumes: list[float] | None = None) -> np.ndarray:
    """Superpose multiple stems with linear volume control.

    Like mix() but takes linear volumes (0.0–1.0) instead of dB.

    Args:
        stems: List of audio arrays.
        volumes: Linear volume per stem. Defaults to 1.0 for all.

    Returns:
        Mixed signal (stereo if any input is stereo).
    """
    if not stems:
        return np.array([], dtype=np.float64)
    if volumes is None:
        volumes = [1.0] * len(stems)
    # Convert linear to dB for mix()
    volumes_db = [20 * np.log10(max(v, 1e-10)) for v in volumes]
    return mix(stems, volumes_db=volumes_db)


def loop_seamless(audio: np.ndarray, total_duration_sec: float,
                  crossfade_sec: float = 3.0,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Loop audio seamlessly with crossfade at junctions.

    Args:
        audio: Source signal to loop.
        total_duration_sec: Target duration in seconds.
        crossfade_sec: Crossfade overlap at each junction.
        sample_rate: Sample rate in Hz.

    Returns:
        Looped signal of approximately the target duration.
    """
    target_samples = int(sample_rate * total_duration_sec)

    if len(audio) >= target_samples:
        return audio[:target_samples]

    result = audio.copy()
    while len(result) < target_samples:
        result = crossfade(result, audio.copy(), crossfade_sec=crossfade_sec,
                           sample_rate=sample_rate)

    return result[:target_samples]


def normalize_lufs(signal: np.ndarray, target_lufs: float = -14.0,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """ITU-R BS.1770 LUFS normalization with K-weighting.

    Applies K-weighting filters (high shelf + high-pass) before measuring
    loudness, then adjusts gain to match target LUFS.

    Args:
        signal: Input signal.
        target_lufs: Target loudness (default -14 for YouTube/streaming).
        sample_rate: Sample rate.

    Returns:
        Normalized signal.
    """
    current = measure_lufs(signal, sample_rate=sample_rate)
    if not np.isfinite(current):
        return signal
    gain_db = target_lufs - current
    gain_linear = 10 ** (gain_db / 20)
    result = signal * gain_linear

    # Safety: prevent digital clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    return result


def measure_lufs(signal: np.ndarray,
                 sample_rate: int = SAMPLE_RATE) -> float:
    """Measure loudness in LUFS (ITU-R BS.1770-4).

    Applies K-weighting (2-stage biquad: high shelf + RLB high-pass),
    then computes mean square across all channels.

    Args:
        signal: Input signal (mono or stereo).
        sample_rate: Sample rate.

    Returns:
        Loudness in LUFS. Returns -np.inf for silent signals.
    """
    from scipy.signal import sosfilt

    # K-weighting filter coefficients (designed for 48 kHz, works well at 44.1 kHz)
    # Stage 1: Pre-filter (high shelf +4 dB at ~1681 Hz)
    # Stage 2: RLB weighting (high-pass at ~38 Hz)
    sos_k = _k_weighting_sos(sample_rate)

    if signal.ndim == 1:
        channels = [signal]
    else:
        channels = [signal[:, i] for i in range(signal.shape[1])]

    mean_sq_sum = 0.0
    for ch in channels:
        filtered = sosfilt(sos_k, ch)
        mean_sq_sum += np.mean(filtered ** 2)

    mean_sq = mean_sq_sum / len(channels)

    if mean_sq <= 0:
        return -np.inf
    return -0.691 + 10 * np.log10(mean_sq)


def _k_weighting_sos(sample_rate: int) -> np.ndarray:
    """Build K-weighting SOS filter for ITU-R BS.1770.

    Two cascaded biquads:
    1. Pre-filter: high shelf boosting high frequencies (~+4 dB above 1.5 kHz)
    2. RLB weighting: 2nd-order high-pass at ~38 Hz
    """

    # Stage 1: Pre-filter (high shelf)
    # Analog prototype: 2nd order shelving filter
    # Designed to match ITU-R BS.1770 curve
    f0 = 1681.974450955533
    Q = 0.7071752369554196
    G = 3.999843853973347  # dB boost

    K = np.tan(np.pi * f0 / sample_rate)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.5

    a0 = 1 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0
    b1 = 2 * (K * K - Vh) / a0
    b2 = (Vh - Vb * K / Q + K * K) / a0
    a1 = 2 * (K * K - 1) / a0
    a2 = (1 - K / Q + K * K) / a0

    sos1 = [b0, b1, b2, 1.0, a1, a2]

    # Stage 2: RLB high-pass at ~38 Hz (2nd order Butterworth)
    f1 = 38.13547087602444
    K2 = np.tan(np.pi * f1 / sample_rate)
    Q2 = 0.5003270373238773

    a0_2 = 1 + K2 / Q2 + K2 * K2
    b0_2 = 1 / a0_2
    b1_2 = -2 / a0_2
    b2_2 = 1 / a0_2
    a1_2 = 2 * (K2 * K2 - 1) / a0_2
    a2_2 = (1 - K2 / Q2 + K2 * K2) / a0_2

    sos2 = [b0_2, b1_2, b2_2, 1.0, a1_2, a2_2]

    return np.array([sos1, sos2])
