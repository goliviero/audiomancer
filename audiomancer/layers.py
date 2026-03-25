"""Layers — mixing, layering, and superposition of audio stems.

Combines multiple audio signals with volume control, fades, and alignment.
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.utils import pad_to_length, mono_to_stereo


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


def normalize_lufs(signal: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
    """Approximate LUFS normalization using RMS.

    True LUFS (ITU-R BS.1770) requires K-weighted filtering and gating.
    For continuous ambient audio, RMS is a close approximation.

    Args:
        signal: Input signal.
        target_lufs: Target loudness (default -14 for YouTube).

    Returns:
        Normalized signal.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0:
        return signal
    current_lufs = 20 * np.log10(rms)
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    result = signal * gain_linear

    # Safety: prevent digital clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    return result
