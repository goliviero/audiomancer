"""Signal utilities — pure functions on numpy arrays.

A Signal is a np.ndarray with shape (n_samples,) for mono or (n_samples, 2) for stereo.
Every function here is stateless: Signal in, Signal out.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE


def duration_samples(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Convert a duration in seconds to a number of samples."""
    return int(sample_rate * duration_sec)


def duration_seconds(n_samples: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Convert a number of samples to duration in seconds."""
    return n_samples / sample_rate


def is_stereo(signal: np.ndarray) -> bool:
    """Check if a signal is stereo (2 channels)."""
    return signal.ndim == 2 and signal.shape[1] == 2


def is_mono(signal: np.ndarray) -> bool:
    """Check if a signal is mono (1D)."""
    return signal.ndim == 1


def mono_to_stereo(signal: np.ndarray) -> np.ndarray:
    """Convert a mono signal to stereo by duplicating the channel."""
    if is_stereo(signal):
        return signal
    return np.column_stack([signal, signal])


def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert a stereo signal to mono by averaging channels."""
    if is_mono(signal):
        return signal
    return np.mean(signal, axis=1)


def silence(duration_sec: float, stereo: bool = False,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a silent signal of the given duration."""
    n = duration_samples(duration_sec, sample_rate)
    if stereo:
        return np.zeros((n, 2), dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def normalize_peak(signal: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize signal so peak amplitude reaches target_db.

    Args:
        signal: Input signal (mono or stereo).
        target_db: Target peak level in dB (e.g., -1.0).

    Returns:
        Normalized signal.
    """
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    target_linear = 10 ** (target_db / 20)
    return signal * (target_linear / peak)


def pad_to_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad signal with zeros to reach target length. No-op if already long enough."""
    current = signal.shape[0]
    if current >= target_samples:
        return signal
    pad_amount = target_samples - current
    if is_stereo(signal):
        padding = np.zeros((pad_amount, 2), dtype=signal.dtype)
    else:
        padding = np.zeros(pad_amount, dtype=signal.dtype)
    return np.concatenate([signal, padding])


def trim(signal: np.ndarray, start_sec: float = 0.0, end_sec: float | None = None,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Trim signal to [start_sec, end_sec). None means end of signal."""
    start = duration_samples(start_sec, sample_rate)
    if end_sec is not None:
        end = duration_samples(end_sec, sample_rate)
        return signal[start:end]
    return signal[start:]


def concat(*signals: np.ndarray) -> np.ndarray:
    """Concatenate multiple signals end-to-end."""
    return np.concatenate(signals)


def mix_signals(signals: list[np.ndarray], volumes_db: list[float] | None = None) -> np.ndarray:
    """Mix (sum) multiple signals at given volume levels.

    All signals are padded to the length of the longest one.

    Args:
        signals: List of signals to mix.
        volumes_db: Optional per-signal volume in dB. Defaults to 0dB (unity) for all.

    Returns:
        Mixed signal.
    """
    if not signals:
        return np.array([], dtype=np.float64)

    if volumes_db is None:
        volumes_db = [0.0] * len(signals)

    max_len = max(s.shape[0] for s in signals)
    result = np.zeros(max_len, dtype=np.float64)

    # If any signal is stereo, mix in stereo
    any_stereo = any(is_stereo(s) for s in signals)
    if any_stereo:
        result = np.zeros((max_len, 2), dtype=np.float64)

    for sig, vol_db in zip(signals, volumes_db):
        gain = 10 ** (vol_db / 20)
        if any_stereo and is_mono(sig):
            sig = mono_to_stereo(sig)
        padded = pad_to_length(sig, max_len)
        result = result + padded * gain

    return result
