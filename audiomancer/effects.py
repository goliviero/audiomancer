"""Effects — audio processing via pedalboard (Spotify) and scipy.

Pedalboard provides pro-grade VST-quality effects.
Scipy fallbacks for when pedalboard is unavailable.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audiomancer import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Scipy-based effects (always available)
# ---------------------------------------------------------------------------

def lowpass(signal: np.ndarray, cutoff_hz: float, order: int = 4,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth low-pass filter."""
    nyquist = sample_rate / 2
    cutoff = np.clip(cutoff_hz / nyquist, 0.001, 0.99)
    sos = butter(order, cutoff, btype="low", output="sos")
    if signal.ndim == 2:
        return np.column_stack([
            sosfiltfilt(sos, signal[:, 0]),
            sosfiltfilt(sos, signal[:, 1]),
        ])
    return sosfiltfilt(sos, signal)


def highpass(signal: np.ndarray, cutoff_hz: float, order: int = 4,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth high-pass filter."""
    nyquist = sample_rate / 2
    cutoff = np.clip(cutoff_hz / nyquist, 0.001, 0.99)
    sos = butter(order, cutoff, btype="high", output="sos")
    if signal.ndim == 2:
        return np.column_stack([
            sosfiltfilt(sos, signal[:, 0]),
            sosfiltfilt(sos, signal[:, 1]),
        ])
    return sosfiltfilt(sos, signal)


# ---------------------------------------------------------------------------
# Pedalboard-based effects (pro-grade)
# ---------------------------------------------------------------------------

def reverb(signal: np.ndarray, room_size: float = 0.5, damping: float = 0.5,
           wet_level: float = 0.33, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply reverb via pedalboard.

    Args:
        signal: Input signal (mono or stereo).
        room_size: Room size (0.0 to 1.0).
        damping: High-frequency damping (0.0 to 1.0).
        wet_level: Wet/dry mix (0.0 to 1.0).
        sample_rate: Sample rate in Hz.

    Returns:
        Processed signal.
    """
    import pedalboard as pb

    board = pb.Pedalboard([
        pb.Reverb(room_size=room_size, damping=damping, wet_level=wet_level),
    ])
    return _process_board(board, signal, sample_rate)


def delay(signal: np.ndarray, delay_seconds: float = 0.3,
          feedback: float = 0.3, mix: float = 0.3,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply delay via pedalboard."""
    import pedalboard as pb

    board = pb.Pedalboard([
        pb.Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix),
    ])
    return _process_board(board, signal, sample_rate)


def chorus(signal: np.ndarray, rate_hz: float = 1.0,
           depth: float = 0.25, mix: float = 0.5,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply chorus via pedalboard."""
    import pedalboard as pb

    board = pb.Pedalboard([
        pb.Chorus(rate_hz=rate_hz, depth=depth, mix=mix),
    ])
    return _process_board(board, signal, sample_rate)


def compress(signal: np.ndarray, threshold_db: float = -20.0,
             ratio: float = 4.0, attack_ms: float = 1.0,
             release_ms: float = 100.0,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply compression via pedalboard."""
    import pedalboard as pb

    board = pb.Pedalboard([
        pb.Compressor(
            threshold_db=threshold_db, ratio=ratio,
            attack_ms=attack_ms, release_ms=release_ms,
        ),
    ])
    return _process_board(board, signal, sample_rate)


def chain(signal: np.ndarray, effects: list,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a chain of pedalboard effects.

    Args:
        signal: Input signal.
        effects: List of pedalboard.Plugin instances.
        sample_rate: Sample rate.

    Returns:
        Processed signal.
    """
    import pedalboard as pb

    board = pb.Pedalboard(effects)
    return _process_board(board, signal, sample_rate)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _process_board(board, signal: np.ndarray,
                   sample_rate: int) -> np.ndarray:
    """Run a pedalboard on a signal, handling shape conventions.

    Pedalboard expects (channels, samples) float32.
    We work with (samples,) or (samples, channels) float64.
    """
    was_mono = signal.ndim == 1
    sig = signal.astype(np.float32)

    if was_mono:
        sig = sig[np.newaxis, :]  # (1, samples)
    else:
        sig = sig.T  # (channels, samples)

    processed = board(sig, sample_rate)

    if was_mono:
        return processed[0].astype(np.float64)
    return processed.T.astype(np.float64)
