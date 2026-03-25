"""Field — processing of field recordings.

Cleanup, noise reduction, normalization for raw field recordings.
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.effects import highpass, lowpass, reverb
from audiomancer.utils import normalize, fade_in, fade_out


def clean(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Basic cleanup: remove DC offset, subsonic rumble, and ultrasonic content.

    Args:
        signal: Raw field recording.
        sample_rate: Sample rate in Hz.

    Returns:
        Cleaned signal.
    """
    # Remove DC offset
    signal = signal - np.mean(signal, axis=0)

    # Remove subsonic rumble (<40 Hz)
    signal = highpass(signal, cutoff_hz=40.0, sample_rate=sample_rate)

    # Remove ultrasonic content (>18 kHz)
    signal = lowpass(signal, cutoff_hz=18000.0, sample_rate=sample_rate)

    return signal


def noise_gate(signal: np.ndarray, threshold_db: float = -40.0,
               attack_ms: float = 5.0, release_ms: float = 50.0,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Simple noise gate — attenuate signal below threshold.

    Args:
        signal: Input signal.
        threshold_db: Gate threshold in dB.
        attack_ms: Attack time in ms.
        release_ms: Release time in ms.
        sample_rate: Sample rate in Hz.

    Returns:
        Gated signal.
    """
    threshold_linear = 10 ** (threshold_db / 20)
    attack_samples = int(sample_rate * attack_ms / 1000)
    release_samples = int(sample_rate * release_ms / 1000)

    if signal.ndim == 2:
        envelope = np.max(np.abs(signal), axis=1)
    else:
        envelope = np.abs(signal)

    # Build gate envelope (0 or 1)
    gate = (envelope > threshold_linear).astype(np.float64)

    # Smooth attack/release
    smoothing = max(attack_samples, release_samples)
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        gate = np.convolve(gate, kernel, mode="same")
        gate = np.clip(gate, 0.0, 1.0)

    if signal.ndim == 2:
        gate = gate[:, np.newaxis]

    return signal * gate


def process_field(signal: np.ndarray, sample_rate: int = SAMPLE_RATE,
                  reverb_wet: float = 0.3,
                  fade_in_sec: float = 2.0,
                  fade_out_sec: float = 3.0) -> np.ndarray:
    """Full field recording processing pipeline.

    Applies: clean → noise gate → reverb → fades → normalize.

    Args:
        signal: Raw field recording.
        sample_rate: Sample rate in Hz.
        reverb_wet: Reverb wet level (0.0 to 1.0).
        fade_in_sec: Fade-in duration.
        fade_out_sec: Fade-out duration.

    Returns:
        Processed signal ready for mixing.
    """
    signal = clean(signal, sample_rate=sample_rate)
    signal = noise_gate(signal, threshold_db=-50.0, sample_rate=sample_rate)
    if reverb_wet > 0:
        signal = reverb(signal, room_size=0.6, wet_level=reverb_wet,
                        sample_rate=sample_rate)
    signal = fade_in(signal, fade_in_sec, sample_rate=sample_rate)
    signal = fade_out(signal, fade_out_sec, sample_rate=sample_rate)
    signal = normalize(signal, target_db=-1.0)
    return signal
