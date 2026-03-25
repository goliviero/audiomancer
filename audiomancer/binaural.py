"""Binaural — binaural beat generation.

Generates stereo binaural beats for meditation/ambient content.
Left = carrier, Right = carrier + beat_hz. The brain perceives the difference.
"""

import numpy as np

from audiomancer import SAMPLE_RATE, DEFAULT_AMPLITUDE


# Common brainwave frequency bands
BANDS = {
    "delta":   (0.5, 4.0),   # Deep sleep
    "theta":   (4.0, 8.0),   # Meditation, drowsiness
    "alpha":   (8.0, 13.0),  # Relaxation, calm focus
    "beta":    (13.0, 30.0), # Active thinking
    "gamma":   (30.0, 100.0), # High-level cognition
}

# Popular carrier frequencies
CARRIERS = {
    "schumann": 7.83,     # Earth resonance
    "om":       136.1,    # Sanskrit Om frequency
    "solfeggio_396": 396.0,
    "solfeggio_432": 432.0,
    "solfeggio_528": 528.0,
    "standard": 200.0,    # Common default
}


def binaural(carrier_hz: float, beat_hz: float, duration_sec: float,
             amplitude: float = DEFAULT_AMPLITUDE,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a stereo binaural beat.

    Args:
        carrier_hz: Base frequency for both ears (e.g., 200 Hz).
        beat_hz: Desired binaural beat frequency (e.g., 10 Hz for alpha).
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Stereo signal as (n_samples, 2) numpy array.
    """
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    left = amplitude * np.sin(2 * np.pi * carrier_hz * t)
    right = amplitude * np.sin(2 * np.pi * (carrier_hz + beat_hz) * t)
    return np.column_stack([left, right])


def binaural_layered(carrier_hz: float, beat_hz: float, duration_sec: float,
                     pink_amount: float = 0.1,
                     amplitude: float = DEFAULT_AMPLITUDE,
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a binaural beat with pink noise bed for comfort.

    Args:
        carrier_hz: Base frequency.
        beat_hz: Binaural beat frequency.
        duration_sec: Duration in seconds.
        pink_amount: Mix level of pink noise (0.0 to 1.0).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Stereo signal.
    """
    from audiomancer.synth import pink_noise

    beat_signal = binaural(carrier_hz, beat_hz, duration_sec,
                           amplitude=amplitude * (1.0 - pink_amount),
                           sample_rate=sample_rate)

    # Pink noise bed (stereo)
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng(42)
    noise_l = pink_noise(duration_sec, amplitude=amplitude * pink_amount,
                         sample_rate=sample_rate)
    noise_r = pink_noise(duration_sec, amplitude=amplitude * pink_amount,
                         sample_rate=sample_rate)
    noise_stereo = np.column_stack([noise_l, noise_r])

    return beat_signal + noise_stereo
