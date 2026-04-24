"""Binaural — binaural beat generation.

Generates stereo binaural beats for meditation/ambient content.
Left = carrier, Right = carrier + beat_hz. The brain perceives the difference.
"""

import numpy as np

from audiomancer import DEFAULT_AMPLITUDE, SAMPLE_RATE

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


# Meditation presets: (carrier_hz, beat_hz)
PRESETS = {
    "theta_deep":     {"carrier": 200.0, "beat": 4.0},     # Deep meditation
    "alpha_relax":    {"carrier": 200.0, "beat": 10.0},    # Relaxation
    "delta_sleep":    {"carrier": 100.0, "beat": 2.0},     # Deep sleep
    "solfeggio_528":  {"carrier": 528.0, "beat": 4.0},     # "Love frequency"
    "solfeggio_432":  {"carrier": 432.0, "beat": 6.0},     # Natural tuning
    "om_theta":       {"carrier": 136.1, "beat": 4.0},     # Om + theta
    # Extended: focus/cognition band
    "beta_13hz":      {"carrier": 264.0, "beat": 13.0},    # Calm focus / attention
    "smr_14hz":       {"carrier": 264.0, "beat": 14.0},    # Sensorimotor rhythm, sharp cognition
    "high_gamma_60hz": {"carrier": 264.0, "beat": 60.0},   # High gamma, short sessions only
}


def from_preset(name: str, duration_sec: float,
                amplitude: float = DEFAULT_AMPLITUDE,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a binaural beat from a named preset.

    Args:
        name: Preset name (see PRESETS dict).
        duration_sec: Duration in seconds.

    Returns:
        Stereo signal as (n_samples, 2).
    """
    preset = PRESETS.get(name)
    if preset is None:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name!r}. Available: {available}")
    return binaural(preset["carrier"], preset["beat"], duration_sec,
                    amplitude=amplitude, sample_rate=sample_rate)


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
    noise_l = pink_noise(duration_sec, amplitude=amplitude * pink_amount,
                         sample_rate=sample_rate)
    noise_r = pink_noise(duration_sec, amplitude=amplitude * pink_amount,
                         sample_rate=sample_rate)
    noise_stereo = np.column_stack([noise_l, noise_r])

    return beat_signal + noise_stereo
