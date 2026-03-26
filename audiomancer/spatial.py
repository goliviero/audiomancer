"""Spatial — stereo positioning, width control, and movement.

Pan, auto-pan, stereo widening, and mid/side processing
for immersive ambient sound design.
All functions operate on numpy arrays. Mono = (n,), Stereo = (n, 2).
"""

import numpy as np

from audiomancer import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Static panning
# ---------------------------------------------------------------------------

def pan(signal: np.ndarray, position: float = 0.0) -> np.ndarray:
    """Pan a signal in the stereo field using constant-power panning.

    Args:
        signal: Input signal (mono or stereo).
        position: Pan position (-1.0 = hard left, 0.0 = center, 1.0 = hard right).

    Returns:
        Stereo signal.
    """
    position = np.clip(position, -1.0, 1.0)
    if signal.ndim == 2:
        mono = np.mean(signal, axis=1)
    else:
        mono = signal

    # Constant-power panning law
    angle = (position + 1) * np.pi / 4  # 0 to pi/2
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    return np.column_stack([mono * left_gain, mono * right_gain])


# ---------------------------------------------------------------------------
# Auto-pan (LFO-driven)
# ---------------------------------------------------------------------------

def auto_pan(signal: np.ndarray, rate_hz: float = 0.1,
             depth: float = 1.0, center: float = 0.0,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Automatically pan a signal back and forth using a sine LFO.

    Creates gentle stereo movement for ambient textures.

    Args:
        signal: Input signal (mono or stereo).
        rate_hz: LFO speed in Hz (0.05 = 20s cycle, 0.1 = 10s cycle).
        depth: Pan depth (0.0 = no movement, 1.0 = full left-right).
        center: Center pan position (-1.0 to 1.0).
        sample_rate: Sample rate.

    Returns:
        Stereo signal with panning movement.
    """
    if signal.ndim == 2:
        mono = np.mean(signal, axis=1)
    else:
        mono = signal

    n = len(mono)
    t = np.linspace(0, n / sample_rate, n, endpoint=False)
    lfo = center + depth * np.sin(2 * np.pi * rate_hz * t)
    lfo = np.clip(lfo, -1.0, 1.0)

    # Constant-power panning per sample
    angle = (lfo + 1) * np.pi / 4
    left = mono * np.cos(angle)
    right = mono * np.sin(angle)

    return np.column_stack([left, right])


# ---------------------------------------------------------------------------
# Stereo width
# ---------------------------------------------------------------------------

def stereo_width(signal: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Adjust stereo width of a signal.

    Args:
        signal: Stereo input signal.
        width: Width factor.
            0.0 = mono (mid only)
            1.0 = unchanged
            2.0 = extra wide (boosted side)
            Values > 2.0 create extreme widening effects.

    Returns:
        Stereo signal with adjusted width.
    """
    if signal.ndim == 1:
        return np.column_stack([signal, signal])

    mid = (signal[:, 0] + signal[:, 1]) * 0.5
    side = (signal[:, 0] - signal[:, 1]) * 0.5

    left = mid + side * width
    right = mid - side * width

    return np.column_stack([left, right])


# ---------------------------------------------------------------------------
# Mid/Side encoding & decoding
# ---------------------------------------------------------------------------

def encode_mid_side(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a stereo signal into mid and side components.

    Mid = (L + R) / 2 (center content)
    Side = (L - R) / 2 (stereo content)

    Args:
        signal: Stereo input signal (n, 2).

    Returns:
        Tuple of (mid, side) — both mono arrays.
    """
    if signal.ndim == 1:
        return signal.copy(), np.zeros_like(signal)
    mid = (signal[:, 0] + signal[:, 1]) * 0.5
    side = (signal[:, 0] - signal[:, 1]) * 0.5
    return mid, side


def decode_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Decode mid/side back to stereo L/R.

    Args:
        mid: Mid (center) signal.
        side: Side (stereo) signal.

    Returns:
        Stereo signal (n, 2).
    """
    left = mid + side
    right = mid - side
    return np.column_stack([left, right])


# ---------------------------------------------------------------------------
# Haas effect (stereo widening via micro-delay)
# ---------------------------------------------------------------------------

def haas_width(signal: np.ndarray, delay_ms: float = 15.0,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Widen a mono signal using the Haas effect (precedence effect).

    Delays one channel by a few milliseconds to create perceived width
    without changing the tonal character. Classic studio trick.

    Args:
        signal: Input signal (mono or stereo).
        delay_ms: Delay time in milliseconds (10-30ms typical).
        sample_rate: Sample rate.

    Returns:
        Stereo signal with Haas widening.
    """
    if signal.ndim == 2:
        mono = np.mean(signal, axis=1)
    else:
        mono = signal

    delay_samples = int(sample_rate * delay_ms / 1000)
    delayed = np.zeros(len(mono) + delay_samples)
    delayed[delay_samples:] = mono[:len(delayed) - delay_samples]
    # Trim to original length
    delayed = delayed[:len(mono)]

    return np.column_stack([mono, delayed])


# ---------------------------------------------------------------------------
# Binaural panning (frequency-dependent)
# ---------------------------------------------------------------------------

def rotate(signal: np.ndarray, duration_sec: float | None = None,
           revolutions: float = 1.0,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Rotate a signal around the stereo field.

    Creates a smooth circular panning motion. Great for drones and pads.

    Args:
        signal: Input signal (mono or stereo).
        duration_sec: Duration (default: signal length).
        revolutions: Number of full rotations (negative = reverse).
        sample_rate: Sample rate.

    Returns:
        Stereo signal with rotation.
    """
    if signal.ndim == 2:
        mono = np.mean(signal, axis=1)
    else:
        mono = signal

    n = len(mono)
    if duration_sec is None:
        duration_sec = n / sample_rate

    t = np.linspace(0, duration_sec, n, endpoint=False)
    angle = 2 * np.pi * revolutions * t / duration_sec

    # Circular panning (constant power)
    left = mono * np.cos(angle) * np.sqrt(2) / 2 + mono * 0.5
    right = mono * np.sin(angle) * np.sqrt(2) / 2 + mono * 0.5

    return np.column_stack([left, right])
