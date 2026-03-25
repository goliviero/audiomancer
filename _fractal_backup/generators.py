"""Signal generators — factory functions that produce numpy arrays.

Each generator returns a raw Signal (np.ndarray) at the given sample rate.
Inspired by Akasha Portal's generate_binaural.py patterns.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE


def _time_axis(duration_sec: float, sample_rate: int) -> np.ndarray:
    """Create a time axis array for the given duration."""
    n_samples = int(sample_rate * duration_sec)
    return np.linspace(0, duration_sec, n_samples, endpoint=False)


def sine(frequency: float, duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono sine wave.

    Args:
        frequency: Frequency in Hz.
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal as 1D numpy array.
    """
    t = _time_axis(duration_sec, sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def square(frequency: float, duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono square wave."""
    t = _time_axis(duration_sec, sample_rate)
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def sawtooth(frequency: float, duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono sawtooth wave."""
    t = _time_axis(duration_sec, sample_rate)
    # Sawtooth: rises from -1 to +1 over each period
    return amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))


def triangle(frequency: float, duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono triangle wave."""
    t = _time_axis(duration_sec, sample_rate)
    # Triangle from sawtooth: fold the sawtooth
    saw = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    return amplitude * (2 * np.abs(saw) - 1)


def white_noise(duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate mono white noise (uniform distribution)."""
    n_samples = int(sample_rate * duration_sec)
    return amplitude * (2 * np.random.default_rng().random(n_samples) - 1)


def pink_noise(duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate mono pink noise (1/f spectrum) via FFT spectral shaping.

    Generates white noise, transforms to frequency domain, applies 1/sqrt(f)
    roll-off to get a 1/f power spectrum, then transforms back. Produces smooth,
    crackle-free pink noise suitable for ambient textures.
    """
    n_samples = int(sample_rate * duration_sec)
    rng = np.random.default_rng()
    white = rng.standard_normal(n_samples)

    # FFT-based 1/f shaping
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    # Avoid division by zero at DC; leave DC component unchanged
    freqs[0] = 1.0
    fft = fft / np.sqrt(freqs)

    pink = np.fft.irfft(fft, n=n_samples)

    # Normalize to target amplitude
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = amplitude * (pink / peak)
    return pink


def binaural(carrier_hz: float, beat_hz: float, duration_sec: float,
             amplitude: float = DEFAULT_AMPLITUDE,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a stereo binaural beat.

    Left channel: carrier frequency.
    Right channel: carrier + beat frequency.
    The brain perceives the difference as a binaural beat.

    Args:
        carrier_hz: Base frequency for the carrier (e.g., 200 Hz).
        beat_hz: Desired binaural beat frequency (e.g., 10 Hz for alpha).
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Sample rate in Hz.

    Returns:
        Stereo signal as (n_samples, 2) numpy array.
    """
    t = _time_axis(duration_sec, sample_rate)
    left = amplitude * np.sin(2 * np.pi * carrier_hz * t)
    right = amplitude * np.sin(2 * np.pi * (carrier_hz + beat_hz) * t)
    return np.column_stack([left, right])


def load_sample(path: Path, sample_rate: int | None = None) -> np.ndarray:
    """Load an audio file as a numpy array via soundfile.

    Args:
        path: Path to the audio file (WAV, FLAC, OGG, etc.).
        sample_rate: If provided and differs from the file's sample rate,
                     a warning is logged (no resampling in v1).

    Returns:
        Signal as numpy array. Stereo files return (n_samples, 2).
    """
    data, sr = sf.read(str(path), dtype="float64")
    if sample_rate is not None and sr != sample_rate:
        import warnings
        warnings.warn(
            f"Sample rate mismatch: file is {sr}Hz, expected {sample_rate}Hz. "
            f"No resampling applied — consider converting the file first.",
            stacklevel=2,
        )
    return data
