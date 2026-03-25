"""Synth — basic waveform generators and synthesis.

Generates raw audio signals as numpy arrays. No MIDI, no DAW.
"""

import numpy as np

from audiomancer import SAMPLE_RATE, DEFAULT_AMPLITUDE


# ---------------------------------------------------------------------------
# Time axis helper
# ---------------------------------------------------------------------------

def _time_axis(duration_sec: float, sample_rate: int) -> np.ndarray:
    n = int(sample_rate * duration_sec)
    return np.linspace(0, duration_sec, n, endpoint=False)


# ---------------------------------------------------------------------------
# Basic waveforms
# ---------------------------------------------------------------------------

def sine(frequency: float, duration_sec: float,
         amplitude: float = DEFAULT_AMPLITUDE,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono sine wave."""
    t = _time_axis(duration_sec, sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def square(frequency: float, duration_sec: float,
           amplitude: float = DEFAULT_AMPLITUDE,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono square wave."""
    t = _time_axis(duration_sec, sample_rate)
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def sawtooth(frequency: float, duration_sec: float,
             amplitude: float = DEFAULT_AMPLITUDE,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono sawtooth wave."""
    t = _time_axis(duration_sec, sample_rate)
    return amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))


def triangle(frequency: float, duration_sec: float,
             amplitude: float = DEFAULT_AMPLITUDE,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono triangle wave."""
    t = _time_axis(duration_sec, sample_rate)
    saw = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    return amplitude * (2 * np.abs(saw) - 1)


def white_noise(duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate mono white noise."""
    n = int(sample_rate * duration_sec)
    return amplitude * (2 * np.random.default_rng().random(n) - 1)


def pink_noise(duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate mono pink noise (1/f spectrum) via FFT spectral shaping."""
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng()
    white = rng.standard_normal(n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    freqs[0] = 1.0  # avoid division by zero at DC
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = amplitude * (pink / peak)
    return pink


def brown_noise(duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate mono brown (Brownian) noise via cumulative sum of white noise."""
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng()
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    peak = np.max(np.abs(brown))
    if peak > 0:
        brown = amplitude * (brown / peak)
    return brown


def noise(color: str = "pink", duration_sec: float = 60.0,
          amplitude: float = DEFAULT_AMPLITUDE,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate colored noise.

    Args:
        color: 'white', 'pink', or 'brown'.
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.
    """
    generators = {
        "white": white_noise,
        "pink": pink_noise,
        "brown": brown_noise,
    }
    gen = generators.get(color)
    if gen is None:
        raise ValueError(f"Unknown noise color: {color!r}. Use 'white', 'pink', or 'brown'.")
    return gen(duration_sec, amplitude=amplitude, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Drone / Pad helpers
# ---------------------------------------------------------------------------

def drone(frequency: float, duration_sec: float,
          harmonics: list[tuple[float, float]] | None = None,
          amplitude: float = DEFAULT_AMPLITUDE,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a harmonic drone — fundamental + overtones.

    Args:
        frequency: Fundamental frequency in Hz.
        duration_sec: Duration in seconds.
        harmonics: List of (harmonic_number, relative_amplitude).
            Defaults to first 6 harmonics with 1/n roll-off.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    if harmonics is None:
        harmonics = [(n, 1.0 / n) for n in range(1, 7)]

    t = _time_axis(duration_sec, sample_rate)
    nyquist = sample_rate / 2
    signal = np.zeros_like(t)

    for harmonic_num, rel_amp in harmonics:
        freq = frequency * harmonic_num
        if freq >= nyquist:
            continue
        signal += rel_amp * np.sin(2 * np.pi * freq * t)

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak
    return signal


def pad(frequency: float, duration_sec: float,
        voices: int = 5, detune_cents: float = 12.0,
        amplitude: float = DEFAULT_AMPLITUDE,
        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a detuned unison pad (supersaw-style).

    Args:
        frequency: Fundamental frequency in Hz.
        duration_sec: Duration in seconds.
        voices: Number of detuned voices.
        detune_cents: Total detune spread in cents.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    offsets = np.linspace(-detune_cents / 2, detune_cents / 2, voices)
    t = _time_axis(duration_sec, sample_rate)
    signal = np.zeros_like(t)

    for offset in offsets:
        freq = frequency * 2 ** (offset / 1200)
        signal += sawtooth(freq, duration_sec, amplitude=1.0, sample_rate=sample_rate)

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak
    return signal


def chord_pad(frequencies: list[float], duration_sec: float,
              voices: int = 3, detune_cents: float = 8.0,
              amplitude: float = DEFAULT_AMPLITUDE,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a held chord with detuned voices per note.

    Args:
        frequencies: List of frequencies forming the chord (e.g., [261.63, 329.63, 392.0]).
        duration_sec: Duration in seconds.
        voices: Detuned voices per note.
        detune_cents: Detune spread in cents.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    t = _time_axis(duration_sec, sample_rate)
    signal = np.zeros_like(t)
    offsets = np.linspace(-detune_cents / 2, detune_cents / 2, voices)

    for freq in frequencies:
        for offset in offsets:
            detuned = freq * 2 ** (offset / 1200)
            signal += np.sin(2 * np.pi * detuned * t)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak
    return signal
