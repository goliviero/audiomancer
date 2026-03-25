"""Synthesizers -- advanced signal generation beyond basic waveforms.

FM synthesis, additive synthesis, wavetable oscillators, subtractive synth,
pulse waves, and unison/detune stacking.

All functions return np.ndarray (mono by default), consistent with generators.py.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE
from fractal.generators import sine, square, sawtooth, triangle, _time_axis


# ---------------------------------------------------------------------------
# FM Synthesis
# ---------------------------------------------------------------------------

def fm_synth(
    carrier_hz: float,
    modulator_hz: float,
    mod_index: float,
    duration_sec: float,
    amplitude: float = DEFAULT_AMPLITUDE,
    mod_envelope: np.ndarray | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """FM synthesis via phase modulation.

    The modulator modulates the phase of the carrier, producing rich harmonic
    spectra from just two oscillators. Classic technique behind the Yamaha DX7.

    Args:
        carrier_hz: Carrier frequency in Hz.
        modulator_hz: Modulator frequency in Hz.
        mod_index: Modulation index (depth). 0 = pure sine, 3+ = metallic.
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude.
        mod_envelope: Optional 1D array (same length as output) that scales
            the modulation index over time. Values 0-1. Use an Envelope's
            .generate() output for time-varying brightness.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    t = _time_axis(duration_sec, sample_rate)
    n = len(t)

    # Modulator signal
    modulator = np.sin(2 * np.pi * modulator_hz * t)

    # Apply time-varying modulation depth
    if mod_envelope is not None:
        env = mod_envelope[:n] if len(mod_envelope) >= n else np.pad(
            mod_envelope, (0, n - len(mod_envelope)), constant_values=0.0,
        )
        modulator = modulator * env

    # Phase modulation: carrier phase offset by modulator * index
    return amplitude * np.sin(2 * np.pi * carrier_hz * t + mod_index * modulator)


# ---------------------------------------------------------------------------
# Additive Synthesis
# ---------------------------------------------------------------------------

# Preset harmonic recipes: (harmonic_number, relative_amplitude)
HARMONIC_PRESETS: dict[str, list[tuple[int, float]]] = {
    "organ":   [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (6, 0.1), (8, 0.05)],
    "string":  [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4), (5, 0.3), (6, 0.2), (7, 0.15)],
    "reed":    [(1, 1.0), (2, 0.3), (3, 0.7), (4, 0.2), (5, 0.5), (6, 0.1)],
    "bell":    [(1, 1.0), (2.76, 0.6), (4.07, 0.4), (5.93, 0.25), (8.2, 0.15)],
    "brass":   [(1, 1.0), (2, 0.9), (3, 0.7), (4, 0.5), (5, 0.35), (6, 0.2)],
    "flute":   [(1, 1.0), (2, 0.1), (3, 0.05)],
    "vowel_a": [(1, 1.0), (2, 0.6), (3, 0.3), (4, 0.2), (5, 0.15)],
    "vowel_o": [(1, 1.0), (2, 0.4), (3, 0.15), (4, 0.1)],
}


def additive(
    fundamental_hz: float,
    harmonics: list[tuple[float, float]],
    duration_sec: float,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Additive synthesis -- build a timbre from a harmonic series.

    Each harmonic is defined as (harmonic_number, relative_amplitude).
    Harmonic numbers can be non-integer for inharmonic timbres (bells, etc.).

    Args:
        fundamental_hz: Fundamental frequency in Hz.
        harmonics: List of (harmonic_number, relative_amplitude) tuples.
            harmonic_number 1 = fundamental, 2 = octave, etc.
        duration_sec: Duration in seconds.
        amplitude: Overall peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    t = _time_axis(duration_sec, sample_rate)
    signal = np.zeros(len(t), dtype=np.float64)

    for harmonic_num, rel_amp in harmonics:
        freq = fundamental_hz * harmonic_num
        # Skip harmonics above Nyquist
        if freq >= sample_rate / 2:
            continue
        signal += rel_amp * np.sin(2 * np.pi * freq * t)

    # Normalize so peak matches amplitude
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Wavetable Synthesis
# ---------------------------------------------------------------------------

def wavetable(
    table: np.ndarray,
    frequency: float,
    duration_sec: float,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Wavetable synthesis -- read a single-cycle waveform at a given frequency.

    The table is a 1D array representing one cycle. The oscillator reads
    through it at the rate determined by frequency, using linear interpolation.

    Args:
        table: Single-cycle waveform (any length). Values should be in [-1, 1].
        frequency: Playback frequency in Hz.
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n_samples = int(sample_rate * duration_sec)
    table_len = len(table)

    # Phase accumulator: how fast we move through the table
    phase_increment = frequency * table_len / sample_rate
    phases = np.arange(n_samples) * phase_increment

    # Wrap to table length
    indices = phases % table_len

    # Linear interpolation between adjacent table entries
    idx_floor = indices.astype(int)
    idx_ceil = (idx_floor + 1) % table_len
    frac = indices - idx_floor

    signal = table[idx_floor] * (1.0 - frac) + table[idx_ceil] * frac

    return amplitude * signal


# ---------------------------------------------------------------------------
# Subtractive Synthesis
# ---------------------------------------------------------------------------

_OSCILLATOR_MAP = {
    "sine": sine,
    "square": square,
    "saw": sawtooth,
    "sawtooth": sawtooth,
    "triangle": triangle,
}

# Block size for filter envelope processing (samples)
_BLOCK_SIZE = 256


def subtractive(
    oscillator: str,
    frequency: float,
    duration_sec: float,
    cutoff_hz: float,
    resonance: float = 0.0,
    envelope: "object | None" = None,
    filter_envelope: np.ndarray | None = None,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Subtractive synthesis -- oscillator through a resonant low-pass filter.

    The classic analog synth signal flow: generate a harmonically rich
    waveform, then sculpt it by filtering out frequencies.

    Args:
        oscillator: Waveform type: "saw", "square", "triangle", "sine".
        frequency: Oscillator frequency in Hz.
        duration_sec: Duration in seconds.
        cutoff_hz: Filter cutoff frequency in Hz.
        resonance: Filter resonance (0.0 to 0.95). Higher = sharper peak.
        envelope: Optional amplitude Envelope. Applied after filtering.
        filter_envelope: Optional 1D array (0-1) that scales cutoff_hz over
            time. At 0.0, cutoff = 20Hz (nearly closed). At 1.0, cutoff =
            cutoff_hz. Use an Envelope's .generate() for filter sweeps.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    if oscillator not in _OSCILLATOR_MAP:
        raise ValueError(f"Unknown oscillator: '{oscillator}'. "
                         f"Available: {list(_OSCILLATOR_MAP.keys())}")

    # Generate raw oscillator
    gen_fn = _OSCILLATOR_MAP[oscillator]
    raw = gen_fn(frequency, duration_sec, amplitude=1.0, sample_rate=sample_rate)
    n = len(raw)

    # Apply filter
    if filter_envelope is not None:
        # Block-based processing with time-varying cutoff
        env = filter_envelope[:n] if len(filter_envelope) >= n else np.pad(
            filter_envelope, (0, n - len(filter_envelope)), constant_values=0.0,
        )
        output = np.zeros(n, dtype=np.float64)
        nyquist = sample_rate / 2

        for start in range(0, n, _BLOCK_SIZE):
            end = min(start + _BLOCK_SIZE, n)
            block = raw[start:end]

            # Interpolate cutoff for this block
            mid = (start + end) // 2
            env_val = env[mid] if mid < n else env[-1]
            block_cutoff = max(20.0, cutoff_hz * env_val)
            norm_cutoff = min(block_cutoff / nyquist, 0.99)

            # Design filter for this block
            order = 2 if resonance < 0.5 else 4
            sos = butter(order, norm_cutoff, btype="low", output="sos")

            # Apply resonance boost by scaling feedback
            if resonance > 0 and len(block) > order * 2:
                output[start:end] = sosfilt(sos, block)
            else:
                output[start:end] = sosfilt(sos, block)
    else:
        # Static filter
        nyquist = sample_rate / 2
        norm_cutoff = min(cutoff_hz / nyquist, 0.99)
        norm_cutoff = max(norm_cutoff, 0.001)
        order = 2 if resonance < 0.5 else 4
        sos = butter(order, norm_cutoff, btype="low", output="sos")
        output = sosfilt(sos, raw)

    # Apply amplitude envelope
    if envelope is not None:
        env_curve = envelope.generate(n, sample_rate)
        output = output * env_curve

    # Normalize to target amplitude
    peak = np.max(np.abs(output))
    if peak > 0:
        output = amplitude * output / peak

    return output


# ---------------------------------------------------------------------------
# Pulse Wave
# ---------------------------------------------------------------------------

def pulse(
    frequency: float,
    duration_sec: float,
    duty: float = 0.5,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Pulse wave with adjustable duty cycle.

    At duty=0.5, identical to a square wave. Lower duty = thinner, nasal sound.
    Higher duty = wider, hollower sound.

    Args:
        frequency: Frequency in Hz.
        duration_sec: Duration in seconds.
        duty: Duty cycle (0.0 to 1.0). 0.5 = square wave.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    t = _time_axis(duration_sec, sample_rate)
    duty = np.clip(duty, 0.01, 0.99)

    # Phase position within each cycle (0 to 1)
    phase = (t * frequency) % 1.0

    # High when phase < duty, low otherwise
    signal = np.where(phase < duty, 1.0, -1.0)

    return amplitude * signal


# ---------------------------------------------------------------------------
# Unison / Detune
# ---------------------------------------------------------------------------

def unison(
    generator_fn,
    frequency: float,
    duration_sec: float,
    voices: int = 3,
    detune_cents: float = 10.0,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    **kwargs,
) -> np.ndarray:
    """Layer N detuned copies of a generator for thickness.

    Classic technique for supersaws, wide pads, and rich leads. Each voice
    is slightly detuned from the center frequency.

    Args:
        generator_fn: Any generator function with signature
            (frequency, duration_sec, amplitude, sample_rate) -> ndarray.
        frequency: Center frequency in Hz.
        duration_sec: Duration in seconds.
        voices: Number of voices (1 = no detune).
        detune_cents: Total detune spread in cents (100 cents = 1 semitone).
            Spread symmetrically around center frequency.
        amplitude: Peak amplitude of the final mixed signal.
        sample_rate: Sample rate in Hz.
        **kwargs: Extra arguments passed to generator_fn.

    Returns:
        Mono signal.
    """
    if voices < 1:
        raise ValueError("voices must be >= 1")
    if voices == 1:
        return generator_fn(frequency, duration_sec, amplitude=amplitude,
                            sample_rate=sample_rate, **kwargs)

    # Spread voices symmetrically in cents
    offsets = np.linspace(-detune_cents / 2, detune_cents / 2, voices)

    # Generate each voice
    result = np.zeros(int(sample_rate * duration_sec), dtype=np.float64)
    for offset_cents in offsets:
        freq = frequency * 2 ** (offset_cents / 1200)
        voice = generator_fn(freq, duration_sec, amplitude=1.0,
                             sample_rate=sample_rate, **kwargs)
        result = result + voice

    # Normalize to target amplitude
    peak = np.max(np.abs(result))
    if peak > 0:
        result = amplitude * result / peak

    return result
