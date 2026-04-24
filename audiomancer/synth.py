"""Synth — basic waveform generators and synthesis.

Generates raw audio signals as numpy arrays. No MIDI, no DAW.
"""

import numpy as np

from audiomancer import DEFAULT_AMPLITUDE, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Time axis helper
# ---------------------------------------------------------------------------

def _time_axis(duration_sec: float, sample_rate: int) -> np.ndarray:
    n = int(sample_rate * duration_sec)
    return np.linspace(0, duration_sec, n, endpoint=False)


def _normalize_peak(signal: np.ndarray, amplitude: float) -> np.ndarray:
    """Scale signal so its peak equals *amplitude*."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        return amplitude * signal / peak
    return signal


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
    return _normalize_peak(pink, amplitude)


def brown_noise(duration_sec: float, amplitude: float = DEFAULT_AMPLITUDE,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate mono brown (Brownian) noise via cumulative sum of white noise."""
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng()
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    return _normalize_peak(brown, amplitude)


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

    return _normalize_peak(signal, amplitude)


def _voice_offsets(voices: int, detune_cents: float,
                   jitter_cents: float, seed: int | None) -> np.ndarray:
    """Compute voice detune offsets. Linspace base + optional seeded jitter."""
    base = np.linspace(-detune_cents / 2, detune_cents / 2, voices)
    if jitter_cents > 0 and seed is not None:
        rng = np.random.default_rng(seed)
        base = base + rng.uniform(-jitter_cents, jitter_cents, voices)
    return base


def pad(frequency: float, duration_sec: float,
        voices: int = 5, detune_cents: float = 12.0,
        amplitude: float = DEFAULT_AMPLITUDE,
        seed: int | None = None,
        jitter_cents: float = 0.0,
        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a detuned unison pad (supersaw-style).

    Args:
        frequency: Fundamental frequency in Hz.
        duration_sec: Duration in seconds.
        voices: Number of detuned voices.
        detune_cents: Total detune spread in cents (linspace base).
        amplitude: Peak amplitude.
        seed: Random seed. Required with jitter_cents > 0 for reproducibility.
        jitter_cents: Random +-cents added to each voice offset (breaks rigid
            linspace beating). 0.0 = deterministic comportement (legacy).
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    offsets = _voice_offsets(voices, detune_cents, jitter_cents, seed)
    signal = np.zeros(int(duration_sec * sample_rate))

    for offset in offsets:
        freq = frequency * 2 ** (offset / 1200)
        signal += sawtooth(freq, duration_sec, amplitude=1.0, sample_rate=sample_rate)

    return _normalize_peak(signal, amplitude)


def chord_pad(frequencies: list[float], duration_sec: float,
              voices: int = 3, detune_cents: float = 8.0,
              amplitude: float = DEFAULT_AMPLITUDE,
              seed: int | None = None,
              jitter_cents: float = 0.0,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a held chord with detuned voices per note.

    Args:
        frequencies: List of frequencies forming the chord (e.g., [261.63, 329.63, 392.0]).
        duration_sec: Duration in seconds.
        voices: Detuned voices per note.
        detune_cents: Detune spread in cents (linspace base).
        amplitude: Peak amplitude.
        seed: Random seed. Required with jitter_cents > 0 for reproducibility.
        jitter_cents: Random +-cents added to each voice offset per-note.
            Breaks rigid beating between chord notes. 0.0 = legacy.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    t = _time_axis(duration_sec, sample_rate)
    signal = np.zeros_like(t)

    # Per-note seed derivation: each note gets its own jitter pattern
    # so the chord isn't just "one pad copied at different pitches"
    for i, freq in enumerate(frequencies):
        note_seed = None if seed is None else seed + i * 101
        offsets = _voice_offsets(voices, detune_cents, jitter_cents, note_seed)
        for offset in offsets:
            detuned = freq * 2 ** (offset / 1200)
            signal += np.sin(2 * np.pi * detuned * t)

    return _normalize_peak(signal, amplitude)


# ---------------------------------------------------------------------------
# Karplus-Strong — physical plucked string synthesis
# ---------------------------------------------------------------------------

def karplus_strong(frequency: float, duration_sec: float,
                   decay: float = 0.998, brightness: float = 0.5,
                   amplitude: float = DEFAULT_AMPLITUDE,
                   seed: int | None = None,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Physical plucked string via Karplus-Strong algorithm.

    A short delay line of size N = sample_rate / frequency is initialized
    with white noise, then looped with a lowpass-filtered feedback.
    Naturally produces a plucked-string timbre with exponential decay.

    Args:
        frequency: Fundamental frequency in Hz.
        duration_sec: Total duration.
        decay: Feedback decay factor (0.99-0.999). Lower = shorter pluck.
        brightness: 0.0 = dark (full smoothing), 1.0 = bright (direct).
            Controls the lowpass in the feedback loop.
        amplitude: Peak amplitude of the output.
        seed: Random seed for the initial noise burst.
        sample_rate: Sample rate.

    Returns:
        Mono signal (n_samples,).
    """
    buf_size = max(2, int(sample_rate / max(frequency, 1e-3)))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1.0, 1.0, size=buf_size)

    n = int(duration_sec * sample_rate)
    out = np.zeros(n)

    for i in range(n):
        idx = i % buf_size
        prev = (idx - 1) % buf_size
        out[i] = buf[idx]
        avg = 0.5 * (buf[idx] + buf[prev])
        buf[idx] = (brightness * buf[idx] + (1 - brightness) * avg) * decay

    return _normalize_peak(out, amplitude)


# ---------------------------------------------------------------------------
# Bowed string — simplified Smith/Perry friction model (cello/violin flavor)
# ---------------------------------------------------------------------------

def bowed_string(frequency: float, duration_sec: float,
                 bow_pressure: float = 0.5, bow_velocity: float = 0.7,
                 decay: float = 0.9995, brightness: float = 0.6,
                 amplitude: float = DEFAULT_AMPLITUDE,
                 seed: int | None = None,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Physical bowed-string synthesis (cello/violin flavor).

    Based on a simplified Smith/Perry friction model: a delay line loops
    with a nonlinear friction curve (tanh) driven by bow velocity and
    pressure. Complement to karplus_strong (plucked string).

    Args:
        frequency: Fundamental in Hz.
        duration_sec: Total duration.
        bow_pressure: 0.3 = light (dull), 0.8 = heavy (scratchy).
        bow_velocity: 0.4 = slow bow, 0.9 = fast bow.
        decay: Delay-line feedback factor (very close to 1 for sustained bow).
        brightness: 0 = dark, 1 = bright (feedback lowpass mix).
        amplitude: Peak amplitude.
        seed: Random seed for initial delay-line noise.
        sample_rate: Sample rate.

    Returns:
        Mono signal (n_samples,).
    """
    buf_size = max(2, int(sample_rate / max(frequency, 1e-3)))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-0.05, 0.05, size=buf_size)  # small initial noise

    n = int(duration_sec * sample_rate)
    out = np.zeros(n)
    # Bow pressure scales the friction curve's steepness
    # Higher pressure -> steeper curve -> more harmonics
    friction_steep = 2.0 + 6.0 * bow_pressure

    for i in range(n):
        idx = i % buf_size
        prev = (idx - 1) % buf_size
        # Bow excitation: velocity minus the string's current velocity
        string_vel = buf[idx]
        bow_input = bow_velocity - string_vel
        # Nonlinear friction via tanh (Smith/Perry style)
        friction = np.tanh(bow_input * friction_steep) * 0.3

        out[i] = buf[idx] + friction
        avg = 0.5 * (buf[idx] + buf[prev])
        # Feedback with brightness-controlled smoothing + friction excitation
        buf[idx] = (brightness * buf[idx] + (1 - brightness) * avg) * decay + friction * 0.2

    return _normalize_peak(out, amplitude)


# ---------------------------------------------------------------------------
# Granular synthesis
# ---------------------------------------------------------------------------

def granular(source: np.ndarray, duration_sec: float,
             grain_size_ms: float = 60.0, grain_density: float = 8.0,
             pitch_spread: float = 0.2, position_spread: float = 1.0,
             pitch_curve: np.ndarray | None = None,
             amplitude: float = DEFAULT_AMPLITUDE,
             seed: int | None = None,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a granular cloud from a source buffer.

    Scatters windowed grains at random positions and pitches to create
    evolving, shimmering textures from any input signal.

    Args:
        source: Source audio buffer (mono).
        duration_sec: Output duration in seconds.
        grain_size_ms: Size of each grain in milliseconds.
        grain_density: Average grains triggered per second.
        pitch_spread: Pitch randomisation range in octaves (0=none, 1=+-1 oct).
            Ignored when pitch_curve is provided.
        position_spread: How much of the source buffer to scan (0-1).
        pitch_curve: Optional array of length n_out giving the CENTER pitch
            shift in octaves at each sample. Each grain samples the curve at
            its output position, then adds a small random jitter scaled by
            pitch_spread (if > 0). Lets you ramp a piano sample from 0 to +1
            octave over 30 seconds, for example.
        amplitude: Output peak amplitude.
        seed: Random seed. None = unique each time.
        sample_rate: Sample rate.

    Returns:
        Mono granular cloud signal.
    """
    rng = np.random.default_rng(seed)
    n_out = int(sample_rate * duration_sec)
    grain_samples = int(sample_rate * grain_size_ms / 1000)
    n_grains = int(duration_sec * grain_density)

    # Hann window for smooth grain edges
    window = np.hanning(grain_samples)
    output = np.zeros(n_out, dtype=np.float64)

    src = source if source.ndim == 1 else source[:, 0]
    src_len = len(src)

    for _ in range(n_grains):
        # Random output position
        out_pos = rng.integers(0, max(1, n_out - grain_samples))

        # Random source position
        scan_range = int(src_len * position_spread)
        src_center = rng.integers(0, max(1, src_len))
        src_start = max(0, src_center - scan_range // 2)
        src_start = min(src_start, max(0, src_len - grain_samples))

        # Extract grain from source
        end = min(src_start + grain_samples, src_len)
        grain = src[src_start:end].copy()
        if len(grain) < grain_samples:
            grain = np.pad(grain, (0, grain_samples - len(grain)))

        # Pitch shift via resampling
        # If pitch_curve given: sample curve at this grain's out_pos as center
        # (+ optional small jitter from pitch_spread)
        center_shift = 0.0
        if pitch_curve is not None:
            center_shift = float(pitch_curve[min(out_pos, len(pitch_curve) - 1)])
        jitter = rng.uniform(-pitch_spread, pitch_spread) if pitch_spread > 0 else 0.0
        total_shift = center_shift + jitter
        if total_shift != 0.0:
            ratio = 2 ** total_shift
            indices = np.linspace(0, len(grain) - 1, int(len(grain) / ratio))
            indices = np.clip(indices, 0, len(grain) - 1)
            grain_resampled = np.interp(indices, np.arange(len(grain)), grain)
            # Fit back to grain_samples
            if len(grain_resampled) >= grain_samples:
                grain = grain_resampled[:grain_samples]
            else:
                grain = np.pad(grain_resampled, (0, grain_samples - len(grain_resampled)))

        # Apply window and add to output
        grain = grain * window
        out_end = min(out_pos + grain_samples, n_out)
        length = out_end - out_pos
        output[out_pos:out_end] += grain[:length]

    return _normalize_peak(output, amplitude)
