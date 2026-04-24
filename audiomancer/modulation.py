"""Modulation — LFOs, random drift, and slow evolution for ambient textures.

Makes static synthesis come alive with subtle, slow-moving changes.
All modulation is designed for hours-long listening — gentle, never jarring.
"""

import numpy as np

from audiomancer import SAMPLE_RATE

# ---------------------------------------------------------------------------
# LFO (Low Frequency Oscillator)
# ---------------------------------------------------------------------------

def lfo_sine(duration_sec: float, rate_hz: float = 0.05,
             depth: float = 1.0, offset: float = 0.0,
             phase: float = 0.0,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine LFO for parameter modulation.

    Args:
        duration_sec: Duration in seconds.
        rate_hz: LFO frequency (default 0.05 Hz = 20s cycle).
        depth: Amplitude of modulation (0.0 to 1.0).
        offset: Center value (output = offset ± depth).
        phase: Initial phase in radians.
        sample_rate: Sample rate.

    Returns:
        Modulation signal (n_samples,). Values in [offset-depth, offset+depth].
    """
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    return offset + depth * np.sin(2 * np.pi * rate_hz * t + phase)


def lfo_triangle(duration_sec: float, rate_hz: float = 0.05,
                 depth: float = 1.0, offset: float = 0.0,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a triangle LFO — smoother transitions than sine at extremes."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    saw = 2 * (t * rate_hz - np.floor(0.5 + t * rate_hz))
    tri = 2 * np.abs(saw) - 1
    return offset + depth * tri


# ---------------------------------------------------------------------------
# Random drift (Brownian motion on parameters)
# ---------------------------------------------------------------------------

def drift(duration_sec: float, speed: float = 0.1,
          depth: float = 1.0, offset: float = 0.0,
          seed: int | None = None,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate slow random drift via smoothed Brownian motion.

    Creates organic, non-repeating parameter movement.
    The drift is heavily smoothed to avoid sudden jumps.

    Args:
        duration_sec: Duration in seconds.
        speed: How fast the drift moves (0.01=glacial, 1.0=fast).
        depth: Max deviation from center.
        offset: Center value.
        seed: Random seed for reproducibility.
        sample_rate: Sample rate.

    Returns:
        Modulation signal (n_samples,). Values clamped to [offset-depth, offset+depth].
    """
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng(seed)

    # Generate white noise, then heavily smooth it
    raw = rng.standard_normal(n) * speed

    # Cumulative sum = Brownian motion
    brown = np.cumsum(raw)

    # Smooth with a 1-second uniform filter. uniform_filter1d is O(N) regardless
    # of window size — np.convolve was O(N*M) and became catastrophically slow
    # at high SR (48kHz * long duration).
    from scipy.ndimage import uniform_filter1d
    window = max(sample_rate, 1)
    smoothed = uniform_filter1d(brown, size=window, mode="nearest")

    # Normalize to [-1, 1] range then scale
    peak = np.max(np.abs(smoothed))
    if peak > 0:
        smoothed = smoothed / peak

    return np.clip(offset + depth * smoothed, offset - depth, offset + depth)


# ---------------------------------------------------------------------------
# Compound modulations
# ---------------------------------------------------------------------------

def evolving_lfo(duration_sec: float, rate_hz: float = 0.05,
                 depth: float = 1.0, offset: float = 0.0,
                 drift_speed: float = 0.05,
                 seed: int | None = None,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """LFO with slowly drifting rate and depth — never exactly repeats.

    Combines a sine LFO with random drift on both rate and depth
    for organic, evolving modulation.

    Args:
        duration_sec: Duration.
        rate_hz: Base LFO rate.
        depth: Base modulation depth.
        offset: Center value.
        drift_speed: How much the LFO parameters drift.
        seed: Random seed.
        sample_rate: Sample rate.

    Returns:
        Modulation signal (n_samples,).
    """
    # Drift the LFO rate slightly
    rate_mod = drift(duration_sec, speed=drift_speed * 0.5,
                     depth=rate_hz * 0.3, offset=rate_hz,
                     seed=seed, sample_rate=sample_rate)

    # Drift the depth slightly
    seed2 = (seed + 1) if seed is not None else None
    depth_mod = drift(duration_sec, speed=drift_speed * 0.3,
                      depth=depth * 0.2, offset=depth,
                      seed=seed2, sample_rate=sample_rate)

    # Build the LFO with varying rate (integrate instantaneous frequency)
    phase = np.cumsum(2 * np.pi * rate_mod / sample_rate)
    signal = offset + depth_mod * np.sin(phase)

    return np.clip(signal, offset - depth * 1.5, offset + depth * 1.5)


# ---------------------------------------------------------------------------
# Apply modulation to audio signals
# ---------------------------------------------------------------------------

def apply_amplitude_mod(signal: np.ndarray, mod: np.ndarray) -> np.ndarray:
    """Apply amplitude modulation to a signal.

    Args:
        signal: Audio signal (mono or stereo).
        mod: Modulation signal (values around 1.0 = no change).
             If shorter than signal, the last value is held.

    Returns:
        Modulated signal.
    """
    n = signal.shape[0]
    if len(mod) < n:
        # Pad mod by holding last value
        padded = np.empty(n)
        padded[:len(mod)] = mod
        padded[len(mod):] = mod[-1]
        mod = padded
    else:
        mod = mod[:n]

    if signal.ndim == 2:
        return signal * mod[:, np.newaxis]
    return signal * mod


def apply_filter_sweep(signal: np.ndarray, mod: np.ndarray,
                       sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a time-varying lowpass filter sweep using block processing.

    Divides the signal into short blocks and applies a different
    cutoff frequency per block, controlled by the modulation signal.
    Filter state (zi) is carried across blocks to prevent clicks at
    block boundaries.

    Args:
        signal: Audio signal.
        mod: Modulation signal where values are cutoff frequencies in Hz.
        sample_rate: Sample rate.

    Returns:
        Filtered signal with sweeping cutoff.
    """
    from scipy.signal import butter, sosfilt, sosfilt_zi

    block_size = sample_rate // 20  # 50ms blocks — smoother cutoff transitions
    n_blocks = (len(signal) + block_size - 1) // block_size
    result = np.zeros_like(signal)
    nyquist = sample_rate / 2

    zi_l = None  # filter state, left/mono channel
    zi_r = None  # filter state, right channel

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, len(signal))
        mid = (start + end) // 2

        # Get cutoff from modulation at block midpoint
        mod_idx = min(mid, len(mod) - 1)
        cutoff_hz = float(np.clip(mod[mod_idx], 200, nyquist - 100))
        cutoff_norm = cutoff_hz / nyquist

        sos = butter(2, cutoff_norm, btype="low", output="sos")
        block = signal[start:end]

        if block.ndim == 2:
            # Stereo: carry state per channel
            if zi_l is None:
                zi_l = sosfilt_zi(sos) * block[0, 0]
                zi_r = sosfilt_zi(sos) * block[0, 1]
            out_l, zi_l = sosfilt(sos, block[:, 0], zi=zi_l)
            out_r, zi_r = sosfilt(sos, block[:, 1], zi=zi_r)
            result[start:end, 0] = out_l
            result[start:end, 1] = out_r
        else:
            # Mono
            if zi_l is None:
                zi_l = sosfilt_zi(sos) * block[0]
            out, zi_l = sosfilt(sos, block, zi=zi_l)
            result[start:end] = out

    return result


# ---------------------------------------------------------------------------
# Phase D — multi-timescale LFO + Ornstein-Uhlenbeck random walk
# ---------------------------------------------------------------------------

def multi_lfo(duration_sec: float,
              layers: list[tuple[float, float]] | None = None,
              seed: int | None = None,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Superpose N sine LFOs at different scales with random phases.

    Purpose: avoid the single-breathing "tell" on long loops. A 45s LFO gets
    recognized by the 7th iteration; stacking 3 non-synced LFOs at different
    scales makes the envelope un-predictable while staying organic.

    Args:
        duration_sec: Duration.
        layers: List of (rate_hz, depth) tuples. Default = 3 layers:
            fast micro-tremolo ~10s, medium breathing ~45s, slow drift ~3min.
        seed: Random seed for per-layer phase offsets.
        sample_rate: Sample rate.

    Returns:
        Envelope centered on 1.0, shape (n_samples,).
    """
    if layers is None:
        layers = [(1 / 10.0, 0.03), (1 / 45.0, 0.10), (1 / 180.0, 0.05)]

    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    rng = np.random.default_rng(seed)

    env = np.ones(n)
    for rate_hz, depth in layers:
        phase = rng.uniform(0, 2 * np.pi)
        env = env + depth * np.sin(2 * np.pi * rate_hz * t + phase)

    # Keep envelope bounded around 1 even if all layers align
    max_depth = sum(d for _, d in layers)
    return np.clip(env, 1 - max_depth, 1 + max_depth)


def random_walk(duration_sec: float, sigma: float = 0.05, tau: float = 10.0,
                seed: int | None = None,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Ornstein-Uhlenbeck mean-reverting random walk (bounded, non-periodic).

    Unlike drift() which is pure Brownian (unbounded), this process is pulled
    back toward 1.0 with time constant `tau`. Suitable for "organic drift" on
    parameters (volume, filter cutoff) where you want unpredictability without
    runaway accumulation on 3h renders.

    dx = -(x - 1) * dt/tau + sigma * sqrt(dt) * N(0,1)

    Args:
        duration_sec: Duration.
        sigma: Noise amplitude (stationary std is ~ sigma * sqrt(tau/2)).
        tau: Mean-reversion time constant in seconds.
        seed: Random seed.
        sample_rate: Sample rate.

    Returns:
        Walk centered on 1.0, shape (n_samples,). Stays bounded in practice
        within 1 +- 3 * sigma * sqrt(tau/2) at 99.7% of time.
    """
    from scipy.signal import lfilter

    n = int(sample_rate * duration_sec)
    dt = 1.0 / sample_rate
    rng = np.random.default_rng(seed)

    # Exact OU discretization: x[i] = alpha * x[i-1] + (1-alpha) * mean + noise
    alpha = np.exp(-dt / tau)
    noise_std = sigma * np.sqrt((1 - alpha**2) * tau / 2)

    noise = rng.standard_normal(n) * noise_std
    # Add mean drift (1.0) into the noise term: since target mean is 1,
    # we shift after the AR(1) filter. AR(1): y[n] = alpha * y[n-1] + noise[n]
    # solves the deviation from the mean; final signal = 1 + deviation.
    deviation = lfilter([1.0], [1.0, -alpha], noise)
    return 1.0 + deviation
