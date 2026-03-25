"""Modulation -- LFO, vibrato, filter sweeps, parameter automation.

The missing link between static sounds and living music. These tools
add movement to any signal or parameter over time.

All functions return np.ndarray, consistent with the rest of Fractal.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE
from fractal.generators import _time_axis


# ---------------------------------------------------------------------------
# LFO (Low Frequency Oscillator)
# ---------------------------------------------------------------------------

class LFO:
    """Low Frequency Oscillator for parameter modulation.

    Generates a control signal (not audio) that can modulate any parameter.
    Output range is [0, 1] (unipolar) or [-1, 1] (bipolar).

    Shapes: sine, triangle, square, sawtooth, random, sample_hold.
    """

    _SHAPES = {"sine", "triangle", "square", "sawtooth", "random", "sample_hold"}

    def __init__(
        self,
        rate: float = 1.0,
        shape: str = "sine",
        depth: float = 1.0,
        bipolar: bool = False,
    ):
        """Initialize LFO.

        Args:
            rate: LFO frequency in Hz.
            shape: Waveform shape.
            depth: Modulation depth (0.0 to 1.0). Scales the output amplitude.
            bipolar: If True, output is [-depth, +depth].
                     If False, output is [0, depth].
        """
        if shape not in self._SHAPES:
            raise ValueError(f"Unknown LFO shape: '{shape}'. "
                             f"Available: {sorted(self._SHAPES)}")
        self.rate = rate
        self.shape = shape
        self.depth = np.clip(depth, 0.0, 1.0)
        self.bipolar = bipolar

    def generate(
        self,
        n_samples: int,
        sample_rate: int = SAMPLE_RATE,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate the LFO control signal.

        Args:
            n_samples: Number of samples to generate.
            sample_rate: Sample rate in Hz.
            seed: Random seed (for random and sample_hold shapes).

        Returns:
            1D array of length n_samples.
        """
        t = np.arange(n_samples) / sample_rate
        phase = (t * self.rate) % 1.0

        if self.shape == "sine":
            signal = np.sin(2 * np.pi * phase)
        elif self.shape == "triangle":
            signal = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
        elif self.shape == "square":
            signal = np.where(phase < 0.5, 1.0, -1.0)
        elif self.shape == "sawtooth":
            signal = 2.0 * phase - 1.0
        elif self.shape == "random":
            rng = np.random.default_rng(seed)
            signal = rng.uniform(-1.0, 1.0, n_samples)
        elif self.shape == "sample_hold":
            rng = np.random.default_rng(seed)
            # New random value at each LFO cycle
            cycle_indices = np.floor(t * self.rate).astype(int)
            n_cycles = cycle_indices[-1] + 1 if n_samples > 0 else 0
            values = rng.uniform(-1.0, 1.0, max(n_cycles, 1))
            signal = values[cycle_indices]

        # Apply depth
        signal = signal * self.depth

        # Convert to unipolar if needed
        if not self.bipolar:
            signal = (signal + self.depth) / 2.0

        return signal

    def modulate_param(
        self,
        base_value: float,
        mod_range: float,
        n_samples: int,
        sample_rate: int = SAMPLE_RATE,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a parameter curve modulated by this LFO.

        Convenience method: returns base_value +/- mod_range scaled by the LFO.

        Args:
            base_value: Center value of the parameter.
            mod_range: Maximum deviation from base_value.
            n_samples: Number of samples.
            sample_rate: Sample rate in Hz.
            seed: Random seed.

        Returns:
            1D array of modulated parameter values.
        """
        # Use bipolar LFO internally for modulation
        lfo = self.generate(n_samples, sample_rate, seed)
        if not self.bipolar:
            # Convert unipolar [0, depth] to bipolar [-depth/2, depth/2]
            lfo = 2.0 * lfo - self.depth
        return base_value + mod_range * lfo / self.depth if self.depth > 0 else np.full(n_samples, base_value)


# ---------------------------------------------------------------------------
# Vibrato
# ---------------------------------------------------------------------------

def apply_vibrato(
    signal: np.ndarray,
    rate: float = 5.0,
    depth_cents: float = 20.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply vibrato (pitch modulation) to a signal.

    Uses variable-speed playback via linear interpolation to shift pitch
    up and down periodically.

    Args:
        signal: Input mono signal.
        rate: Vibrato rate in Hz (typically 4-7 Hz).
        depth_cents: Vibrato depth in cents (100 cents = 1 semitone).
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal with vibrato applied, same length as input.
    """
    n = len(signal)
    t = np.arange(n) / sample_rate

    # LFO modulates playback position
    # depth_cents -> fractional speed change
    max_deviation = depth_cents / 1200.0  # in semitones fraction
    lfo = np.sin(2 * np.pi * rate * t)

    # Modulated read positions: cumulative speed variation
    speed = 1.0 + max_deviation * lfo
    read_positions = np.cumsum(speed) - 1.0  # start at 0

    # Clamp to valid range
    read_positions = np.clip(read_positions, 0, n - 1)

    # Linear interpolation
    idx_floor = read_positions.astype(int)
    idx_ceil = np.minimum(idx_floor + 1, n - 1)
    frac = read_positions - idx_floor

    return signal[idx_floor] * (1.0 - frac) + signal[idx_ceil] * frac


# ---------------------------------------------------------------------------
# Filter Sweep
# ---------------------------------------------------------------------------

# Block size for filter processing
_BLOCK_SIZE = 256


def apply_filter_sweep(
    signal: np.ndarray,
    start_hz: float,
    end_hz: float,
    filter_type: str = "lowpass",
    curve: str = "exponential",
    order: int = 2,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply a filter sweep to a signal.

    Smoothly moves the filter cutoff from start_hz to end_hz over the
    duration of the signal. Processed in blocks for efficiency.

    Args:
        signal: Input mono signal.
        start_hz: Starting cutoff frequency in Hz.
        end_hz: Ending cutoff frequency in Hz.
        filter_type: Filter type: "lowpass" or "highpass".
        curve: Sweep curve: "linear" or "exponential".
        order: Filter order (1-4).
        sample_rate: Sample rate in Hz.

    Returns:
        Filtered mono signal.
    """
    if filter_type not in ("lowpass", "highpass"):
        raise ValueError(f"filter_type must be 'lowpass' or 'highpass', got '{filter_type}'")
    if curve not in ("linear", "exponential"):
        raise ValueError(f"curve must be 'linear' or 'exponential', got '{curve}'")

    n = len(signal)
    nyquist = sample_rate / 2
    output = np.zeros(n, dtype=np.float64)

    # Generate cutoff curve
    if curve == "linear":
        cutoffs = np.linspace(start_hz, end_hz, n)
    else:
        # Exponential sweep (perceptually linear in frequency)
        cutoffs = np.geomspace(max(start_hz, 1.0), max(end_hz, 1.0), n)

    # Block-based processing
    for start in range(0, n, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, n)
        block = signal[start:end]

        # Cutoff for this block (midpoint)
        mid = (start + end) // 2
        block_cutoff = cutoffs[mid] if mid < n else cutoffs[-1]
        norm_cutoff = np.clip(block_cutoff / nyquist, 0.001, 0.99)

        sos = butter(order, norm_cutoff, btype=filter_type.replace("pass", ""), output="sos")
        output[start:end] = sosfilt(sos, block)

    return output


# ---------------------------------------------------------------------------
# Parameter Automation (generic)
# ---------------------------------------------------------------------------

def apply_param_automation(
    signal: np.ndarray,
    effect,
    param_name: str,
    automation: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply time-varying parameter automation to an effect.

    Processes the signal in blocks, updating the effect parameter at each
    block boundary. This is the generic version of filter sweeps — it works
    with any Effect that has a settable attribute.

    Args:
        signal: Input signal.
        effect: An Effect instance with a settable attribute matching param_name.
        param_name: Name of the attribute to automate (e.g., "cutoff_hz").
        automation: 1D array of parameter values over time (same length as signal,
            or will be interpolated).
        sample_rate: Sample rate in Hz.

    Returns:
        Processed signal.
    """
    n = len(signal)

    # Resize automation if needed
    if len(automation) != n:
        x_old = np.linspace(0, 1, len(automation))
        x_new = np.linspace(0, 1, n)
        automation = np.interp(x_new, x_old, automation)

    output = np.zeros_like(signal)

    for start in range(0, n, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, n)
        mid = (start + end) // 2

        # Set the parameter for this block
        setattr(effect, param_name, automation[mid])

        # Process the block
        block = signal[start:end]
        output[start:end] = effect.process(block, sample_rate)

    return output
