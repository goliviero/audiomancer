"""Envelopes — amplitude shaping over time.

An envelope is a gain curve (values between 0.0 and 1.0) multiplied onto a signal.
All envelopes follow the same pattern:
  1. generate(n_samples, sample_rate) -> 1D array of gain values
  2. apply(signal, sample_rate) -> shaped signal

Envelopes never mutate the input signal — they always return a new array (DEC-003).
"""

import numpy as np

from fractal.constants import SAMPLE_RATE


class Envelope:
    """Base class for amplitude envelopes."""

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Return a 1D array of gain values [0.0, 1.0]."""
        raise NotImplementedError

    def apply(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Multiply signal by this envelope.

        Works for both mono and stereo signals.
        """
        n_samples = signal.shape[0]
        curve = self.generate(n_samples, sample_rate)

        if signal.ndim == 2:
            # Stereo: broadcast (n_samples,) -> (n_samples, 1) to multiply both channels
            return signal * curve[:, np.newaxis]
        return signal * curve


class FadeInOut(Envelope):
    """Linear fade in at the start and fade out at the end.

    Args:
        fade_in: Fade-in duration in seconds (0 = no fade in).
        fade_out: Fade-out duration in seconds (0 = no fade out).
    """

    def __init__(self, fade_in: float = 0.0, fade_out: float = 0.0):
        self.fade_in = fade_in
        self.fade_out = fade_out

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        curve = np.ones(n_samples)

        fade_in_samples = min(int(sample_rate * self.fade_in), n_samples)
        if fade_in_samples > 0:
            curve[:fade_in_samples] = np.linspace(0.0, 1.0, fade_in_samples)

        fade_out_samples = min(int(sample_rate * self.fade_out), n_samples)
        if fade_out_samples > 0:
            curve[-fade_out_samples:] = np.linspace(1.0, 0.0, fade_out_samples)

        return curve


class ADSR(Envelope):
    """Classic Attack-Decay-Sustain-Release envelope.

    The total duration of the envelope is determined by the signal it's applied to.
    Release phase takes up the last `release` seconds. The remaining time is split
    between attack, decay, and sustain.

    Args:
        attack: Attack time in seconds (ramp from 0 to 1).
        decay: Decay time in seconds (ramp from 1 to sustain level).
        sustain: Sustain level (0.0 to 1.0). Held until release begins.
        release: Release time in seconds (ramp from sustain to 0).
    """

    def __init__(self, attack: float = 0.1, decay: float = 0.1,
                 sustain: float = 0.7, release: float = 0.2):
        self.attack = attack
        self.decay = decay
        self.sustain = max(0.0, min(1.0, sustain))
        self.release = release

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        curve = np.ones(n_samples)

        attack_samples = int(sample_rate * self.attack)
        decay_samples = int(sample_rate * self.decay)
        release_samples = int(sample_rate * self.release)

        # Clamp to available samples
        total_adr = attack_samples + decay_samples + release_samples
        if total_adr > n_samples:
            # Scale proportionally
            scale = n_samples / total_adr
            attack_samples = int(attack_samples * scale)
            decay_samples = int(decay_samples * scale)
            release_samples = n_samples - attack_samples - decay_samples

        sustain_samples = n_samples - attack_samples - decay_samples - release_samples

        idx = 0

        # Attack: 0 -> 1
        if attack_samples > 0:
            curve[idx:idx + attack_samples] = np.linspace(0.0, 1.0, attack_samples)
            idx += attack_samples

        # Decay: 1 -> sustain
        if decay_samples > 0:
            curve[idx:idx + decay_samples] = np.linspace(1.0, self.sustain, decay_samples)
            idx += decay_samples

        # Sustain: hold at sustain level
        if sustain_samples > 0:
            curve[idx:idx + sustain_samples] = self.sustain
            idx += sustain_samples

        # Release: sustain -> 0
        if release_samples > 0:
            curve[idx:idx + release_samples] = np.linspace(self.sustain, 0.0, release_samples)

        return curve


class SmoothFade(Envelope):
    """Cosine-based (S-curve) fade in/out — smoother than linear.

    Uses a raised cosine curve: sounds more natural to the ear because the
    transition accelerates smoothly rather than at constant rate.

    Args:
        fade_in: Fade-in duration in seconds.
        fade_out: Fade-out duration in seconds.
    """

    def __init__(self, fade_in: float = 0.0, fade_out: float = 0.0):
        self.fade_in = fade_in
        self.fade_out = fade_out

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        curve = np.ones(n_samples)

        fade_in_samples = min(int(sample_rate * self.fade_in), n_samples)
        if fade_in_samples > 0:
            # Raised cosine: 0.5 * (1 - cos(pi * t))
            t = np.linspace(0.0, 1.0, fade_in_samples)
            curve[:fade_in_samples] = 0.5 * (1 - np.cos(np.pi * t))

        fade_out_samples = min(int(sample_rate * self.fade_out), n_samples)
        if fade_out_samples > 0:
            t = np.linspace(0.0, 1.0, fade_out_samples)
            curve[-fade_out_samples:] = 0.5 * (1 + np.cos(np.pi * t))

        return curve


class ExponentialFade(Envelope):
    """Exponential fade — fast attack or slow, natural-sounding decay.

    Perceived loudness is logarithmic, so exponential curves often feel
    more linear to the ear than actual linear fades.

    Args:
        fade_in: Fade-in duration in seconds.
        fade_out: Fade-out duration in seconds.
        steepness: Controls curve steepness (higher = steeper). Default 5.0.
    """

    def __init__(self, fade_in: float = 0.0, fade_out: float = 0.0,
                 steepness: float = 5.0):
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.steepness = steepness

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        curve = np.ones(n_samples)
        k = self.steepness

        fade_in_samples = min(int(sample_rate * self.fade_in), n_samples)
        if fade_in_samples > 0:
            t = np.linspace(0.0, 1.0, fade_in_samples)
            # Exponential ramp: (e^(kt) - 1) / (e^k - 1)
            curve[:fade_in_samples] = (np.exp(k * t) - 1) / (np.exp(k) - 1)

        fade_out_samples = min(int(sample_rate * self.fade_out), n_samples)
        if fade_out_samples > 0:
            t = np.linspace(0.0, 1.0, fade_out_samples)
            # Reverse exponential
            curve[-fade_out_samples:] = (np.exp(k * (1 - t)) - 1) / (np.exp(k) - 1)

        return curve


class Swell(Envelope):
    """Swell envelope — gradual rise to peak then hold.

    Useful for ambient pads and drones that build up over time.

    Args:
        rise_time: Time in seconds from 0 to peak.
        peak: Peak gain level (default 1.0).
        curve_type: 'linear', 'cosine', or 'exponential'.
    """

    def __init__(self, rise_time: float = 5.0, peak: float = 1.0,
                 curve_type: str = "cosine"):
        self.rise_time = rise_time
        self.peak = peak
        self.curve_type = curve_type

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        rise_samples = min(int(sample_rate * self.rise_time), n_samples)
        curve = np.full(n_samples, self.peak)

        if rise_samples > 0:
            t = np.linspace(0.0, 1.0, rise_samples)
            if self.curve_type == "linear":
                ramp = t * self.peak
            elif self.curve_type == "cosine":
                ramp = self.peak * 0.5 * (1 - np.cos(np.pi * t))
            elif self.curve_type == "exponential":
                ramp = self.peak * (np.exp(3 * t) - 1) / (np.exp(3) - 1)
            else:
                ramp = t * self.peak
            curve[:rise_samples] = ramp

        return curve


class Gate(Envelope):
    """Gate envelope — on/off pattern for rhythmic effects.

    Creates a repeating pattern of on (1.0) and off (0.0) with optional
    smooth transitions to avoid clicks.

    Args:
        on_time: Duration of the "on" portion in seconds.
        off_time: Duration of the "off" portion in seconds.
        smooth_ms: Transition time in milliseconds to smooth on/off edges.
    """

    def __init__(self, on_time: float = 0.25, off_time: float = 0.25,
                 smooth_ms: float = 5.0):
        self.on_time = on_time
        self.off_time = off_time
        self.smooth_ms = smooth_ms

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        on_samples = int(sample_rate * self.on_time)
        off_samples = int(sample_rate * self.off_time)
        smooth_samples = max(1, int(sample_rate * self.smooth_ms / 1000))
        cycle_samples = on_samples + off_samples

        if cycle_samples == 0:
            return np.ones(n_samples)

        # Build one cycle
        cycle = np.zeros(cycle_samples)
        cycle[:on_samples] = 1.0

        # Smooth transitions (skip if smooth_ms is 0)
        if self.smooth_ms > 0 and smooth_samples > 0 and smooth_samples < on_samples:
            # Fade in at start of on
            cycle[:smooth_samples] = np.linspace(0.0, 1.0, smooth_samples)
            # Fade out at end of on
            cycle[on_samples - smooth_samples:on_samples] = np.linspace(1.0, 0.0, smooth_samples)

        # Tile to fill the signal length
        n_cycles = (n_samples // cycle_samples) + 2
        full = np.tile(cycle, n_cycles)

        return full[:n_samples]


class Tremolo(Envelope):
    """Tremolo — periodic volume modulation via LFO.

    Multiplies the signal by a low-frequency oscillation, creating a
    pulsating volume effect.

    Args:
        rate: LFO frequency in Hz (typical: 2-10 Hz).
        depth: Modulation depth (0.0 = no effect, 1.0 = full on/off).
        shape: LFO waveform: 'sine', 'triangle', or 'square'.
    """

    def __init__(self, rate: float = 5.0, depth: float = 0.5,
                 shape: str = "sine"):
        self.rate = rate
        self.depth = min(1.0, max(0.0, depth))
        self.shape = shape

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)

        if self.shape == "sine":
            lfo = np.sin(2 * np.pi * self.rate * t)
        elif self.shape == "triangle":
            lfo = 2 * np.abs(2 * (t * self.rate - np.floor(0.5 + t * self.rate))) - 1
        elif self.shape == "square":
            lfo = np.sign(np.sin(2 * np.pi * self.rate * t))
        else:
            lfo = np.sin(2 * np.pi * self.rate * t)

        # LFO is [-1, 1]. Map to [1-depth, 1]:
        # gain = 1 - depth * (1 - lfo) / 2 = 1 - depth/2 + depth*lfo/2
        return 1.0 - self.depth * (1.0 - lfo) / 2.0


class AutomationCurve(Envelope):
    """Freeform automation — define breakpoints and interpolate.

    Lets you draw arbitrary gain curves by specifying (time, value) points.
    Linear interpolation between points.

    Args:
        points: List of (time_sec, gain) tuples. Must be sorted by time.
                First point should be at time 0. Last point's time can extend
                to or beyond the signal duration.
    """

    def __init__(self, points: list[tuple[float, float]]):
        if len(points) < 2:
            raise ValueError("AutomationCurve requires at least 2 points")
        self.points = sorted(points, key=lambda p: p[0])

    def generate(self, n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        times = [p[0] for p in self.points]
        values = [p[1] for p in self.points]

        # Create sample-accurate time axis
        t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)

        # Interpolate
        return np.interp(t, times, values)
