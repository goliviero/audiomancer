"""Effects — audio processing transforms.

Each effect follows the same interface:
  effect.process(signal, sample_rate) -> processed signal

Pure numpy/scipy implementations. No pedalboard dependency here (see pedalboard_fx.py).
Inspired by Akasha Portal's enhance_audio.py patterns.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from fractal.constants import SAMPLE_RATE


class Effect:
    """Base class for audio effects.

    Subclasses must implement process(signal, sample_rate) -> np.ndarray.
    Rules: never mutate input, handle both mono and stereo.
    """

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Process a signal and return a new one. Never mutates input."""
        raise NotImplementedError


def _apply_sos_stereo(sos: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Apply an SOS filter to a signal, handling mono and stereo transparently."""
    if signal.ndim == 2:
        return np.column_stack([
            sosfiltfilt(sos, signal[:, 0]),
            sosfiltfilt(sos, signal[:, 1]),
        ])
    return sosfiltfilt(sos, signal)


class LowPassFilter(Effect):
    """Butterworth low-pass filter.

    Attenuates frequencies above the cutoff.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order (steepness). Higher = steeper rolloff.
    """

    def __init__(self, cutoff_hz: float, order: int = 4):
        self.cutoff_hz = cutoff_hz
        self.order = order

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        nyquist = sample_rate / 2
        normalized_cutoff = min(self.cutoff_hz / nyquist, 0.99)
        sos = butter(self.order, normalized_cutoff, btype="low", output="sos")
        return _apply_sos_stereo(sos, signal)


class HighPassFilter(Effect):
    """Butterworth high-pass filter.

    Attenuates frequencies below the cutoff. Useful for removing rumble/DC offset.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order.
    """

    def __init__(self, cutoff_hz: float, order: int = 4):
        self.cutoff_hz = cutoff_hz
        self.order = order

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        nyquist = sample_rate / 2
        normalized_cutoff = max(self.cutoff_hz / nyquist, 0.001)
        sos = butter(self.order, normalized_cutoff, btype="high", output="sos")
        return _apply_sos_stereo(sos, signal)


class BandPassFilter(Effect):
    """Butterworth band-pass filter.

    Passes frequencies between low_hz and high_hz, attenuates the rest.

    Args:
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.
        order: Filter order.
    """

    def __init__(self, low_hz: float, high_hz: float, order: int = 4):
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        nyquist = sample_rate / 2
        low = max(self.low_hz / nyquist, 0.001)
        high = min(self.high_hz / nyquist, 0.99)
        sos = butter(self.order, [low, high], btype="band", output="sos")
        return _apply_sos_stereo(sos, signal)


class EQ(Effect):
    """Parametric EQ — apply multiple filter bands.

    Each band is a dict with keys:
      - type: 'low_shelf', 'high_shelf', or 'peak'
      - freq: center/corner frequency in Hz
      - gain_db: boost (positive) or cut (negative) in dB

    Implementation: for shelf filters, uses a combination of filtered and
    original signal blended by gain. For peak, uses a narrow bandpass.

    Args:
        bands: List of EQ band dicts.
    """

    def __init__(self, bands: list[dict]):
        self.bands = bands

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        result = signal.copy()
        nyquist = sample_rate / 2

        for band in self.bands:
            band_type = band["type"]
            freq = band["freq"]
            gain_db = band["gain_db"]
            gain_linear = 10 ** (gain_db / 20)

            normalized = min(max(freq / nyquist, 0.001), 0.99)

            if band_type == "low_shelf":
                sos = butter(2, normalized, btype="low", output="sos")
                filtered = self._apply_sos(sos, result)
                # Blend: add (gain-1) * filtered portion to original
                result = result + (gain_linear - 1) * filtered

            elif band_type == "high_shelf":
                sos = butter(2, normalized, btype="high", output="sos")
                filtered = self._apply_sos(sos, result)
                result = result + (gain_linear - 1) * filtered

            elif band_type == "peak":
                # Narrow bandpass around center frequency
                bw = 0.5  # octaves
                low_f = freq / (2 ** (bw / 2))
                high_f = freq * (2 ** (bw / 2))
                low_n = max(low_f / nyquist, 0.001)
                high_n = min(high_f / nyquist, 0.99)
                if low_n < high_n:
                    sos = butter(2, [low_n, high_n], btype="band", output="sos")
                    filtered = self._apply_sos(sos, result)
                    result = result + (gain_linear - 1) * filtered

        return result

    @staticmethod
    def _apply_sos(sos: np.ndarray, signal: np.ndarray) -> np.ndarray:
        return _apply_sos_stereo(sos, signal)


class StereoWidth(Effect):
    """Stereo width adjustment using mid/side processing.

    width < 1.0 narrows toward mono.
    width = 1.0 is unchanged.
    width > 1.0 widens the stereo image.

    Args:
        width: Width factor. 0.0 = full mono, 1.0 = original, 2.0 = max wide.
    """

    def __init__(self, width: float = 1.0):
        self.width = max(0.0, width)

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if signal.ndim == 1:
            return signal  # mono signal, nothing to widen

        left = signal[:, 0]
        right = signal[:, 1]

        # Mid/Side encoding
        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        # Apply width to side channel
        side = side * self.width

        # Mid/Side decoding
        new_left = mid + side
        new_right = mid - side

        return np.column_stack([new_left, new_right])


class Reverb(Effect):
    """Simple algorithmic reverb using Schroeder's comb + allpass structure.

    Not studio-quality (use pedalboard for that), but functional for adding
    depth and space to signals.

    Args:
        decay: Reverb tail decay (0.0 to 1.0). Higher = longer tail.
        mix: Wet/dry ratio (0.0 = fully dry, 1.0 = fully wet).
        room_size: Controls delay times. 'small', 'medium', or 'large'.
    """

    # Delay times in ms for comb filters (Schroeder values)
    _COMB_DELAYS = {
        "small":  [23.3, 29.1, 37.1, 41.3],
        "medium": [29.7, 37.1, 41.1, 43.7],
        "large":  [37.3, 43.1, 47.1, 53.3],
    }
    _ALLPASS_DELAYS = [5.0, 1.7]  # ms
    _ALLPASS_COEFFICIENT = 0.7    # standard Schroeder allpass gain

    def __init__(self, decay: float = 0.3, mix: float = 0.2,
                 room_size: str = "medium"):
        self.decay = max(0.0, min(1.0, decay))
        self.mix = max(0.0, min(1.0, mix))
        self.room_size = room_size

    def _comb_filter(self, signal: np.ndarray, delay_ms: float,
                     sample_rate: int) -> np.ndarray:
        delay_samples = int(delay_ms * sample_rate / 1000)
        output = np.zeros(len(signal) + delay_samples)
        output[:len(signal)] = signal
        for i in range(delay_samples, len(output)):
            output[i] += self.decay * output[i - delay_samples]
        return output

    def _allpass_filter(self, signal: np.ndarray, delay_ms: float,
                        sample_rate: int) -> np.ndarray:
        delay_samples = int(delay_ms * sample_rate / 1000)
        output = np.zeros_like(signal)
        g = self._ALLPASS_COEFFICIENT
        for i in range(len(signal)):
            if i >= delay_samples:
                output[i] = -g * signal[i] + signal[i - delay_samples] + g * output[i - delay_samples]
            else:
                output[i] = -g * signal[i]
        return output

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        is_stereo = signal.ndim == 2

        if is_stereo:
            # Process mid channel for reverb
            mono = (signal[:, 0] + signal[:, 1]) * 0.5
        else:
            mono = signal

        # Parallel comb filters (each returns extended buffer with tail)
        delays = self._COMB_DELAYS.get(self.room_size, self._COMB_DELAYS["medium"])
        comb_outputs = [self._comb_filter(mono, d, sample_rate) for d in delays]
        max_comb_len = max(len(c) for c in comb_outputs)
        wet = np.zeros(max_comb_len)
        for c in comb_outputs:
            wet[:len(c)] += c
        wet = wet / len(delays)

        # Series allpass filters
        for delay_ms in self._ALLPASS_DELAYS:
            wet = self._allpass_filter(wet, delay_ms, sample_rate)

        # Truncate wet to original signal length
        wet = wet[:len(mono)]

        # Mix dry + wet
        if is_stereo:
            wet_stereo = np.column_stack([wet, wet])
            return signal * (1 - self.mix) + wet_stereo * self.mix
        return mono * (1 - self.mix) + wet * self.mix


class Delay(Effect):
    """Simple delay / echo effect.

    Args:
        delay_ms: Delay time in milliseconds.
        feedback: Amount of delayed signal fed back (0.0 to 0.95).
        mix: Wet/dry ratio (0.0 = dry, 1.0 = only echoes).
        n_echoes: Maximum number of echoes to render.
    """

    def __init__(self, delay_ms: float = 250.0, feedback: float = 0.4,
                 mix: float = 0.3, n_echoes: int = 8):
        self.delay_ms = delay_ms
        self.feedback = max(0.0, min(0.95, feedback))
        self.mix = max(0.0, min(1.0, mix))
        self.n_echoes = n_echoes

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        delay_samples = int(self.delay_ms * sample_rate / 1000)
        wet = np.zeros_like(signal)
        gain = 1.0

        for i in range(self.n_echoes):
            gain *= self.feedback
            offset = delay_samples * (i + 1)
            if offset >= len(signal):
                break
            if signal.ndim == 2:
                wet[offset:] = wet[offset:] + signal[:len(signal) - offset] * gain
            else:
                wet[offset:] = wet[offset:] + signal[:len(signal) - offset] * gain

        return signal * (1 - self.mix) + wet * self.mix


class Distortion(Effect):
    """Soft clipping distortion via tanh saturation.

    Adds harmonic richness by softly clipping the signal. Low drive values
    give subtle warmth; high values give aggressive fuzz.

    Args:
        drive: Amount of distortion (1.0 = subtle, 5.0 = heavy, 10+ = fuzz).
        mix: Wet/dry ratio.
    """

    def __init__(self, drive: float = 2.0, mix: float = 1.0):
        self.drive = max(0.1, drive)
        self.mix = max(0.0, min(1.0, mix))

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        wet = np.tanh(signal * self.drive) / np.tanh(self.drive)
        return signal * (1 - self.mix) + wet * self.mix


class NormalizePeak(Effect):
    """Peak normalization as an effect (chainable).

    Args:
        target_db: Target peak level in dB.
    """

    def __init__(self, target_db: float = -1.0):
        self.target_db = target_db

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        peak = np.max(np.abs(signal))
        if peak == 0:
            return signal
        target_linear = 10 ** (self.target_db / 20)
        return signal * (target_linear / peak)


class EffectChain(Effect):
    """Chain multiple effects in series.

    Args:
        effects: List of Effect instances, applied in order.
    """

    def __init__(self, effects: list[Effect]):
        self.effects = effects

    def process(self, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        result = signal
        for effect in self.effects:
            result = effect.process(result, sample_rate)
        return result
