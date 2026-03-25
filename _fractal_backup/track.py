"""Track — container for a named signal with volume, pan, effects, and bus routing.

A Track holds a signal and its mix parameters. Calling render() applies
effects, gain, and panning to produce a stereo output ready for the mixer.

Equal-power panning is used by default to avoid the "center dip" of linear panning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.effects import Effect
from fractal.signal import is_mono, is_stereo, mono_to_stereo


def apply_pan(signal: np.ndarray, pan: float,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply stereo panning to a signal using equal-power law.

    Args:
        signal: Mono (n,) or stereo (n, 2) signal.
        pan: -1.0 = full left, 0.0 = center, +1.0 = full right.
        sample_rate: Not used directly, kept for API consistency.

    Returns:
        Stereo (n, 2) signal with panning applied.
    """
    pan = np.clip(pan, -1.0, 1.0)

    if is_mono(signal):
        signal = mono_to_stereo(signal)

    # Equal-power: angle from 0 (left) to pi/2 (right)
    # pan=-1 -> angle=0 -> left=1, right=0
    # pan=0  -> angle=pi/4 -> left=right=0.707
    # pan=+1 -> angle=pi/2 -> left=0, right=1
    angle = (pan + 1.0) / 2.0 * (np.pi / 2.0)
    gain_left = np.cos(angle)
    gain_right = np.sin(angle)

    result = np.empty_like(signal)
    result[:, 0] = signal[:, 0] * gain_left
    result[:, 1] = signal[:, 1] * gain_right
    return result


@dataclass
class Track:
    """A named audio track with mix parameters.

    Attributes:
        name: Track identifier.
        signal: Raw audio signal (mono or stereo).
        volume_db: Gain in dB (0.0 = unity).
        pan: Stereo position (-1 left, 0 center, +1 right).
        mute: If True, render() returns silence.
        solo: Solo flag — handled by Session, not Track itself.
        effects: Effect chain applied before gain/pan.
        bus: Name of the bus this track routes to.
    """
    name: str
    signal: np.ndarray
    volume_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    effects: list[Effect] = field(default_factory=list)
    bus: str = "master"

    def render(self, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Render the track: effects -> gain -> pan -> stereo output.

        Returns:
            Stereo (n, 2) numpy array.
        """
        if self.mute:
            n = self.signal.shape[0]
            return np.zeros((n, 2), dtype=np.float64)

        sig = self.signal.copy()

        # Apply effects chain
        for fx in self.effects:
            sig = fx.process(sig, sample_rate)

        # Apply volume
        gain = 10 ** (self.volume_db / 20)
        sig = sig * gain

        # Apply panning (returns stereo)
        sig = apply_pan(sig, self.pan, sample_rate)

        return sig
