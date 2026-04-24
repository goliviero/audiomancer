"""Convolution reverb via impulse responses.

Two modes:
1. Load real CC0 IRs from samples/cc0/ir/ (authentic cathedral/hall/etc.)
2. Generate synthetic IRs on-the-fly (exponential-decay filtered noise)

Runs via scipy.signal.fftconvolve — fast FFT-based convolution.
"""

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from audiomancer import SAMPLE_RATE


def load_ir(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an impulse response WAV.

    Resamples if needed. Returns stereo IR (duplicates mono to L/R).
    """
    from audiomancer.utils import load_audio, mono_to_stereo

    ir, _ = load_audio(Path(path), target_sr=target_sr)
    if ir.ndim == 1:
        ir = mono_to_stereo(ir)
    # Peak-normalize so convolution gain stays predictable
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = ir / peak * 0.95
    return ir


def convolve_reverb(signal: np.ndarray, ir: np.ndarray,
                    wet: float = 0.5,
                    pre_delay_ms: float = 0.0,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply convolution reverb and mix dry+wet.

    Args:
        signal: Input signal (mono or stereo).
        ir: Impulse response (mono or stereo).
        wet: Wet/dry mix (0 = dry, 1 = fully wet).
        pre_delay_ms: Delay before the reverb tail enters (authentic rooms 10-30 ms).
        sample_rate: Sample rate.

    Returns:
        Same-length stereo signal (signal gets converted to stereo if mono).
    """
    from audiomancer.utils import mono_to_stereo

    if signal.ndim == 1:
        signal = mono_to_stereo(signal)
    if ir.ndim == 1:
        ir = mono_to_stereo(ir)

    n = signal.shape[0]
    # Convolve per channel
    wet_l = fftconvolve(signal[:, 0], ir[:, 0])
    wet_r = fftconvolve(signal[:, 1], ir[:, 1])
    wet_stereo = np.column_stack([wet_l[:n], wet_r[:n]])

    # Pre-delay shift (pushes wet signal forward, dry stays aligned)
    pre_samples = int(pre_delay_ms * sample_rate / 1000.0)
    if pre_samples > 0:
        wet_stereo = np.concatenate(
            [np.zeros((pre_samples, 2)), wet_stereo[:n - pre_samples]], axis=0
        )

    # Peak-normalize the wet before mixing to avoid runaway levels
    wet_peak = np.max(np.abs(wet_stereo))
    if wet_peak > 0:
        wet_stereo = wet_stereo / wet_peak * np.max(np.abs(signal))

    return signal * (1 - wet) + wet_stereo * wet


# ---------------------------------------------------------------------------
# Synthetic IRs — no download required, works out of the box
# ---------------------------------------------------------------------------

_PRESETS = {
    # (duration_sec, early_cutoff_hz, late_cutoff_hz, decay_shape, density)
    "room":       (0.6, 4000, 2500, 4.5, 2.0),
    "hall":       (2.5, 5000, 2000, 3.0, 1.5),
    "cathedral":  (6.0, 6000, 1500, 2.0, 1.0),
    "plate":      (1.8, 8000, 4000, 2.8, 2.5),
}


def synthetic_ir(space: str = "hall", seed: int | None = None,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a synthetic stereo IR for a named space.

    Noise-based exponential-decay IR with a two-stage lowpass to
    emulate the progressive HF absorption of real reverbs.

    Args:
        space: 'room' | 'hall' | 'cathedral' | 'plate'.
        seed: Random seed for the noise base.
        sample_rate: Sample rate.

    Returns:
        Stereo IR (n, 2) peak-normalized to 0.95.
    """
    if space not in _PRESETS:
        raise ValueError(
            f"Unknown space {space!r}. Valid: {list(_PRESETS)}"
        )
    duration_sec, early_cut, late_cut, decay_shape, density = _PRESETS[space]

    n = int(duration_sec * sample_rate)
    rng = np.random.default_rng(seed)

    # Dense noise base with slight L/R decorrelation
    noise_l = rng.standard_normal(n)
    noise_r = rng.standard_normal(n)

    # Exponential decay: exp(-t * shape / duration)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    decay = np.exp(-t * decay_shape / duration_sec)

    # Early-to-late frequency sweep: HF fades faster than LF
    from scipy.signal import butter, sosfiltfilt

    # First pass: lowpass at `early_cut` (full spectrum at t=0)
    sos_early = butter(2, early_cut / (sample_rate / 2), btype="low", output="sos")
    lp_l = sosfiltfilt(sos_early, noise_l * decay)
    lp_r = sosfiltfilt(sos_early, noise_r * decay)

    # Second pass: progressive LP toward `late_cut` (time-varying via mix)
    late_decay = np.exp(-t * decay_shape * 0.6 / duration_sec)
    sos_late = butter(2, late_cut / (sample_rate / 2), btype="low", output="sos")
    lp2_l = sosfiltfilt(sos_late, noise_l * late_decay)
    lp2_r = sosfiltfilt(sos_late, noise_r * late_decay)

    # Blend: early portion brighter, late portion duller
    blend = np.linspace(1.0, 0.0, n)
    ir_l = lp_l * blend + lp2_l * (1 - blend)
    ir_r = lp_r * blend + lp2_r * (1 - blend)

    # Density scaling (subtle tilt for denser/sparser feel)
    ir_l *= density
    ir_r *= density

    ir = np.column_stack([ir_l, ir_r])
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = ir / peak * 0.95
    return ir


def reverb_from_synthetic(signal: np.ndarray, space: str = "hall",
                          wet: float = 0.45, pre_delay_ms: float = 15.0,
                          seed: int | None = None,
                          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convenience: generate a synthetic IR and apply it in one call."""
    ir = synthetic_ir(space, seed=seed, sample_rate=sample_rate)
    return convolve_reverb(signal, ir, wet=wet, pre_delay_ms=pre_delay_ms,
                           sample_rate=sample_rate)
