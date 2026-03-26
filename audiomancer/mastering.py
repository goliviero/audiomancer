"""Mastering — final processing chain for release-ready stems.

Highpass rumble removal, sub-bass mono, soft clipping, true peak limiting.
Applied after normalize_lufs as the very last step before export.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audiomancer import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Sub-bass mono (mono everything below crossover)
# ---------------------------------------------------------------------------

def mono_bass(signal: np.ndarray, crossover_hz: float = 100.0,
              order: int = 4,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Force sub-bass frequencies to mono for speaker compatibility.

    Splits signal into low (mono) and high (stereo) via Linkwitz-Riley
    crossover, then recombines. Prevents phase issues on playback systems
    that sum low frequencies (most speakers, all subs).

    Args:
        signal: Stereo input (n, 2). Mono signals pass through unchanged.
        crossover_hz: Crossover frequency (default 100 Hz).
        order: Filter order (4 = 24 dB/oct Linkwitz-Riley).
        sample_rate: Sample rate.

    Returns:
        Signal with mono bass below crossover.
    """
    if signal.ndim == 1:
        return signal

    nyq = sample_rate / 2
    cutoff = np.clip(crossover_hz / nyq, 0.001, 0.99)

    sos_lp = butter(order, cutoff, btype="low", output="sos")
    sos_hp = butter(order, cutoff, btype="high", output="sos")

    # Extract low and high bands per channel
    low_l = sosfiltfilt(sos_lp, signal[:, 0])
    low_r = sosfiltfilt(sos_lp, signal[:, 1])
    high_l = sosfiltfilt(sos_hp, signal[:, 0])
    high_r = sosfiltfilt(sos_hp, signal[:, 1])

    # Mono the lows
    low_mono = (low_l + low_r) * 0.5

    left = low_mono + high_l
    right = low_mono + high_r

    return np.column_stack([left, right])


# ---------------------------------------------------------------------------
# Soft clipper (gentle saturation)
# ---------------------------------------------------------------------------

def soft_clip(signal: np.ndarray, threshold_db: float = -3.0,
              drive: float = 1.0) -> np.ndarray:
    """Soft clipping via tanh saturation.

    Gently rounds peaks instead of hard clipping. Adds subtle warmth
    and controls transients before the limiter.

    Args:
        signal: Input signal.
        threshold_db: Level at which saturation begins (in dB).
        drive: Saturation intensity (1.0 = gentle, 2.0 = warm, 3.0 = hot).

    Returns:
        Soft-clipped signal.
    """
    threshold = 10 ** (threshold_db / 20)
    driven = signal * drive / threshold
    clipped = np.tanh(driven) * threshold
    return clipped


# ---------------------------------------------------------------------------
# True peak limiter
# ---------------------------------------------------------------------------

def limit(signal: np.ndarray, ceiling_dbtp: float = -1.0,
          release_ms: float = 100.0,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Brick-wall limiter via pedalboard.

    Ensures no sample exceeds ceiling. Applied last in the chain
    to guarantee safe levels for streaming/broadcast.

    Args:
        signal: Input signal.
        ceiling_dbtp: Maximum true peak level in dBTP.
        release_ms: Release time in ms.
        sample_rate: Sample rate.

    Returns:
        Limited signal with peak at ceiling_dbtp.
    """
    import pedalboard as pb

    ceiling_linear = 10 ** (ceiling_dbtp / 20)

    board = pb.Pedalboard([
        pb.Limiter(threshold_db=ceiling_dbtp, release_ms=release_ms),
    ])

    was_mono = signal.ndim == 1
    sig = signal.astype(np.float32)

    if was_mono:
        sig = sig[np.newaxis, :]
    else:
        sig = sig.T

    processed = board(sig, sample_rate)

    if was_mono:
        result = processed[0].astype(np.float64)
    else:
        result = processed.T.astype(np.float64)

    # Safety clamp (pedalboard limiter is not always sample-accurate)
    peak = np.max(np.abs(result))
    if peak > ceiling_linear:
        result = result * (ceiling_linear / peak)

    return result


# ---------------------------------------------------------------------------
# Full mastering chain
# ---------------------------------------------------------------------------

def master_chain(signal: np.ndarray,
                 highpass_hz: float = 30.0,
                 crossover_hz: float = 100.0,
                 clip_threshold_db: float = -3.0,
                 ceiling_dbtp: float = -1.0,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Full mastering chain: highpass -> mono bass -> soft clip -> limit.

    Apply after normalize_lufs, before export_wav.

    Args:
        signal: Input signal (mono or stereo).
        highpass_hz: Subsonic rumble removal cutoff.
        crossover_hz: Bass mono crossover frequency.
        clip_threshold_db: Soft clip threshold.
        ceiling_dbtp: Limiter ceiling in dBTP.
        sample_rate: Sample rate.

    Returns:
        Mastered signal ready for export.
    """
    from audiomancer.effects import highpass

    # 1. Remove subsonic rumble
    sig = highpass(signal, cutoff_hz=highpass_hz, sample_rate=sample_rate)

    # 2. Mono the sub-bass
    sig = mono_bass(sig, crossover_hz=crossover_hz, sample_rate=sample_rate)

    # 3. Gentle saturation to tame peaks
    sig = soft_clip(sig, threshold_db=clip_threshold_db)

    # 4. Brick-wall limiting
    sig = limit(sig, ceiling_dbtp=ceiling_dbtp, sample_rate=sample_rate)

    return sig
