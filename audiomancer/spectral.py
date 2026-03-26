"""Spectral — FFT-based audio processing for ambient sound design.

Spectral freeze, blur, pitch shift, and gating via STFT/ISTFT.
All functions operate on numpy arrays. Mono = (n,), Stereo = (n, 2).
"""

import numpy as np
from scipy.signal import get_window

from audiomancer import SAMPLE_RATE


# ---------------------------------------------------------------------------
# STFT / ISTFT helpers
# ---------------------------------------------------------------------------

HOP_DIVISOR = 4  # 75% overlap for good reconstruction


def _stft(signal: np.ndarray, fft_size: int = 2048,
          hop: int | None = None) -> np.ndarray:
    """Short-time Fourier transform.

    Pads the signal so no trailing samples are lost.

    Returns:
        Complex array of shape (n_frames, fft_size // 2 + 1).
    """
    if hop is None:
        hop = fft_size // HOP_DIVISOR
    window = get_window("hann", fft_size)
    # Pad to ensure full coverage
    pad_len = fft_size + hop * ((len(signal) - fft_size + hop - 1) // hop) - len(signal)
    if pad_len > 0:
        signal = np.concatenate([signal, np.zeros(pad_len)])
    n_frames = 1 + (len(signal) - fft_size) // hop
    frames = np.zeros((n_frames, fft_size // 2 + 1), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + fft_size] * window
        frames[i] = np.fft.rfft(frame)
    return frames


def _istft(frames: np.ndarray, fft_size: int = 2048,
           hop: int | None = None,
           target_length: int | None = None) -> np.ndarray:
    """Inverse short-time Fourier transform (overlap-add)."""
    if hop is None:
        hop = fft_size // HOP_DIVISOR
    window = get_window("hann", fft_size)
    n_frames = frames.shape[0]
    out_length = fft_size + (n_frames - 1) * hop
    output = np.zeros(out_length, dtype=np.float64)
    norm = np.zeros(out_length, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        reconstructed = np.fft.irfft(frames[i], n=fft_size) * window
        output[start:start + fft_size] += reconstructed
        norm[start:start + fft_size] += window ** 2

    # Normalize by overlap factor (avoid division by zero)
    mask = norm > 1e-8
    output[mask] /= norm[mask]

    if target_length is not None:
        output = output[:target_length]
    return output


def _process_stereo(fn, signal, **kwargs):
    """Apply a mono spectral function to stereo signals."""
    if signal.ndim == 1:
        return fn(signal, **kwargs)
    left = fn(signal[:, 0], **kwargs)
    right = fn(signal[:, 1], **kwargs)
    min_len = min(len(left), len(right))
    return np.column_stack([left[:min_len], right[:min_len]])


# ---------------------------------------------------------------------------
# Spectral freeze
# ---------------------------------------------------------------------------

def freeze(signal: np.ndarray, freeze_time: float = 0.5,
           duration_sec: float | None = None,
           fft_size: int = 4096,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Freeze a moment in time — sustain one spectral frame indefinitely.

    Captures the frequency content at freeze_time and loops it smoothly.
    Creates infinite sustain / pad-like textures from any source.

    Args:
        signal: Input signal (mono or stereo).
        freeze_time: Time in seconds to capture the spectral snapshot.
        duration_sec: Output duration (default: same as input).
        fft_size: FFT window size (larger = smoother).
        sample_rate: Sample rate.

    Returns:
        Frozen signal.
    """
    def _freeze_mono(sig, freeze_time=freeze_time, duration_sec=duration_sec,
                     fft_size=fft_size, sample_rate=sample_rate):
        if duration_sec is None:
            duration_sec_local = len(sig) / sample_rate
        else:
            duration_sec_local = duration_sec
        hop = fft_size // HOP_DIVISOR
        frames = _stft(sig, fft_size, hop)
        # Find the frame closest to freeze_time
        frame_idx = int(freeze_time * sample_rate / hop)
        frame_idx = np.clip(frame_idx, 0, len(frames) - 1)
        frozen_mag = np.abs(frames[frame_idx])

        # Rebuild with frozen magnitude but evolving phase for smoothness
        n_out = int(duration_sec_local * sample_rate)
        n_frames_out = 1 + (n_out - fft_size) // hop
        out_frames = np.zeros((n_frames_out, fft_size // 2 + 1),
                              dtype=np.complex128)
        # Use random-walk phase for natural texture
        rng = np.random.default_rng(42)
        phase = np.angle(frames[frame_idx])
        phase_inc = np.angle(frames[frame_idx])  # base phase increment
        for i in range(n_frames_out):
            out_frames[i] = frozen_mag * np.exp(1j * phase)
            phase += phase_inc + rng.uniform(-0.1, 0.1, len(phase))

        return _istft(out_frames, fft_size, hop, n_out)

    return _process_stereo(_freeze_mono, signal)


# ---------------------------------------------------------------------------
# Spectral blur (smear)
# ---------------------------------------------------------------------------

def blur(signal: np.ndarray, amount: float = 0.5,
         fft_size: int = 2048,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Blur the spectrum across time — smears transients into washes.

    Averages adjacent spectral frames, creating ghostly trailing tones.
    Higher amount = more blur = more ethereal.

    Args:
        signal: Input signal (mono or stereo).
        amount: Blur strength (0.0 = none, 1.0 = maximum).
        fft_size: FFT window size.
        sample_rate: Sample rate.

    Returns:
        Blurred signal.
    """
    def _blur_mono(sig, amount=amount, fft_size=fft_size,
                   sample_rate=sample_rate):
        orig_len = len(sig)
        hop = fft_size // HOP_DIVISOR
        frames = _stft(sig, fft_size, hop)
        # Exponential moving average on magnitudes
        alpha = np.clip(1.0 - amount * 0.95, 0.05, 1.0)
        mags = np.abs(frames)
        phases = np.angle(frames)
        smoothed = mags.copy()
        for i in range(1, len(smoothed)):
            smoothed[i] = alpha * mags[i] + (1 - alpha) * smoothed[i - 1]
        blurred = smoothed * np.exp(1j * phases)
        out = _istft(blurred, fft_size, hop, orig_len)
        # Ensure exact original length
        if len(out) < orig_len:
            out = np.concatenate([out, np.zeros(orig_len - len(out))])
        return out[:orig_len]

    return _process_stereo(_blur_mono, signal)


# ---------------------------------------------------------------------------
# Spectral pitch shift
# ---------------------------------------------------------------------------

def pitch_shift(signal: np.ndarray, semitones: float = 0.0,
                fft_size: int = 4096,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Shift pitch by semitones using spectral bin shifting.

    Simple frequency-domain pitch shift. Best for small shifts (±5 semitones).
    Preserves duration (no time-stretching artifacts).

    Args:
        signal: Input signal (mono or stereo).
        semitones: Pitch shift in semitones (positive = up, negative = down).
        fft_size: FFT window size.
        sample_rate: Sample rate.

    Returns:
        Pitch-shifted signal (same length as input).
    """
    def _shift_mono(sig, semitones=semitones, fft_size=fft_size,
                    sample_rate=sample_rate):
        orig_len = len(sig)
        ratio = 2 ** (semitones / 12.0)
        hop = fft_size // HOP_DIVISOR
        frames = _stft(sig, fft_size, hop)
        n_bins = frames.shape[1]
        shifted = np.zeros_like(frames)

        for i in range(n_bins):
            src_bin = i / ratio
            lo = int(np.floor(src_bin))
            hi = lo + 1
            frac = src_bin - lo
            if 0 <= lo < n_bins:
                shifted[:, i] += (1 - frac) * frames[:, lo]
            if 0 <= hi < n_bins:
                shifted[:, i] += frac * frames[:, hi]

        out = _istft(shifted, fft_size, hop, orig_len)
        if len(out) < orig_len:
            out = np.concatenate([out, np.zeros(orig_len - len(out))])
        return out[:orig_len]

    return _process_stereo(_shift_mono, signal)


# ---------------------------------------------------------------------------
# Spectral gate
# ---------------------------------------------------------------------------

def spectral_gate(signal: np.ndarray, threshold_db: float = -40.0,
                  fft_size: int = 2048,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gate individual frequency bins below a threshold.

    Unlike a time-domain gate, this removes quiet frequencies while
    preserving loud ones — great for cleaning noise from tonal content.

    Args:
        signal: Input signal (mono or stereo).
        threshold_db: Bins below this level (relative to max) are zeroed.
        fft_size: FFT window size.
        sample_rate: Sample rate.

    Returns:
        Gated signal.
    """
    def _gate_mono(sig, threshold_db=threshold_db, fft_size=fft_size,
                   sample_rate=sample_rate):
        orig_len = len(sig)
        hop = fft_size // HOP_DIVISOR
        frames = _stft(sig, fft_size, hop)
        mags = np.abs(frames)
        max_mag = np.max(mags)
        if max_mag == 0:
            return sig
        threshold_linear = max_mag * 10 ** (threshold_db / 20)
        mask = mags >= threshold_linear
        gated = frames * mask
        out = _istft(gated, fft_size, hop, orig_len)
        if len(out) < orig_len:
            out = np.concatenate([out, np.zeros(orig_len - len(out))])
        return out[:orig_len]

    return _process_stereo(_gate_mono, signal)


# ---------------------------------------------------------------------------
# Spectral morph
# ---------------------------------------------------------------------------

def morph(signal_a: np.ndarray, signal_b: np.ndarray,
          mix: float = 0.5, fft_size: int = 2048,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Morph between two signals in the frequency domain.

    Interpolates magnitudes between A and B while preserving A's phase.
    Creates hybrid timbres — e.g., morph a drone into noise.

    Args:
        signal_a: First signal (mono or stereo).
        signal_b: Second signal (must be same length and shape as A).
        mix: Crossfade amount (0.0 = all A, 1.0 = all B).
        fft_size: FFT window size.
        sample_rate: Sample rate.

    Returns:
        Morphed signal.
    """
    def _morph_mono(sig_a, sig_b, mix=mix, fft_size=fft_size):
        min_len = min(len(sig_a), len(sig_b))
        sig_a = sig_a[:min_len]
        sig_b = sig_b[:min_len]
        hop = fft_size // HOP_DIVISOR
        frames_a = _stft(sig_a, fft_size, hop)
        frames_b = _stft(sig_b, fft_size, hop)
        n_frames = min(len(frames_a), len(frames_b))
        mag_a = np.abs(frames_a[:n_frames])
        mag_b = np.abs(frames_b[:n_frames])
        phase_a = np.angle(frames_a[:n_frames])
        morphed_mag = (1 - mix) * mag_a + mix * mag_b
        result = morphed_mag * np.exp(1j * phase_a)
        out = _istft(result, fft_size, hop, min_len)
        if len(out) < min_len:
            out = np.concatenate([out, np.zeros(min_len - len(out))])
        return out[:min_len]

    if signal_a.ndim == 1:
        return _morph_mono(signal_a, signal_b)
    left = _morph_mono(signal_a[:, 0], signal_b[:, 0])
    right = _morph_mono(signal_a[:, 1], signal_b[:, 1])
    min_len = min(len(left), len(right))
    return np.column_stack([left[:min_len], right[:min_len]])
