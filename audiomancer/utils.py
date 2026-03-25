"""Utilities — I/O, normalization, fade, signal helpers.

All functions operate on numpy arrays. Mono = (n,), Stereo = (n, 2).
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from audiomancer import SAMPLE_RATE, DEFAULT_AMPLITUDE


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def silence(duration_sec: float, stereo: bool = False,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a silent signal."""
    n = int(sample_rate * duration_sec)
    if stereo:
        return np.zeros((n, 2), dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def normalize(signal: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize signal peak to target_db."""
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    target_linear = 10 ** (target_db / 20)
    return signal * (target_linear / peak)


def fade_in(signal: np.ndarray, duration_sec: float,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a linear fade-in."""
    n_fade = min(int(sample_rate * duration_sec), len(signal))
    out = signal.copy()
    ramp = np.linspace(0.0, 1.0, n_fade)
    if out.ndim == 2:
        out[:n_fade] *= ramp[:, np.newaxis]
    else:
        out[:n_fade] *= ramp
    return out


def fade_out(signal: np.ndarray, duration_sec: float,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a linear fade-out."""
    n_fade = min(int(sample_rate * duration_sec), len(signal))
    out = signal.copy()
    ramp = np.linspace(1.0, 0.0, n_fade)
    if out.ndim == 2:
        out[-n_fade:] *= ramp[:, np.newaxis]
    else:
        out[-n_fade:] *= ramp
    return out


def concat(*signals: np.ndarray) -> np.ndarray:
    """Concatenate signals end-to-end."""
    return np.concatenate(signals)


def pad_to_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad signal with zeros to reach target length."""
    current = signal.shape[0]
    if current >= target_samples:
        return signal
    pad_amount = target_samples - current
    if signal.ndim == 2:
        padding = np.zeros((pad_amount, 2), dtype=signal.dtype)
    else:
        padding = np.zeros(pad_amount, dtype=signal.dtype)
    return np.concatenate([signal, padding])


def mono_to_stereo(signal: np.ndarray) -> np.ndarray:
    """Convert mono to stereo by duplicating the channel."""
    if signal.ndim == 2:
        return signal
    return np.column_stack([signal, signal])


def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert stereo to mono by averaging channels."""
    if signal.ndim == 1:
        return signal
    return np.mean(signal, axis=1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def export_wav(signal: np.ndarray, path: str | Path,
               sample_rate: int = SAMPLE_RATE,
               bit_depth: int = 16) -> Path:
    """Export signal to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    subtype = subtype_map.get(bit_depth)
    if subtype is None:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 16, 24, or 32.")
    sf.write(str(path), signal, sample_rate, subtype=subtype)
    return path


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file. Returns (signal, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float64")
    return data, sr
