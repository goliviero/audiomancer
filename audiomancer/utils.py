"""Utilities — I/O, normalization, fade, signal helpers.

All functions operate on numpy arrays. Mono = (n,), Stereo = (n, 2).
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from audiomancer import SAMPLE_RATE

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
    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    subtype = subtype_map.get(bit_depth)
    if subtype is None:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 16, 24, or 32.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), signal, sample_rate, subtype=subtype)
    except PermissionError:
        raise PermissionError(f"Cannot write to {path} — check directory permissions")
    except OSError as e:
        raise OSError(f"Failed to export WAV to {path}: {e}") from e
    return path


def load_audio(path: str | Path,
               target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file. Returns (signal, sample_rate).

    Args:
        path: Audio file path (WAV, FLAC, etc).
        target_sr: If provided and differs from the file's SR, the signal is
            resampled via scipy.signal.resample_poly (high-quality polyphase,
            not linear interp). Returns (resampled_signal, target_sr).
            If None (default), returns raw signal + native SR.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        data, sr = sf.read(str(path), dtype="float64")
    except RuntimeError as e:
        raise RuntimeError(f"Cannot read {path} — unsupported or corrupted file: {e}") from e

    if target_sr is not None and sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(target_sr, sr)
        up, down = target_sr // g, sr // g
        if data.ndim == 2:
            data = np.column_stack([
                resample_poly(data[:, c], up, down) for c in range(data.shape[1])
            ])
        else:
            data = resample_poly(data, up, down)
        sr = target_sr

    return data, sr


def load_sample(name: str, target_sr: int = SAMPLE_RATE,
                samples_dir: Path | None = None,
                target_db: float = -1.0) -> np.ndarray:
    """Load a named sample from the samples/ bank.

    Searches samples/own/ first (priority to personal recordings), then
    samples/cc0/. Extension is detected (.wav, .flac, .mp3).

    Args:
        name: Sample name without extension (e.g., "piano_C3_mezzo").
        target_sr: Target sample rate (auto-resample if needed).
        samples_dir: Override samples/ location (default = repo_root/samples/).
        target_db: Peak normalization target in dBFS.

    Returns:
        Stereo signal, normalized to target_db. Mono sources are duplicated.

    Raises:
        FileNotFoundError: if no matching file is found.
    """
    if samples_dir is None:
        # Default: sibling of the audiomancer package
        samples_dir = Path(__file__).resolve().parent.parent / "samples"
    samples_dir = Path(samples_dir)

    exts = (".wav", ".flac", ".mp3")
    for subdir in ("own", "cc0"):
        for ext in exts:
            candidate = samples_dir / subdir / f"{name}{ext}"
            if candidate.exists():
                sig, _sr = load_audio(candidate, target_sr=target_sr)
                if sig.ndim == 1:
                    sig = mono_to_stereo(sig)
                return normalize(sig, target_db=target_db)

    raise FileNotFoundError(
        f"Sample {name!r} not found in {samples_dir}/own/ or {samples_dir}/cc0/. "
        f"Tried extensions: {exts}."
    )


def trim_silence(signal: np.ndarray, threshold_db: float = -60.0,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Trim leading and trailing silence below threshold."""
    threshold_linear = 10 ** (threshold_db / 20)
    if signal.ndim == 2:
        envelope = np.max(np.abs(signal), axis=1)
    else:
        envelope = np.abs(signal)
    above = np.where(envelope > threshold_linear)[0]
    if len(above) == 0:
        return signal[:0]
    return signal[above[0]:above[-1] + 1]


def duration(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Return signal duration in seconds."""
    return signal.shape[0] / sample_rate
