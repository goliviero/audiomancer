"""Export — render signals to audio files.

Supports WAV, FLAC, and any format soundfile handles.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from fractal.constants import SAMPLE_RATE, BIT_DEPTH


# Map bit depth to soundfile subtypes
_SUBTYPE_MAP = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",
}


def export_wav(signal: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE,
               bit_depth: int = BIT_DEPTH) -> Path:
    """Export a signal to WAV file.

    Args:
        signal: Audio signal (mono or stereo float64).
        path: Output file path.
        sample_rate: Sample rate in Hz.
        bit_depth: Bit depth (16, 24, or 32).

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    subtype = _SUBTYPE_MAP.get(bit_depth)
    if subtype is None:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 16, 24, or 32.")

    sf.write(str(path), signal, sample_rate, subtype=subtype)
    return path


def export_flac(signal: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Export a signal to FLAC file (lossless compression).

    Args:
        signal: Audio signal (mono or stereo float64).
        path: Output file path.
        sample_rate: Sample rate in Hz.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, sample_rate, format="FLAC", subtype="PCM_16")
    return path


def export_auto(signal: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE,
                bit_depth: int = BIT_DEPTH) -> Path:
    """Export a signal, inferring format from file extension.

    Supported: .wav, .flac
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".wav":
        return export_wav(signal, path, sample_rate, bit_depth)
    elif ext == ".flac":
        return export_flac(signal, path, sample_rate)
    else:
        raise ValueError(f"Unsupported export format: {ext}. Use .wav or .flac.")
