"""Sample-based instrument player — free pitch + ambient transforms.

The practical alternative to synthesis for ethnic/acoustic instruments:
load a CC0 single-shot WAV and play it at any target pitch.

Two modes:
    play_note(...)    — preserve the source timbre, shift to any frequency
    pitched_pad(...)  — stretch the sample into a long ambient pad (paulstretch)

Multisample support: load_multisample() auto-detects note names from
filenames (e.g. 'handpan_D3.wav', 'handpan_A3.wav') and picks the closest
pitch at play time, shifting the smallest interval.

See samples/README.md for CC0 source recommendations.
"""

from pathlib import Path

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.utils import load_audio, mono_to_stereo


def play_note(sample: np.ndarray, source_hz: float, target_hz: float,
              duration_sec: float | None = None,
              amplitude: float = 0.8,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pitch-shift a sample from source_hz to target_hz.

    Uses scipy.signal.resample_poly for high-quality polyphase resampling.
    Beyond +-5 semitones the timbre starts to shift audibly ("chipmunk"
    going up, "muddy" going down) — for ambient pads use pitched_pad which
    combines pitch-shift with paulstretch for a more forgiving result.

    Args:
        sample: Source signal (mono or stereo).
        source_hz: Native pitch of the sample.
        target_hz: Desired pitch.
        duration_sec: If set, trim or loop to this length.
        amplitude: Peak normalize target.
        sample_rate: Sample rate of the source (must match).

    Returns: Signal shifted to target_hz, same channels as source.
    """
    from math import gcd

    from scipy.signal import resample_poly

    # resample_poly(sig, up, down) outputs len(sig) * up / down samples.
    # Played at the same SR, that changes the perceived pitch by down/up.
    # To raise pitch by `ratio`, we need output shorter by `ratio` =>
    # up/down = 1/ratio => up = den_of_ratio, down = num_of_ratio.
    ratio = target_hz / source_hz
    num = int(round(ratio * 10000))
    den = 10000
    g = gcd(num, den)
    up, down = den // g, num // g

    if sample.ndim == 2:
        shifted = np.column_stack([
            resample_poly(sample[:, 0], up, down),
            resample_poly(sample[:, 1], up, down),
        ])
    else:
        shifted = resample_poly(sample, up, down)

    # Optional: fit to duration
    if duration_sec is not None:
        target_n = int(duration_sec * sample_rate)
        if shifted.shape[0] >= target_n:
            shifted = shifted[:target_n]
        else:
            # Loop-tile with fade to stay musical
            repeats = (target_n // shifted.shape[0]) + 1
            if shifted.ndim == 2:
                shifted = np.tile(shifted, (repeats, 1))[:target_n]
            else:
                shifted = np.tile(shifted, repeats)[:target_n]

    # Peak normalize
    peak = np.max(np.abs(shifted))
    if peak > 0:
        shifted = shifted * amplitude / peak
    return shifted


def pitched_pad(sample: np.ndarray, source_hz: float, target_hz: float,
                duration_sec: float = 60.0,
                stretch_factor: float | None = None,
                window_sec: float = 0.4,
                amplitude: float = 0.7,
                seed: int | None = None,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Turn a short sample into a long ambient pad at any pitch.

    Chain:
        pitch-shift sample -> source_hz -> target_hz
        paulstretch to reach duration_sec (forgiving to extreme stretches)

    This is the "free-pitch" method for instruments you only have a single
    sample of (handpan D3 -> play any note, hold for 5 min as a pad).

    Args:
        sample: Source signal (mono or stereo). Stereo is processed per channel.
        source_hz: Native pitch of the sample.
        target_hz: Desired pitch.
        duration_sec: Target pad length.
        stretch_factor: If None, auto-computed to reach duration_sec from the
            shifted sample length. Set explicitly for fine-grain control.
        window_sec: Paulstretch window (larger = smoother pad, smaller = grainy).
        seed: Seed for phase randomization.
        sample_rate: Sample rate.

    Returns: Stereo padded signal.
    """
    from audiomancer.spectral import paulstretch

    shifted = play_note(sample, source_hz, target_hz,
                        amplitude=0.9, sample_rate=sample_rate)
    if shifted.ndim == 1:
        shifted = mono_to_stereo(shifted)

    # Determine stretch to hit target duration
    shifted_dur = shifted.shape[0] / sample_rate
    if stretch_factor is None:
        stretch_factor = max(1.0, duration_sec / max(shifted_dur, 0.01))

    stretched = paulstretch(shifted, stretch_factor=stretch_factor,
                            window_sec=window_sec, seed=seed or 42,
                            sample_rate=sample_rate)

    # Trim / pad to exact duration
    target_n = int(duration_sec * sample_rate)
    if stretched.shape[0] >= target_n:
        stretched = stretched[:target_n]
    else:
        # Pad with silence (stretch should usually be long enough)
        pad_n = target_n - stretched.shape[0]
        if stretched.ndim == 2:
            stretched = np.concatenate(
                [stretched, np.zeros((pad_n, 2))], axis=0
            )
        else:
            stretched = np.concatenate([stretched, np.zeros(pad_n)])

    # Peak normalize
    peak = np.max(np.abs(stretched))
    if peak > 0:
        stretched = stretched * amplitude / peak
    return stretched


# ---------------------------------------------------------------------------
# Multisample support — load several samples, pick closest pitch
# ---------------------------------------------------------------------------

_NOTE_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}


def _note_name_to_hz(note: str, a4_hz: float = 440.0) -> float:
    """Convert 'A3' or 'F#2' to Hz. Same logic as audiomancer.harmony.note_to_hz."""
    if len(note) >= 2 and note[1] in "#b":
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[:1]
        octave = int(note[1:])
    semitone = _NOTE_OFFSETS[name] + (octave + 1) * 12
    return a4_hz * 2 ** ((semitone - 69) / 12)


def load_multisample(prefix: str, samples_dir: Path | None = None,
                     target_sr: int = SAMPLE_RATE) -> dict[float, np.ndarray]:
    """Load every 'prefix_<NOTE>.wav' into a dict {pitch_hz: sample}.

    Args:
        prefix: e.g. 'handpan' matches 'handpan_D3.wav', 'handpan_A3.wav'.
        samples_dir: Defaults to repo_root/samples/. Searches own/ then cc0/.
        target_sr: Auto-resample to this SR.

    Returns:
        dict mapping each sample's native pitch (Hz) to the loaded signal.

    Raises:
        FileNotFoundError: if no matching samples are found.
    """
    if samples_dir is None:
        samples_dir = Path(__file__).resolve().parent.parent / "samples"
    samples_dir = Path(samples_dir)

    found = {}
    for subdir in ("own", "cc0"):
        sub = samples_dir / subdir
        if not sub.exists():
            continue
        for path in sub.glob(f"{prefix}_*.wav"):
            # Extract note name from stem (e.g. 'handpan_D3' -> 'D3')
            note = path.stem[len(prefix) + 1:]
            try:
                pitch = _note_name_to_hz(note)
            except (KeyError, ValueError):
                continue
            sig, _ = load_audio(path, target_sr=target_sr)
            found[pitch] = sig

    if not found:
        raise FileNotFoundError(
            f"No samples matching {prefix}_*.wav in {samples_dir}/own/ or "
            f"{samples_dir}/cc0/."
        )
    return found


def play_note_multisample(samples: dict[float, np.ndarray],
                          target_hz: float,
                          duration_sec: float | None = None,
                          amplitude: float = 0.8,
                          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pick the closest-pitch sample and shift the smallest interval.

    Args:
        samples: dict from load_multisample().
        target_hz: Desired note.
        duration_sec: Optional trim/loop.

    Returns: shifted signal.
    """
    if not samples:
        raise ValueError("No samples provided")
    # Pick sample with smallest log-pitch distance
    closest_hz = min(samples.keys(), key=lambda h: abs(np.log2(h / target_hz)))
    return play_note(samples[closest_hz], closest_hz, target_hz,
                     duration_sec=duration_sec, amplitude=amplitude,
                     sample_rate=sample_rate)
