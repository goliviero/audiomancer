"""Harmony — musical theory helpers for ambient sound design.

Scales, tuning systems, chord generators, intervals, and sacred frequency ratios.
Everything returns Hz frequencies ready to feed into synth/texture generators.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A4_HZ = 440.0  # Reference pitch

# Sacred / healing frequency ratios (relative to root)
SACRED_RATIOS = {
    "unison": 1 / 1,
    "octave": 2 / 1,
    "perfect_fifth": 3 / 2,
    "perfect_fourth": 4 / 3,
    "major_third": 5 / 4,
    "minor_third": 6 / 5,
    "major_sixth": 5 / 3,
    "minor_sixth": 8 / 5,
    "golden": (1 + np.sqrt(5)) / 2,  # Golden ratio ~1.618
    "phi_squared": ((1 + np.sqrt(5)) / 2) ** 2,  # ~2.618
    "sqrt2": np.sqrt(2),  # Tritone in equal temperament
    "sqrt3": np.sqrt(3),  # ~1.732
}

# Solfeggio frequencies (Hz)
SOLFEGGIO = {
    "ut": 396,   # Liberating guilt and fear
    "re": 417,   # Undoing situations, facilitating change
    "mi": 528,   # Transformation, miracles, DNA repair
    "fa": 639,   # Connecting, relationships
    "sol": 741,  # Awakening intuition
    "la": 852,   # Returning to spiritual order
    "si": 963,   # Divine consciousness
}

# Planetary frequencies (Hans Cousto octave method)
PLANETARY = {
    "earth_day": 194.18,    # Rotation period
    "earth_year": 136.10,   # Orbital period (Om frequency)
    "moon": 210.42,         # Synodic month
    "sun": 126.22,          # Central frequency
    "mercury": 141.27,
    "venus": 221.23,
    "mars": 144.72,
    "jupiter": 183.58,
    "saturn": 147.85,
}

# Note names to semitone offset from C
_NOTE_OFFSETS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


# ---------------------------------------------------------------------------
# Note / frequency conversion
# ---------------------------------------------------------------------------

def note_to_hz(name: str, tuning: float = A4_HZ) -> float:
    """Convert a note name to frequency in Hz.

    Supports sharps (#), flats (b), and octave numbers.
    Examples: 'A4' = 440, 'C#3' = 139.5, 'Bb2' = 116.5

    Args:
        name: Note name (e.g., 'A4', 'C#3', 'Eb5').
        tuning: A4 reference frequency.

    Returns:
        Frequency in Hz.
    """
    note_part = name[0].upper()
    rest = name[1:]

    semitone = _NOTE_OFFSETS[note_part]

    # Parse accidentals
    i = 0
    while i < len(rest) and rest[i] in "#b":
        if rest[i] == "#":
            semitone += 1
        else:
            semitone -= 1
        i += 1

    # Parse octave
    octave = int(rest[i:]) if rest[i:] else 4

    # MIDI note number (A4 = 69)
    midi = semitone + (octave + 1) * 12
    return tuning * 2 ** ((midi - 69) / 12)


def hz_to_note(freq: float, tuning: float = A4_HZ) -> str:
    """Convert a frequency to the nearest note name.

    Args:
        freq: Frequency in Hz.
        tuning: A4 reference frequency.

    Returns:
        Note name string (e.g., 'A4', 'C#3').
    """
    if freq <= 0:
        return "---"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi = 69 + 12 * np.log2(freq / tuning)
    midi_round = int(round(midi))
    octave = (midi_round // 12) - 1
    note_idx = midi_round % 12
    return f"{names[note_idx]}{octave}"


def midi_to_hz(midi_note: int, tuning: float = A4_HZ) -> float:
    """Convert MIDI note number to frequency."""
    return tuning * 2 ** ((midi_note - 69) / 12)


def hz_to_midi(freq: float, tuning: float = A4_HZ) -> float:
    """Convert frequency to MIDI note number (float for microtuning)."""
    if freq <= 0:
        return 0.0
    return 69 + 12 * np.log2(freq / tuning)


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

# Scale intervals in semitones from root
SCALES = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "aeolian":          [0, 2, 3, 5, 7, 8, 10],
    "locrian":          [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "chromatic":        list(range(12)),
    "whole_tone":       [0, 2, 4, 6, 8, 10],
    "diminished":       [0, 2, 3, 5, 6, 8, 9, 11],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    # Exotic / ambient-friendly scales
    "hirajoshi":        [0, 2, 3, 7, 8],         # Japanese
    "in_sen":           [0, 1, 5, 7, 10],         # Japanese
    "pelog":            [0, 1, 3, 7, 8],           # Balinese gamelan
    "raga_bhairav":     [0, 1, 4, 5, 7, 8, 11],   # Indian morning raga
    "raga_yaman":       [0, 2, 4, 6, 7, 9, 11],   # Indian evening raga (= Lydian)
    "hungarian_minor":  [0, 2, 3, 6, 7, 8, 11],
    "enigmatic":        [0, 1, 4, 6, 8, 10, 11],
    "prometheus":       [0, 2, 4, 6, 9, 10],       # Scriabin's mystic scale
}


def scale(root: str | float, name: str = "major",
          octaves: int = 1, tuning: float = A4_HZ) -> list[float]:
    """Generate scale frequencies.

    Args:
        root: Root note name ('C4') or frequency in Hz.
        name: Scale name (see SCALES dict).
        octaves: Number of octaves to span.
        tuning: A4 reference frequency.

    Returns:
        List of frequencies in Hz.
    """
    if isinstance(root, str):
        root_hz = note_to_hz(root, tuning)
    else:
        root_hz = float(root)

    intervals = SCALES[name]
    freqs = []
    for octave in range(octaves):
        for semitone in intervals:
            freq = root_hz * 2 ** ((semitone + octave * 12) / 12)
            freqs.append(freq)
    return freqs


# ---------------------------------------------------------------------------
# Tuning systems
# ---------------------------------------------------------------------------

def just_intonation(root_hz: float, degree: int) -> float:
    """Get frequency for a scale degree in just intonation.

    Uses pure harmonic ratios (no equal temperament beating).
    Perfect for ambient drones and meditation.

    Args:
        root_hz: Root frequency in Hz.
        degree: Scale degree (0-based, wraps at octave).

    Returns:
        Frequency in Hz.
    """
    # Just intonation ratios for chromatic scale
    ratios = [
        1/1,      # unison
        16/15,    # minor second
        9/8,      # major second
        6/5,      # minor third
        5/4,      # major third
        4/3,      # perfect fourth
        45/32,    # tritone
        3/2,      # perfect fifth
        8/5,      # minor sixth
        5/3,      # major sixth
        9/5,      # minor seventh
        15/8,     # major seventh
    ]
    octave = degree // 12
    step = degree % 12
    return root_hz * ratios[step] * (2 ** octave)


def pythagorean(root_hz: float, degree: int) -> float:
    """Get frequency for a scale degree in Pythagorean tuning.

    Built entirely from perfect fifths (3:2 ratio).
    Has a raw, ancient quality. Good for modal drones.

    Args:
        root_hz: Root frequency in Hz.
        degree: Scale degree (0-based).

    Returns:
        Frequency in Hz.
    """
    # Pythagorean ratios (circle of fifths)
    ratios = [
        1/1,        # unison
        256/243,    # minor second
        9/8,        # major second
        32/27,      # minor third
        81/64,      # major third
        4/3,        # perfect fourth
        729/512,    # tritone
        3/2,        # perfect fifth
        128/81,     # minor sixth
        27/16,      # major sixth
        16/9,       # minor seventh
        243/128,    # major seventh
    ]
    octave = degree // 12
    step = degree % 12
    return root_hz * ratios[step] * (2 ** octave)


def just_chord(root_hz: float, chord_type: str = "major") -> list[float]:
    """Generate a chord using just intonation ratios.

    Args:
        root_hz: Root frequency.
        chord_type: One of 'major', 'minor', 'sus2', 'sus4', 'power',
                    'maj7', 'min7', 'dim', 'aug'.

    Returns:
        List of frequencies.
    """
    chord_ratios = {
        "major":  [1/1, 5/4, 3/2],
        "minor":  [1/1, 6/5, 3/2],
        "sus2":   [1/1, 9/8, 3/2],
        "sus4":   [1/1, 4/3, 3/2],
        "power":  [1/1, 3/2],
        "maj7":   [1/1, 5/4, 3/2, 15/8],
        "min7":   [1/1, 6/5, 3/2, 9/5],
        "dim":    [1/1, 6/5, 64/45],
        "aug":    [1/1, 5/4, 25/16],
    }
    ratios = chord_ratios[chord_type]
    return [root_hz * r for r in ratios]


# ---------------------------------------------------------------------------
# Harmonic series
# ---------------------------------------------------------------------------

def harmonic_series(fundamental: float, n_harmonics: int = 8,
                    odd_only: bool = False) -> list[float]:
    """Generate frequencies from the natural harmonic series.

    Args:
        fundamental: Fundamental frequency in Hz.
        n_harmonics: Number of harmonics to generate.
        odd_only: If True, only odd harmonics (clarinet-like timbre).

    Returns:
        List of frequencies.
    """
    if odd_only:
        return [fundamental * (2 * i + 1) for i in range(n_harmonics)]
    return [fundamental * (i + 1) for i in range(n_harmonics)]


def subharmonic_series(fundamental: float, n_subharmonics: int = 4) -> list[float]:
    """Generate subharmonic frequencies below a fundamental.

    Creates deep, rumbling undertones. Great for bass drones.

    Args:
        fundamental: Fundamental frequency in Hz.
        n_subharmonics: Number of subharmonics.

    Returns:
        List of frequencies (ascending order, lowest first).
    """
    return [fundamental / (i + 2) for i in range(n_subharmonics)][::-1]


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def transpose(freq: float, semitones: float) -> float:
    """Transpose a frequency by semitones."""
    return freq * 2 ** (semitones / 12)


def interval_ratio(semitones: float) -> float:
    """Get the frequency ratio for a given interval in semitones."""
    return 2 ** (semitones / 12)


def cents_between(freq_a: float, freq_b: float) -> float:
    """Calculate the interval in cents between two frequencies."""
    if freq_a <= 0 or freq_b <= 0:
        return 0.0
    return 1200 * np.log2(freq_b / freq_a)


# ---------------------------------------------------------------------------
# Generative chord progressions
# ---------------------------------------------------------------------------

def drone_cluster(root_hz: float, spread_cents: float = 50.0,
                  n_voices: int = 5, seed: int | None = None) -> list[float]:
    """Generate a tight cluster of frequencies around a root.

    Creates dense, shimmering drone textures. The slight detuning
    produces natural beating patterns.

    Args:
        root_hz: Center frequency.
        spread_cents: Maximum deviation in cents.
        n_voices: Number of voices.
        seed: Random seed for reproducibility.

    Returns:
        List of frequencies.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-spread_cents, spread_cents, n_voices)
    return [root_hz * 2 ** (c / 1200) for c in sorted(offsets)]


def fibonacci_freqs(root_hz: float, n: int = 8) -> list[float]:
    """Generate frequencies based on the Fibonacci sequence ratios.

    Creates mathematically beautiful, nature-inspired pitch relationships.

    Args:
        root_hz: Starting frequency.
        n: Number of frequencies.

    Returns:
        List of frequencies (octave-folded to stay within 2 octaves).
    """
    fib = [1, 1]
    for _ in range(n + 5):
        fib.append(fib[-1] + fib[-2])

    freqs = []
    for i in range(1, n + 1):
        ratio = fib[i + 1] / fib[i]
        freq = root_hz * ratio
        # Fold into 2 octaves
        while freq > root_hz * 4:
            freq /= 2
        while freq < root_hz:
            freq *= 2
        freqs.append(freq)
    return sorted(set(freqs))[:n]
