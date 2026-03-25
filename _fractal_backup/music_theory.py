"""Music theory — notes, scales, chords, and progressions.

Pure data + pure functions. No dependencies beyond the standard library.
Everything is based on 12-tone equal temperament (12-TET) with A4 = 440 Hz.

Usage:
    from fractal.music_theory import note_to_hz, scale_hz, chord_hz

    freq = note_to_hz("C4")              # 261.626 Hz
    notes = scale_hz("A3", "pentatonic_minor")  # [220.0, 261.63, ...]
    freqs = chord_hz("Dm7")             # D minor 7th chord frequencies
"""

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A4_HZ = 440.0
A4_MIDI = 69

# Note names in chromatic order (sharps)
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Flat equivalents for parsing
_FLAT_MAP = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B", "B#": "C",
    "E#": "F",
}

# Regex for parsing note names: letter + optional accidental + octave
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


# ---------------------------------------------------------------------------
# Note ↔ frequency conversion
# ---------------------------------------------------------------------------

def _parse_note(note: str) -> tuple[str, int]:
    """Parse a note string into (pitch_class, octave).

    Accepts: "C4", "C#4", "Db4", "c4", "Bb3", etc.
    Returns: ("C#", 4) for "Db4".
    """
    m = _NOTE_RE.match(note.strip())
    if not m:
        raise ValueError(f"Invalid note name: '{note}'. Expected format: 'C4', 'C#4', 'Bb3'.")
    letter = m.group(1).upper()
    accidental = m.group(2)
    octave = int(m.group(3))
    name = letter + accidental

    # Normalize flats and enharmonics to sharp names
    if name in _FLAT_MAP:
        name = _FLAT_MAP[name]
        # Handle octave wrap: Cb4 = B3
        if letter + accidental in ("Cb",):
            octave -= 1
        elif letter + accidental in ("B#",):
            octave += 1

    if name not in _SHARP_NAMES:
        raise ValueError(f"Unknown pitch class: '{name}'.")
    return name, octave


def _note_to_midi(note: str) -> int:
    """Convert a note name to MIDI note number. C4 = 60, A4 = 69."""
    name, octave = _parse_note(note)
    pitch_class = _SHARP_NAMES.index(name)
    return (octave + 1) * 12 + pitch_class


def _midi_to_note(midi: int) -> str:
    """Convert a MIDI note number to a note name."""
    pitch_class = midi % 12
    octave = (midi // 12) - 1
    return f"{_SHARP_NAMES[pitch_class]}{octave}"


def note_to_hz(note: str) -> float:
    """Convert a note name to frequency in Hz.

    Args:
        note: Note in scientific pitch notation (e.g., "C4", "A#3", "Bb5").

    Returns:
        Frequency in Hz.

    Examples:
        >>> note_to_hz("A4")
        440.0
        >>> note_to_hz("C4")  # middle C
        261.6255653005986
    """
    midi = _note_to_midi(note)
    return A4_HZ * 2 ** ((midi - A4_MIDI) / 12)


def hz_to_note(hz: float) -> str:
    """Find the nearest note name for a frequency.

    Args:
        hz: Frequency in Hz.

    Returns:
        Nearest note name (e.g., "A4" for 440.0).
    """
    if hz <= 0:
        raise ValueError(f"Frequency must be positive, got {hz}.")
    import math
    midi = round(A4_MIDI + 12 * math.log2(hz / A4_HZ))
    return _midi_to_note(midi)


def interval_hz(root_hz: float, semitones: int) -> float:
    """Compute the frequency at N semitones above a root frequency.

    Args:
        root_hz: Root frequency in Hz.
        semitones: Number of semitones (positive = up, negative = down).

    Returns:
        Frequency in Hz.
    """
    return root_hz * 2 ** (semitones / 12)


def transpose(note: str, semitones: int) -> str:
    """Transpose a note by N semitones.

    Args:
        note: Note name (e.g., "C4").
        semitones: Number of semitones (positive = up, negative = down).

    Returns:
        Transposed note name.

    Examples:
        >>> transpose("C4", 7)
        'G4'
        >>> transpose("A4", -12)
        'A3'
    """
    midi = _note_to_midi(note) + semitones
    return _midi_to_note(midi)


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

# Each scale is defined as semitone intervals from the root.
SCALES: dict[str, list[int]] = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "locrian":          [0, 1, 3, 5, 6, 8, 10],
    "whole_tone":       [0, 2, 4, 6, 8, 10],
    "chromatic":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "hungarian_minor":  [0, 2, 3, 6, 7, 8, 11],
}


def scale(root: str, scale_type: str = "major", octaves: int = 1) -> list[str]:
    """Return note names in a scale.

    Args:
        root: Root note (e.g., "C4").
        scale_type: Scale name (see SCALES dict).
        octaves: Number of octaves to span.

    Returns:
        List of note names including the root and the octave above.

    Examples:
        >>> scale("C4", "major")
        ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    """
    if scale_type not in SCALES:
        raise ValueError(f"Unknown scale: '{scale_type}'. Available: {list(SCALES.keys())}")
    intervals = SCALES[scale_type]
    root_midi = _note_to_midi(root)
    notes = []
    for octave_i in range(octaves):
        for interval in intervals:
            notes.append(_midi_to_note(root_midi + interval + octave_i * 12))
    # Add the final octave note
    notes.append(_midi_to_note(root_midi + octaves * 12))
    return notes


def scale_hz(root: str, scale_type: str = "major", octaves: int = 1) -> list[float]:
    """Return frequencies (Hz) for a scale.

    Same as scale() but returns Hz values, ready for generators.

    Examples:
        >>> scale_hz("A4", "pentatonic_minor")
        [440.0, 523.25, 587.33, 659.26, 783.99, 880.0]
    """
    return [note_to_hz(n) for n in scale(root, scale_type, octaves)]


# ---------------------------------------------------------------------------
# Chords
# ---------------------------------------------------------------------------

# Each chord type is defined as semitone intervals from the root.
CHORD_TYPES: dict[str, list[int]] = {
    "major":  [0, 4, 7],
    "minor":  [0, 3, 7],
    "dim":    [0, 3, 6],
    "aug":    [0, 4, 8],
    "sus2":   [0, 2, 7],
    "sus4":   [0, 5, 7],
    "7":      [0, 4, 7, 10],       # dominant 7th
    "maj7":   [0, 4, 7, 11],       # major 7th
    "min7":   [0, 3, 7, 10],       # minor 7th
    "dim7":   [0, 3, 6, 9],        # diminished 7th
    "min_maj7": [0, 3, 7, 11],     # minor major 7th
    "6":      [0, 4, 7, 9],        # major 6th
    "min6":   [0, 3, 7, 9],        # minor 6th
    "9":      [0, 4, 7, 10, 14],   # dominant 9th
    "maj9":   [0, 4, 7, 11, 14],   # major 9th
    "min9":   [0, 3, 7, 10, 14],   # minor 9th
    "add9":   [0, 4, 7, 14],       # add 9 (no 7th)
    "power":  [0, 7],              # power chord (root + fifth)
}

# Shorthand parsing: "Cm" → ("C", "minor"), "Dmaj7" → ("D", "maj7")
_CHORD_SHORTHAND = re.compile(
    r"^([A-G][#b]?)(m(?:aj|in)?7?|maj[79]?|min[679]?|dim7?|aug|sus[24]|add9|7|9|6|power)?$"
)
_SHORTHAND_MAP = {
    "": "major", "m": "minor", "min": "minor", "maj": "major",
    "7": "7", "maj7": "maj7", "min7": "min7", "m7": "min7",
    "mmaj7": "min_maj7", "dim": "dim", "dim7": "dim7",
    "aug": "aug", "sus2": "sus2", "sus4": "sus4",
    "6": "6", "min6": "min6", "m6": "min6",
    "9": "9", "maj9": "maj9", "min9": "min9", "m9": "min9",
    "add9": "add9", "power": "power",
}


def _parse_chord_shorthand(name: str) -> tuple[str, str] | None:
    """Try to parse a chord shorthand like 'Cm7' or 'Dmaj7'.

    Returns (root_with_octave, chord_type) or None if not a shorthand.
    """
    m = _CHORD_SHORTHAND.match(name)
    if not m:
        return None
    root_letter = m.group(1)
    quality = m.group(2) or ""
    chord_type = _SHORTHAND_MAP.get(quality)
    if chord_type is None:
        return None
    return root_letter, chord_type


def chord(root: str, chord_type: str = "major", inversion: int = 0) -> list[str]:
    """Return note names in a chord.

    Args:
        root: Root note with octave (e.g., "C4") or shorthand (e.g., "Cm7").
              If shorthand has no octave, defaults to octave 4.
        chord_type: Chord quality (see CHORD_TYPES dict). Ignored if root is shorthand.
        inversion: 0 = root position, 1 = first inversion, etc.

    Returns:
        List of note names.

    Examples:
        >>> chord("C4", "major")
        ['C4', 'E4', 'G4']
        >>> chord("C4", "major", inversion=1)
        ['E4', 'G4', 'C5']
        >>> chord("Am7")  # shorthand, defaults to octave 4
        ['A4', 'C5', 'E5', 'G5']
    """
    # Try shorthand parsing (e.g., "Cm7", "Dmaj7")
    parsed = _parse_chord_shorthand(root)
    if parsed:
        root_letter, chord_type = parsed
        # Check if the original string had an octave
        if re.match(r"^[A-G][#b]?\d", root):
            pass  # Has octave, use _parse_note directly
        else:
            root = root_letter + "4"  # Default octave 4

    if chord_type not in CHORD_TYPES:
        raise ValueError(f"Unknown chord type: '{chord_type}'. Available: {list(CHORD_TYPES.keys())}")

    intervals = CHORD_TYPES[chord_type]
    root_midi = _note_to_midi(root)

    midi_notes = [root_midi + i for i in intervals]

    # Apply inversion: move bottom notes up an octave
    for _ in range(inversion % len(midi_notes)):
        midi_notes.append(midi_notes.pop(0) + 12)

    return [_midi_to_note(m) for m in midi_notes]


def chord_hz(root: str, chord_type: str = "major", inversion: int = 0) -> list[float]:
    """Return frequencies (Hz) for a chord.

    Same as chord() but returns Hz values.

    Examples:
        >>> chord_hz("A4", "minor")
        [440.0, 523.25, 659.26]
    """
    return [note_to_hz(n) for n in chord(root, chord_type, inversion)]


# ---------------------------------------------------------------------------
# Chord progressions
# ---------------------------------------------------------------------------

# Progressions as scale degree offsets (0-indexed).
# Each degree maps to a chord root in the key.
PROGRESSIONS: dict[str, list[int]] = {
    "I_IV_V_I":     [0, 3, 4, 0],       # Classic
    "I_V_vi_IV":    [0, 4, 5, 3],       # Pop (Let It Be, etc.)
    "ii_V_I":       [1, 4, 0],           # Jazz
    "I_vi_IV_V":    [0, 5, 3, 4],       # 50s doo-wop
    "I_IV_vi_V":    [0, 3, 5, 4],       # Axis
    "vi_IV_I_V":    [5, 3, 0, 4],       # Am-F-C-G pattern
    "I_V_vi_iii_IV": [0, 4, 5, 2, 3],  # Canon in D
    "blues_12bar":  [0, 0, 0, 0, 3, 3, 0, 0, 4, 3, 0, 4],  # 12-bar blues
    "i_VII_VI_V":   [0, 6, 5, 4],       # Andalusian cadence
}

# Default chord quality for each scale degree in major key
_MAJOR_DEGREE_QUALITY = ["major", "minor", "minor", "major", "major", "minor", "dim"]


def progression(key: str, prog_name: str, scale_type: str = "major") -> list[list[str]]:
    """Return a list of chords for a named progression in a key.

    Args:
        key: Root note with octave (e.g., "C4").
        prog_name: Progression name (see PROGRESSIONS dict).
        scale_type: Scale to derive chord qualities from.

    Returns:
        List of chords, each chord as a list of note names.

    Examples:
        >>> progression("C4", "I_V_vi_IV")
        [['C4', 'E4', 'G4'], ['G4', 'B4', 'D5'], ['A4', 'C5', 'E5'], ['F4', 'A4', 'C5']]
    """
    if prog_name not in PROGRESSIONS:
        raise ValueError(f"Unknown progression: '{prog_name}'. Available: {list(PROGRESSIONS.keys())}")

    degrees = PROGRESSIONS[prog_name]
    scale_intervals = SCALES.get(scale_type, SCALES["major"])
    root_midi = _note_to_midi(key)

    chords = []
    for degree in degrees:
        # Get the note at this scale degree
        degree_idx = degree % len(scale_intervals)
        chord_root_midi = root_midi + scale_intervals[degree_idx]
        chord_root = _midi_to_note(chord_root_midi)

        # Determine chord quality from scale degree
        quality = _MAJOR_DEGREE_QUALITY[degree_idx] if scale_type == "major" else "minor"
        chords.append(chord(chord_root, quality))

    return chords


def progression_hz(key: str, prog_name: str, scale_type: str = "major") -> list[list[float]]:
    """Return Hz values for a chord progression.

    Same as progression() but returns Hz values.
    """
    return [[note_to_hz(n) for n in ch] for ch in progression(key, prog_name, scale_type)]
