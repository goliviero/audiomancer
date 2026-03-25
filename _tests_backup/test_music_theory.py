"""Tests for fractal.music_theory — notes, scales, chords, progressions."""

import math

import pytest

from fractal.music_theory import (
    A4_HZ,
    CHORD_TYPES,
    PROGRESSIONS,
    SCALES,
    chord,
    chord_hz,
    hz_to_note,
    interval_hz,
    note_to_hz,
    progression,
    progression_hz,
    scale,
    scale_hz,
    transpose,
)


# =============================================
# Note ↔ frequency conversion
# =============================================

class TestNoteToHz:
    def test_a4_is_440(self):
        assert note_to_hz("A4") == 440.0

    def test_a3_is_220(self):
        assert abs(note_to_hz("A3") - 220.0) < 0.01

    def test_a5_is_880(self):
        assert abs(note_to_hz("A5") - 880.0) < 0.01

    def test_middle_c(self):
        assert abs(note_to_hz("C4") - 261.626) < 0.01

    def test_sharp_note(self):
        assert abs(note_to_hz("C#4") - 277.183) < 0.01

    def test_flat_note_equals_sharp(self):
        assert abs(note_to_hz("Db4") - note_to_hz("C#4")) < 0.001

    def test_cb_equals_b_below(self):
        """Cb4 should equal B3."""
        assert abs(note_to_hz("Cb4") - note_to_hz("B3")) < 0.001

    def test_case_insensitive(self):
        assert note_to_hz("a4") == note_to_hz("A4")

    def test_invalid_note_raises(self):
        with pytest.raises(ValueError):
            note_to_hz("X4")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            note_to_hz("CC4")

    def test_octave_doubling(self):
        """Going up one octave should double the frequency."""
        assert abs(note_to_hz("C5") / note_to_hz("C4") - 2.0) < 0.001


class TestHzToNote:
    def test_440_is_a4(self):
        assert hz_to_note(440.0) == "A4"

    def test_261_is_c4(self):
        assert hz_to_note(261.626) == "C4"

    def test_roundtrip(self):
        """note_to_hz -> hz_to_note should round-trip for all standard notes."""
        for note in ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]:
            hz = note_to_hz(note)
            assert hz_to_note(hz) == note

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError):
            hz_to_note(-100)

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError):
            hz_to_note(0)


class TestTranspose:
    def test_up_octave(self):
        assert transpose("C4", 12) == "C5"

    def test_down_octave(self):
        assert transpose("A4", -12) == "A3"

    def test_up_fifth(self):
        assert transpose("C4", 7) == "G4"

    def test_zero_semitones(self):
        assert transpose("E4", 0) == "E4"


class TestIntervalHz:
    def test_octave(self):
        assert abs(interval_hz(440.0, 12) - 880.0) < 0.01

    def test_fifth(self):
        assert abs(interval_hz(440.0, 7) - 659.255) < 0.01

    def test_unison(self):
        assert interval_hz(440.0, 0) == 440.0


# =============================================
# Scales
# =============================================

class TestScale:
    def test_c_major_scale(self):
        notes = scale("C4", "major")
        assert notes == ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

    def test_major_scale_length(self):
        """Major scale: 7 notes + octave = 8."""
        assert len(scale("C4", "major")) == 8

    def test_pentatonic_minor_length(self):
        """Pentatonic minor: 5 notes + octave = 6."""
        assert len(scale("A4", "pentatonic_minor")) == 6

    def test_chromatic_length(self):
        """Chromatic: 12 notes + octave = 13."""
        assert len(scale("C4", "chromatic")) == 13

    def test_two_octaves(self):
        notes = scale("C4", "major", octaves=2)
        # 7 * 2 + 1 = 15
        assert len(notes) == 15
        assert notes[0] == "C4"
        assert notes[-1] == "C6"

    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError):
            scale("C4", "nonexistent_scale")

    def test_scale_hz_returns_floats(self):
        freqs = scale_hz("A4", "pentatonic_minor")
        assert all(isinstance(f, float) for f in freqs)
        assert abs(freqs[0] - 440.0) < 0.01

    def test_scale_hz_ascending(self):
        """Frequencies should be ascending."""
        freqs = scale_hz("C4", "major")
        for i in range(1, len(freqs)):
            assert freqs[i] > freqs[i - 1]


# =============================================
# Chords
# =============================================

class TestChord:
    def test_c_major_triad(self):
        notes = chord("C4", "major")
        assert notes == ["C4", "E4", "G4"]

    def test_a_minor_triad(self):
        notes = chord("A4", "minor")
        assert notes == ["A4", "C5", "E5"]

    def test_first_inversion(self):
        notes = chord("C4", "major", inversion=1)
        assert notes == ["E4", "G4", "C5"]

    def test_second_inversion(self):
        notes = chord("C4", "major", inversion=2)
        assert notes == ["G4", "C5", "E5"]

    def test_seventh_chord(self):
        notes = chord("C4", "maj7")
        assert len(notes) == 4
        assert notes[0] == "C4"

    def test_power_chord(self):
        notes = chord("E2", "power")
        assert len(notes) == 2
        assert notes == ["E2", "B2"]

    def test_shorthand_cm(self):
        """'Cm' should parse as C minor, default octave 4."""
        notes = chord("Cm")
        assert notes[0] == "C4"
        assert len(notes) == 3

    def test_shorthand_dmaj7(self):
        notes = chord("Dmaj7")
        assert notes[0] == "D4"
        assert len(notes) == 4

    def test_unknown_chord_type_raises(self):
        with pytest.raises(ValueError):
            chord("C4", "nonexistent_chord")

    def test_chord_hz_returns_floats(self):
        freqs = chord_hz("A4", "minor")
        assert all(isinstance(f, float) for f in freqs)
        assert abs(freqs[0] - 440.0) < 0.01


# =============================================
# Progressions
# =============================================

class TestProgression:
    def test_i_v_vi_iv_length(self):
        """I-V-vi-IV has 4 chords."""
        chords = progression("C4", "I_V_vi_IV")
        assert len(chords) == 4

    def test_i_v_vi_iv_roots(self):
        """I-V-vi-IV in C: C, G, A, F."""
        chords = progression("C4", "I_V_vi_IV")
        roots = [ch[0] for ch in chords]
        assert roots == ["C4", "G4", "A4", "F4"]

    def test_blues_12bar_length(self):
        chords = progression("A3", "blues_12bar")
        assert len(chords) == 12

    def test_unknown_progression_raises(self):
        with pytest.raises(ValueError):
            progression("C4", "nonexistent_progression")

    def test_progression_hz_returns_nested_floats(self):
        result = progression_hz("C4", "ii_V_I")
        assert len(result) == 3
        assert all(isinstance(f, float) for f in result[0])


# =============================================
# Data completeness
# =============================================

class TestDataCompleteness:
    def test_all_scales_have_intervals(self):
        for name, intervals in SCALES.items():
            assert len(intervals) >= 2, f"Scale '{name}' has fewer than 2 intervals"
            assert intervals[0] == 0, f"Scale '{name}' should start at 0"

    def test_all_chord_types_have_intervals(self):
        for name, intervals in CHORD_TYPES.items():
            assert len(intervals) >= 2, f"Chord '{name}' has fewer than 2 intervals"
            assert intervals[0] == 0, f"Chord '{name}' should start at 0"

    def test_all_progressions_have_degrees(self):
        for name, degrees in PROGRESSIONS.items():
            assert len(degrees) >= 2, f"Progression '{name}' has fewer than 2 chords"
