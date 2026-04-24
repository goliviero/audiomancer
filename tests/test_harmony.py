"""Tests for audiomancer.harmony — musical theory helpers."""

import pytest

from audiomancer.harmony import (
    PLANETARY,
    SACRED_RATIOS,
    SCALES,
    SOLFEGGIO,
    arpeggio_from_chord,
    cents_between,
    drone_cluster,
    fibonacci_freqs,
    harmonic_series,
    hz_to_midi,
    hz_to_note,
    interval_ratio,
    just_chord,
    just_intonation,
    midi_to_hz,
    note_to_hz,
    pythagorean,
    scale,
    subharmonic_series,
    transpose,
)

# ---------------------------------------------------------------------------
# Note / frequency conversion
# ---------------------------------------------------------------------------

class TestNoteConversion:
    def test_a4_is_440(self):
        assert note_to_hz("A4") == pytest.approx(440.0)

    def test_c4_middle_c(self):
        assert note_to_hz("C4") == pytest.approx(261.626, rel=1e-3)

    def test_sharp(self):
        assert note_to_hz("C#4") == pytest.approx(277.183, rel=1e-3)

    def test_flat(self):
        assert note_to_hz("Db4") == pytest.approx(277.183, rel=1e-3)

    def test_octave_doubling(self):
        assert note_to_hz("A5") == pytest.approx(880.0)
        assert note_to_hz("A3") == pytest.approx(220.0)

    def test_hz_to_note_a4(self):
        assert hz_to_note(440.0) == "A4"

    def test_hz_to_note_c4(self):
        assert hz_to_note(261.63) == "C4"

    def test_roundtrip(self):
        for note in ["C3", "F#5", "Bb2", "G4"]:
            hz = note_to_hz(note)
            back = hz_to_note(hz)
            # Flats get converted to sharps, so check frequency match
            assert note_to_hz(back) == pytest.approx(hz, rel=1e-2)

    def test_custom_tuning(self):
        assert note_to_hz("A4", tuning=432.0) == pytest.approx(432.0)


class TestMidi:
    def test_midi_69_is_a4(self):
        assert midi_to_hz(69) == pytest.approx(440.0)

    def test_midi_60_is_c4(self):
        assert midi_to_hz(60) == pytest.approx(261.626, rel=1e-3)

    def test_hz_to_midi_440(self):
        assert hz_to_midi(440.0) == pytest.approx(69.0)

    def test_roundtrip(self):
        for midi in [36, 48, 60, 72, 84]:
            hz = midi_to_hz(midi)
            back = hz_to_midi(hz)
            assert back == pytest.approx(midi, abs=0.01)


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

class TestScales:
    def test_major_scale_7_notes(self):
        freqs = scale("C4", "major")
        assert len(freqs) == 7

    def test_pentatonic_5_notes(self):
        freqs = scale("A3", "pentatonic_minor")
        assert len(freqs) == 5

    def test_chromatic_12_notes(self):
        freqs = scale("C4", "chromatic")
        assert len(freqs) == 12

    def test_multi_octave(self):
        freqs = scale("C4", "major", octaves=2)
        assert len(freqs) == 14

    def test_ascending_order(self):
        freqs = scale("C4", "major", octaves=2)
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1]

    def test_root_as_float(self):
        freqs = scale(440.0, "major")
        assert len(freqs) == 7
        assert freqs[0] == pytest.approx(440.0)

    def test_all_scales_defined(self):
        for name in SCALES:
            freqs = scale("C4", name)
            assert len(freqs) > 0

    def test_exotic_scales_exist(self):
        for name in ["hirajoshi", "pelog", "raga_bhairav", "prometheus"]:
            assert name in SCALES


# ---------------------------------------------------------------------------
# Tuning systems
# ---------------------------------------------------------------------------

class TestTuning:
    def test_just_unison(self):
        assert just_intonation(440.0, 0) == pytest.approx(440.0)

    def test_just_octave(self):
        assert just_intonation(440.0, 12) == pytest.approx(880.0)

    def test_just_fifth(self):
        assert just_intonation(440.0, 7) == pytest.approx(440.0 * 3 / 2)

    def test_pythagorean_unison(self):
        assert pythagorean(440.0, 0) == pytest.approx(440.0)

    def test_pythagorean_fifth(self):
        assert pythagorean(440.0, 7) == pytest.approx(440.0 * 3 / 2)

    def test_just_chord_major(self):
        freqs = just_chord(440.0, "major")
        assert len(freqs) == 3
        assert freqs[0] == pytest.approx(440.0)
        assert freqs[1] == pytest.approx(440.0 * 5 / 4)
        assert freqs[2] == pytest.approx(440.0 * 3 / 2)

    def test_just_chord_types(self):
        for chord_type in ["major", "minor", "sus2", "sus4", "power",
                           "maj7", "min7", "dim", "aug"]:
            freqs = just_chord(440.0, chord_type)
            assert len(freqs) >= 2


# ---------------------------------------------------------------------------
# Harmonic series
# ---------------------------------------------------------------------------

class TestHarmonicSeries:
    def test_fundamental_included(self):
        freqs = harmonic_series(100.0, 4)
        assert freqs[0] == pytest.approx(100.0)

    def test_correct_count(self):
        freqs = harmonic_series(100.0, 8)
        assert len(freqs) == 8

    def test_harmonic_ratios(self):
        freqs = harmonic_series(100.0, 4)
        assert freqs == [100.0, 200.0, 300.0, 400.0]

    def test_odd_only(self):
        freqs = harmonic_series(100.0, 4, odd_only=True)
        assert freqs == [100.0, 300.0, 500.0, 700.0]

    def test_subharmonics(self):
        freqs = subharmonic_series(440.0, 4)
        assert len(freqs) == 4
        assert all(f < 440.0 for f in freqs)
        # Should be ascending
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1]


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

class TestIntervals:
    def test_transpose_octave(self):
        assert transpose(440.0, 12) == pytest.approx(880.0)

    def test_transpose_down(self):
        assert transpose(440.0, -12) == pytest.approx(220.0)

    def test_interval_ratio_octave(self):
        assert interval_ratio(12) == pytest.approx(2.0)

    def test_cents_octave(self):
        assert cents_between(440.0, 880.0) == pytest.approx(1200.0)

    def test_cents_zero(self):
        assert cents_between(440.0, 440.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Generative
# ---------------------------------------------------------------------------

class TestGenerative:
    def test_drone_cluster_count(self):
        freqs = drone_cluster(440.0, n_voices=5, seed=42)
        assert len(freqs) == 5

    def test_drone_cluster_near_root(self):
        freqs = drone_cluster(440.0, spread_cents=50, seed=42)
        for f in freqs:
            assert abs(cents_between(440.0, f)) <= 50.0

    def test_drone_cluster_reproducible(self):
        a = drone_cluster(440.0, seed=42)
        b = drone_cluster(440.0, seed=42)
        assert a == b

    def test_fibonacci_freqs(self):
        freqs = fibonacci_freqs(440.0, n=6)
        assert len(freqs) <= 6
        assert all(f > 0 for f in freqs)

    def test_fibonacci_within_range(self):
        root = 440.0
        freqs = fibonacci_freqs(root, n=8)
        for f in freqs:
            assert f >= root
            assert f <= root * 4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_solfeggio_values(self):
        assert SOLFEGGIO["mi"] == 528
        assert SOLFEGGIO["ut"] == 396

    def test_planetary_om(self):
        assert PLANETARY["earth_year"] == pytest.approx(136.10)

    def test_sacred_ratios(self):
        assert SACRED_RATIOS["perfect_fifth"] == pytest.approx(1.5)


class TestArpeggioFromChord:
    def test_cmaj_up(self):
        freqs = arpeggio_from_chord("Cmaj", pattern="up", octaves=1)
        # 3 notes: root, major 3rd, 5th
        assert len(freqs) == 3
        assert freqs[0] < freqs[1] < freqs[2]

    def test_cmaj9_2_octaves(self):
        freqs = arpeggio_from_chord("Cmaj9", pattern="up", octaves=2)
        # Cmaj9 = 5 notes, 2 octaves -> 10 notes
        assert len(freqs) == 10

    def test_up_down_is_palindrome(self):
        up_down = arpeggio_from_chord("Cmaj", pattern="up_down", octaves=1)
        # First == last (palindrome)
        assert up_down[0] == pytest.approx(up_down[-1])

    def test_down_reverses_up(self):
        up = arpeggio_from_chord("Cmaj", pattern="up", octaves=1)
        down = arpeggio_from_chord("Cmaj", pattern="down", octaves=1)
        assert down == list(reversed(up))

    def test_am7(self):
        """Am7 = minor 7th chord."""
        freqs = arpeggio_from_chord("Am7", pattern="up", octaves=1)
        assert len(freqs) == 4  # root, m3, 5, m7

    def test_random_pattern_is_shuffled(self):
        freqs = arpeggio_from_chord("Cmaj9", pattern="random",
                                    octaves=1, seed=42)
        base = arpeggio_from_chord("Cmaj9", pattern="up", octaves=1)
        # Same frequencies, different order
        assert sorted(freqs) == sorted(base)

    def test_invalid_chord_type_raises(self):
        with pytest.raises(ValueError):
            arpeggio_from_chord("Cwat", pattern="up")

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError):
            arpeggio_from_chord("Cmaj", pattern="upsidedown")
        assert SACRED_RATIOS["golden"] == pytest.approx(1.618, rel=1e-3)
