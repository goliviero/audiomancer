"""Tests for fractal.generative -- random melody, ambient texture, chord progression."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.music_theory import scale_hz
from fractal.generative import (
    random_melody, weighted_random_notes, evolving_parameter,
    phrase_generator, ambient_texture, chord_progression_render,
)


SR = SAMPLE_RATE


# =============================================
# Random Melody
# =============================================

class TestRandomMelody:
    def test_returns_correct_count(self):
        notes = scale_hz("C4", "major")
        melody = random_melody(notes, n_notes=8, seed=42)
        assert len(melody) == 8

    def test_each_entry_is_tuple(self):
        notes = scale_hz("C4", "major")
        melody = random_melody(notes, n_notes=4, seed=42)
        for freq, dur in melody:
            assert isinstance(freq, float)
            assert isinstance(dur, float)

    def test_rests_with_high_probability(self):
        notes = scale_hz("C4", "major")
        melody = random_melody(notes, n_notes=20, rest_probability=0.9, seed=42)
        rests = sum(1 for freq, _ in melody if freq == 0.0)
        assert rests > 10  # most should be rests

    def test_seed_reproducibility(self):
        notes = scale_hz("A3", "pentatonic_minor")
        m1 = random_melody(notes, n_notes=8, seed=123)
        m2 = random_melody(notes, n_notes=8, seed=123)
        assert m1 == m2

    def test_different_seeds_differ(self):
        notes = scale_hz("A3", "pentatonic_minor")
        m1 = random_melody(notes, n_notes=8, seed=1)
        m2 = random_melody(notes, n_notes=8, seed=2)
        assert m1 != m2


# =============================================
# Weighted Random Notes
# =============================================

class TestWeightedRandomNotes:
    def test_returns_correct_count(self):
        notes = scale_hz("C4", "major")
        result = weighted_random_notes(notes, n_notes=10, seed=42)
        assert len(result) == 10

    def test_custom_weights(self):
        """With extreme weights, should strongly favor one note."""
        notes = [100.0, 200.0, 300.0]
        weights = [100.0, 0.0, 0.0]
        result = weighted_random_notes(notes, weights=weights, n_notes=10, seed=42)
        assert all(f == 100.0 for f in result)

    def test_seed_reproducibility(self):
        notes = scale_hz("C4", "major")
        r1 = weighted_random_notes(notes, n_notes=8, seed=42)
        r2 = weighted_random_notes(notes, n_notes=8, seed=42)
        assert r1 == r2


# =============================================
# Evolving Parameter
# =============================================

class TestEvolvingParameter:
    def test_returns_correct_length(self):
        result = evolving_parameter(1000, 500, 1.0, seed=42)
        assert len(result) == SR

    def test_stays_in_range(self):
        result = evolving_parameter(1000, 500, 1.0, seed=42)
        assert np.min(result) >= 500
        assert np.max(result) <= 1500

    def test_seed_reproducibility(self):
        r1 = evolving_parameter(1000, 500, 0.5, seed=42)
        r2 = evolving_parameter(1000, 500, 0.5, seed=42)
        assert np.allclose(r1, r2)


# =============================================
# Phrase Generator
# =============================================

class TestPhraseGenerator:
    def test_returns_signal(self):
        notes = scale_hz("C4", "pentatonic_minor")
        sig = phrase_generator(notes, "pluck", tempo_bpm=120,
                               measures=1, seed=42)
        assert len(sig) > 0
        assert np.isfinite(sig).all()

    def test_seed_reproducibility(self):
        notes = scale_hz("C4", "pentatonic_minor")
        s1 = phrase_generator(notes, "pluck", measures=1, seed=42)
        s2 = phrase_generator(notes, "pluck", measures=1, seed=42)
        assert np.allclose(s1, s2)


# =============================================
# Ambient Texture
# =============================================

class TestAmbientTexture:
    def test_returns_signal(self):
        sig = ambient_texture(key="D3", duration_sec=5.0, layers=1, seed=42)
        assert len(sig) > 0
        assert np.isfinite(sig).all()

    def test_duration_respected(self):
        dur = 5.0
        sig = ambient_texture(key="D3", duration_sec=dur, layers=1, seed=42)
        # Allow some tolerance for reverb tail
        expected = int(SR * dur)
        assert abs(len(sig) - expected) < SR  # within 1 second

    def test_seed_reproducibility(self):
        s1 = ambient_texture(key="A2", duration_sec=3.0, layers=1, seed=42)
        s2 = ambient_texture(key="A2", duration_sec=3.0, layers=1, seed=42)
        assert np.allclose(s1, s2)


# =============================================
# Chord Progression Render
# =============================================

class TestChordProgressionRender:
    def test_returns_signal(self):
        sig = chord_progression_render(key="C4", prog_name="I_V_vi_IV",
                                       preset="pluck", bars_per_chord=1,
                                       tempo_bpm=120)
        assert len(sig) > 0
        assert np.isfinite(sig).all()

    def test_amplitude_bounded(self):
        sig = chord_progression_render(key="C4", prog_name="I_V_vi_IV",
                                       preset="pluck", bars_per_chord=1,
                                       amplitude=0.5)
        assert np.max(np.abs(sig)) <= 0.51
