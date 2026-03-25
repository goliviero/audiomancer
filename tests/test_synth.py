"""Tests for audiomancer.synth module."""

import numpy as np
import pytest

from audiomancer.synth import (
    sine, square, sawtooth, triangle, white_noise, pink_noise, drone, pad,
)

SR = 44100


class TestWaveforms:
    def test_sine_shape(self):
        sig = sine(440, 1.0, sample_rate=SR)
        assert sig.shape == (SR,)

    def test_sine_amplitude(self):
        sig = sine(440, 1.0, amplitude=0.8, sample_rate=SR)
        assert np.max(np.abs(sig)) == pytest.approx(0.8, abs=0.01)

    def test_square_values(self):
        sig = square(440, 0.1, amplitude=1.0, sample_rate=SR)
        unique = np.unique(np.sign(sig[sig != 0]))
        assert set(unique) == {-1.0, 1.0}

    def test_sawtooth_shape(self):
        sig = sawtooth(440, 1.0, sample_rate=SR)
        assert sig.shape == (SR,)

    def test_triangle_shape(self):
        sig = triangle(440, 1.0, sample_rate=SR)
        assert sig.shape == (SR,)

    def test_white_noise_shape(self):
        sig = white_noise(1.0, sample_rate=SR)
        assert sig.shape == (SR,)

    def test_pink_noise_shape(self):
        sig = pink_noise(1.0, sample_rate=SR)
        assert sig.shape == (SR,)


class TestDrone:
    def test_drone_shape(self):
        sig = drone(100.0, 2.0, sample_rate=SR)
        assert sig.shape == (SR * 2,)

    def test_drone_custom_harmonics(self):
        harmonics = [(1, 1.0), (2, 0.5), (3, 0.25)]
        sig = drone(100.0, 1.0, harmonics=harmonics, sample_rate=SR)
        assert sig.shape == (SR,)

    def test_drone_amplitude(self):
        sig = drone(100.0, 1.0, amplitude=0.5, sample_rate=SR)
        assert np.max(np.abs(sig)) == pytest.approx(0.5, abs=0.01)


class TestPad:
    def test_pad_shape(self):
        sig = pad(220.0, 2.0, sample_rate=SR)
        assert sig.shape == (SR * 2,)

    def test_pad_voices(self):
        sig1 = pad(220.0, 1.0, voices=1, sample_rate=SR)
        sig5 = pad(220.0, 1.0, voices=5, sample_rate=SR)
        # Both should be valid signals
        assert sig1.shape == sig5.shape

    def test_pad_amplitude(self):
        sig = pad(220.0, 1.0, amplitude=0.7, sample_rate=SR)
        assert np.max(np.abs(sig)) == pytest.approx(0.7, abs=0.01)
