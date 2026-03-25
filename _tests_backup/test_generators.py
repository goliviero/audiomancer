"""Tests for fractal.generators — signal factory functions."""

import numpy as np
import pytest
from pathlib import Path

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE
from fractal.generators import (
    sine,
    square,
    sawtooth,
    triangle,
    white_noise,
    pink_noise,
    binaural,
    load_sample,
)
from fractal.export import export_wav
from fractal.signal import is_mono, is_stereo


class TestSine:
    def test_shape(self):
        sig = sine(440, 1.0)
        assert is_mono(sig)
        assert len(sig) == SAMPLE_RATE

    def test_amplitude(self):
        sig = sine(440, 1.0, amplitude=0.8)
        assert np.max(np.abs(sig)) <= 0.8 + 1e-10

    def test_frequency_period(self):
        # A 1Hz sine over 1 second should cross zero ~2 times per period
        sig = sine(1.0, 1.0, amplitude=1.0)
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        assert zero_crossings == 2  # exactly 2 for 1Hz over 1 second

    def test_custom_sample_rate(self):
        sig = sine(440, 1.0, sample_rate=22050)
        assert len(sig) == 22050


class TestSquare:
    def test_values_are_binary(self):
        sig = square(100, 0.1)
        unique_vals = np.unique(sig)
        # Should only contain +amplitude and -amplitude (and possibly 0)
        assert len(unique_vals) <= 3

    def test_shape(self):
        sig = square(100, 1.0)
        assert is_mono(sig)
        assert len(sig) == SAMPLE_RATE


class TestSawtooth:
    def test_range(self):
        sig = sawtooth(100, 1.0, amplitude=0.5)
        assert np.max(sig) <= 0.5 + 1e-10
        assert np.min(sig) >= -0.5 - 1e-10

    def test_shape(self):
        sig = sawtooth(100, 1.0)
        assert is_mono(sig)


class TestTriangle:
    def test_range(self):
        sig = triangle(100, 1.0, amplitude=0.5)
        assert np.max(sig) <= 0.5 + 1e-10
        assert np.min(sig) >= -0.5 - 1e-10

    def test_shape(self):
        sig = triangle(100, 1.0)
        assert is_mono(sig)


class TestWhiteNoise:
    def test_shape(self):
        sig = white_noise(1.0)
        assert is_mono(sig)
        assert len(sig) == SAMPLE_RATE

    def test_amplitude(self):
        sig = white_noise(1.0, amplitude=0.3)
        assert np.max(np.abs(sig)) <= 0.3 + 1e-10

    def test_randomness(self):
        a = white_noise(0.1)
        b = white_noise(0.1)
        assert not np.array_equal(a, b)


class TestPinkNoise:
    def test_shape(self):
        sig = pink_noise(1.0)
        assert is_mono(sig)
        assert len(sig) == SAMPLE_RATE

    def test_amplitude(self):
        sig = pink_noise(1.0, amplitude=0.5)
        assert np.max(np.abs(sig)) <= 0.5 + 1e-10


class TestBinaural:
    def test_stereo_output(self):
        sig = binaural(200, 10, 1.0)
        assert is_stereo(sig)
        assert sig.shape == (SAMPLE_RATE, 2)

    def test_amplitude(self):
        sig = binaural(200, 10, 1.0, amplitude=0.4)
        assert np.max(np.abs(sig)) <= 0.4 + 1e-10

    def test_channels_differ(self):
        sig = binaural(200, 10, 1.0)
        # Left and right should NOT be identical (different frequencies)
        assert not np.array_equal(sig[:, 0], sig[:, 1])


class TestLoadSample:
    def test_roundtrip(self, tmp_path):
        """Write a WAV, reload it, verify content."""
        original = sine(440, 0.5)
        wav_path = tmp_path / "test.wav"
        export_wav(original, wav_path)

        loaded = load_sample(wav_path)
        assert len(loaded) == len(original)
        # Allow small rounding from int16 quantization
        np.testing.assert_allclose(loaded, original, atol=1e-4)

    def test_sample_rate_mismatch_warning(self, tmp_path):
        """Loading a file and specifying wrong sample rate should warn."""
        sig = sine(440, 0.1)
        wav_path = tmp_path / "test.wav"
        export_wav(sig, wav_path, sample_rate=44100)

        with pytest.warns(UserWarning, match="Sample rate mismatch"):
            load_sample(wav_path, sample_rate=22050)
