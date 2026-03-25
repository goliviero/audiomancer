"""Tests for fractal.signal — utility functions on numpy arrays."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.signal import (
    concat,
    duration_samples,
    duration_seconds,
    is_mono,
    is_stereo,
    mix_signals,
    mono_to_stereo,
    normalize_peak,
    pad_to_length,
    silence,
    stereo_to_mono,
    trim,
)


class TestDurationConversion:
    def test_duration_samples(self):
        assert duration_samples(1.0) == SAMPLE_RATE
        assert duration_samples(0.5) == SAMPLE_RATE // 2

    def test_duration_seconds(self):
        assert duration_seconds(SAMPLE_RATE) == 1.0
        assert duration_seconds(SAMPLE_RATE * 2) == 2.0


class TestChannelDetection:
    def test_mono(self):
        mono = np.zeros(100)
        assert is_mono(mono)
        assert not is_stereo(mono)

    def test_stereo(self):
        stereo = np.zeros((100, 2))
        assert is_stereo(stereo)
        assert not is_mono(stereo)


class TestChannelConversion:
    def test_mono_to_stereo(self):
        mono = np.ones(100)
        stereo = mono_to_stereo(mono)
        assert stereo.shape == (100, 2)
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])

    def test_mono_to_stereo_idempotent(self):
        stereo = np.zeros((100, 2))
        result = mono_to_stereo(stereo)
        assert result.shape == (100, 2)

    def test_stereo_to_mono(self):
        stereo = np.column_stack([np.ones(100), np.ones(100) * 3])
        mono = stereo_to_mono(stereo)
        assert is_mono(mono)
        np.testing.assert_allclose(mono, 2.0)


class TestSilence:
    def test_mono_silence(self):
        s = silence(1.0)
        assert s.shape == (SAMPLE_RATE,)
        assert np.all(s == 0)

    def test_stereo_silence(self):
        s = silence(0.5, stereo=True)
        assert s.shape == (SAMPLE_RATE // 2, 2)


class TestNormalizePeak:
    def test_normalize_to_minus_1db(self):
        sig = np.array([0.5, -0.3, 0.2])
        result = normalize_peak(sig, target_db=-1.0)
        target_linear = 10 ** (-1.0 / 20)
        assert pytest.approx(np.max(np.abs(result)), rel=1e-6) == target_linear

    def test_normalize_silent_signal(self):
        sig = np.zeros(100)
        result = normalize_peak(sig)
        np.testing.assert_array_equal(result, sig)


class TestPadTrimConcat:
    def test_pad_to_length(self):
        sig = np.ones(50)
        padded = pad_to_length(sig, 100)
        assert len(padded) == 100
        assert np.all(padded[50:] == 0)

    def test_pad_no_op_if_long_enough(self):
        sig = np.ones(100)
        padded = pad_to_length(sig, 50)
        assert len(padded) == 100

    def test_trim(self):
        sig = np.arange(SAMPLE_RATE * 2, dtype=np.float64)
        trimmed = trim(sig, start_sec=0.5, end_sec=1.0)
        expected_len = SAMPLE_RATE // 2
        assert len(trimmed) == expected_len

    def test_concat(self):
        a = np.ones(100)
        b = np.ones(200) * 2
        result = concat(a, b)
        assert len(result) == 300


class TestMixSignals:
    def test_mix_equal_volume(self):
        a = np.ones(100)
        b = np.ones(100) * 2
        mixed = mix_signals([a, b])
        np.testing.assert_allclose(mixed, 3.0)

    def test_mix_with_volume(self):
        a = np.ones(100)
        mixed = mix_signals([a], volumes_db=[-6.0])
        # -6dB ≈ 0.5012
        assert np.max(mixed) < 1.0
        assert np.max(mixed) > 0.4

    def test_mix_different_lengths(self):
        a = np.ones(50)
        b = np.ones(100)
        mixed = mix_signals([a, b])
        assert len(mixed) == 100

    def test_mix_empty(self):
        result = mix_signals([])
        assert len(result) == 0
