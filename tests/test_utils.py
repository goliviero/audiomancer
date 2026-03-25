"""Tests for audiomancer.utils module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from audiomancer.utils import (
    silence, normalize, fade_in, fade_out, concat, pad_to_length,
    mono_to_stereo, stereo_to_mono, export_wav, load_audio,
    trim_silence, duration,
)

SR = 44100


class TestSignalHelpers:
    def test_silence_mono(self):
        sig = silence(1.0, sample_rate=SR)
        assert sig.shape == (SR,)
        assert np.all(sig == 0)

    def test_silence_stereo(self):
        sig = silence(1.0, stereo=True, sample_rate=SR)
        assert sig.shape == (SR, 2)

    def test_normalize(self):
        sig = np.array([0.5, -0.3, 0.2])
        normed = normalize(sig, target_db=-6.0)
        target = 10 ** (-6.0 / 20)
        assert np.max(np.abs(normed)) == pytest.approx(target, abs=0.001)

    def test_normalize_silence(self):
        sig = np.zeros(100)
        normed = normalize(sig)
        assert np.all(normed == 0)

    def test_fade_in(self):
        sig = np.ones(SR)
        faded = fade_in(sig, 0.5, sample_rate=SR)
        assert faded[0] == pytest.approx(0.0, abs=0.001)
        assert faded[-1] == pytest.approx(1.0)

    def test_fade_out(self):
        sig = np.ones(SR)
        faded = fade_out(sig, 0.5, sample_rate=SR)
        assert faded[0] == pytest.approx(1.0)
        assert faded[-1] == pytest.approx(0.0, abs=0.001)

    def test_concat(self):
        a = np.ones(100)
        b = np.zeros(50)
        c = concat(a, b)
        assert len(c) == 150

    def test_pad_to_length(self):
        sig = np.ones(50)
        padded = pad_to_length(sig, 100)
        assert len(padded) == 100
        assert padded[50] == 0.0

    def test_pad_no_op(self):
        sig = np.ones(100)
        padded = pad_to_length(sig, 50)
        assert len(padded) == 100

    def test_mono_to_stereo(self):
        sig = np.ones(100)
        stereo = mono_to_stereo(sig)
        assert stereo.shape == (100, 2)

    def test_stereo_to_mono(self):
        stereo = np.column_stack([np.ones(100), np.zeros(100)])
        mono = stereo_to_mono(stereo)
        assert mono.shape == (100,)
        assert np.all(mono == 0.5)


class TestIO:
    def test_export_and_load(self):
        sig = np.random.default_rng(42).uniform(-0.5, 0.5, SR)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            export_wav(sig, path, sample_rate=SR)
            loaded, sr = load_audio(path)
            assert sr == SR
            assert loaded.shape[0] == SR


class TestTrimSilence:
    def test_trims_leading_trailing(self):
        sig = np.zeros(SR * 3)
        sig[SR:SR * 2] = 0.5  # Sound in the middle second
        trimmed = trim_silence(sig, threshold_db=-40.0)
        assert len(trimmed) < len(sig)
        assert len(trimmed) >= SR  # At least the loud part

    def test_all_silence(self):
        sig = np.zeros(100)
        trimmed = trim_silence(sig, threshold_db=-40.0)
        assert len(trimmed) == 0


class TestDuration:
    def test_mono(self):
        sig = np.ones(SR * 2)
        assert duration(sig, sample_rate=SR) == pytest.approx(2.0)

    def test_stereo(self):
        sig = np.ones((SR * 3, 2))
        assert duration(sig, sample_rate=SR) == pytest.approx(3.0)
