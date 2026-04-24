"""Tests for audiomancer.utils module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiomancer.utils import (
    concat,
    duration,
    export_wav,
    fade_in,
    fade_out,
    load_audio,
    mono_to_stereo,
    normalize,
    pad_to_length,
    silence,
    stereo_to_mono,
    trim_silence,
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


class TestLoadAudioResample:
    """Phase A3: target_sr triggers polyphase resample."""

    def test_resample_mono_44100_to_48000(self):
        """A 1s @ 44.1kHz file should return 48000 samples at target_sr=48000."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_441.wav"
            src_sr = 44100
            src_dur = 1.0
            sig = np.sin(2 * np.pi * 440 * np.linspace(0, src_dur, src_sr, endpoint=False))
            export_wav(sig, path, sample_rate=src_sr)

            loaded, sr = load_audio(path, target_sr=48000)
            assert sr == 48000
            # 44100 in -> 48000 out; allow +/-1 sample tolerance on polyphase
            assert abs(len(loaded) - 48000) <= 1

    def test_no_resample_when_target_matches(self):
        """target_sr == native SR must be a no-op."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_native.wav"
            src_sr = 48000
            sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, src_sr, endpoint=False))
            export_wav(sig, path, sample_rate=src_sr)

            loaded, sr = load_audio(path, target_sr=48000)
            assert sr == 48000
            assert len(loaded) == src_sr

    def test_default_no_resample(self):
        """target_sr=None keeps legacy behavior (native SR returned)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_legacy.wav"
            src_sr = 22050
            sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, src_sr, endpoint=False))
            export_wav(sig, path, sample_rate=src_sr)

            loaded, sr = load_audio(path)
            assert sr == 22050
            assert len(loaded) == src_sr

    def test_resample_stereo(self):
        """Stereo files must resample each channel independently."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_stereo.wav"
            src_sr = 44100
            t = np.linspace(0, 1.0, src_sr, endpoint=False)
            left = np.sin(2 * np.pi * 440 * t)
            right = np.sin(2 * np.pi * 660 * t)
            stereo = np.column_stack([left, right])
            export_wav(stereo, path, sample_rate=src_sr)

            loaded, sr = load_audio(path, target_sr=48000)
            assert sr == 48000
            assert loaded.ndim == 2
            assert abs(loaded.shape[0] - 48000) <= 1
            assert loaded.shape[1] == 2
