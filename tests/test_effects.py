"""Tests for audiomancer.effects module."""

import numpy as np
import pytest

from audiomancer.effects import lowpass, highpass

SR = 44100


class TestFilters:
    def test_lowpass_reduces_highs(self):
        # Generate a mix of low and high frequency
        t = np.linspace(0, 1, SR, endpoint=False)
        low = np.sin(2 * np.pi * 100 * t)
        high = np.sin(2 * np.pi * 8000 * t)
        mixed = low + high

        filtered = lowpass(mixed, cutoff_hz=500, sample_rate=SR)

        # High frequency energy should be reduced
        fft_orig = np.abs(np.fft.rfft(mixed))
        fft_filt = np.abs(np.fft.rfft(filtered))
        high_bin = int(8000 * len(t) / SR)
        assert fft_filt[high_bin] < fft_orig[high_bin] * 0.1

    def test_highpass_reduces_lows(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        low = np.sin(2 * np.pi * 50 * t)
        high = np.sin(2 * np.pi * 5000 * t)
        mixed = low + high

        filtered = highpass(mixed, cutoff_hz=1000, sample_rate=SR)

        fft_orig = np.abs(np.fft.rfft(mixed))
        fft_filt = np.abs(np.fft.rfft(filtered))
        low_bin = int(50 * len(t) / SR)
        assert fft_filt[low_bin] < fft_orig[low_bin] * 0.1

    def test_lowpass_stereo(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        mono = np.sin(2 * np.pi * 100 * t)
        stereo = np.column_stack([mono, mono])
        filtered = lowpass(stereo, cutoff_hz=500, sample_rate=SR)
        assert filtered.shape == (SR, 2)
