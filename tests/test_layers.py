"""Tests for audiomancer.layers module."""

import numpy as np
import pytest

from audiomancer.layers import mix, layer_at_offset, crossfade

SR = 44100


class TestMix:
    def test_mix_two_signals(self):
        a = np.ones(100)
        b = np.ones(100)
        result = mix([a, b])
        np.testing.assert_allclose(result, 2.0 * np.ones(100))

    def test_mix_with_volume(self):
        a = np.ones(100)
        result = mix([a], volumes_db=[-6.0])
        expected_gain = 10 ** (-6.0 / 20)
        np.testing.assert_allclose(result, expected_gain * np.ones(100), atol=0.001)

    def test_mix_different_lengths(self):
        a = np.ones(100)
        b = np.ones(50)
        result = mix([a, b])
        assert len(result) == 100

    def test_mix_empty(self):
        result = mix([])
        assert len(result) == 0


class TestLayerAtOffset:
    def test_basic_layer(self):
        base = np.ones(SR)
        overlay = np.ones(int(SR * 0.5))
        result = layer_at_offset(base, overlay, 0.5, sample_rate=SR)
        assert result.shape[0] == SR
        # First half: base only (1.0), second half: base + overlay (2.0)
        mid = int(SR * 0.5)
        assert result[0] == pytest.approx(1.0)
        assert result[mid + 100] == pytest.approx(2.0)

    def test_layer_extends(self):
        base = np.ones(SR)
        overlay = np.ones(SR)
        result = layer_at_offset(base, overlay, 0.5, sample_rate=SR)
        assert result.shape[0] == int(SR * 1.5)


class TestCrossfade:
    def test_crossfade_length(self):
        a = np.ones(SR)
        b = np.zeros(SR)
        result = crossfade(a, b, 0.5, sample_rate=SR)
        expected_len = SR + SR - int(SR * 0.5)
        assert len(result) == expected_len

    def test_crossfade_middle(self):
        a = np.ones(SR)
        b = np.zeros(SR)
        result = crossfade(a, b, 0.5, sample_rate=SR)
        # Middle of crossfade should be ~0.5
        xf_start = SR - int(SR * 0.5)
        mid = xf_start + int(SR * 0.25)
        assert result[mid] == pytest.approx(0.5, abs=0.05)
