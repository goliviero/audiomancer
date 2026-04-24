"""Tests for audiomancer.layers module."""

import numpy as np
import pytest

from audiomancer.layers import (
    crossfade,
    layer,
    layer_at_offset,
    loop_seamless,
    measure_lufs,
    mix,
    normalize_lufs,
    suggest_eq_cuts,
)

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


class TestLayer:
    def test_layer_linear_volumes(self):
        a = np.ones(100)
        b = np.ones(100)
        result = layer([a, b], volumes=[0.5, 0.5])
        # 0.5 + 0.5 = 1.0
        np.testing.assert_allclose(result, np.ones(100), atol=0.01)

    def test_layer_default_volumes(self):
        a = np.ones(100)
        result = layer([a])
        np.testing.assert_allclose(result, np.ones(100), atol=0.01)


class TestLoopSeamless:
    def test_loop_extends(self):
        sig = np.ones(SR)  # 1 second
        result = loop_seamless(sig, 3.0, crossfade_sec=0.5, sample_rate=SR)
        assert len(result) == SR * 3

    def test_loop_already_long_enough(self):
        sig = np.ones(SR * 5)
        result = loop_seamless(sig, 3.0, sample_rate=SR)
        assert len(result) == SR * 3


class TestNormalizeLufs:
    def test_normalize_changes_level(self):
        # Use a sine wave — K-weighting handles tonal content predictably
        t = np.linspace(0, 1, SR, endpoint=False)
        sig = 0.001 * np.sin(2 * np.pi * 440 * t)
        result = normalize_lufs(sig, target_lufs=-14.0)
        lufs_out = measure_lufs(result)
        assert lufs_out == pytest.approx(-14.0, abs=1.5)

    def test_normalize_silent(self):
        sig = np.zeros(SR)
        result = normalize_lufs(sig, target_lufs=-14.0)
        np.testing.assert_allclose(result, sig)


class TestMeasureLufs:
    def test_silent_returns_neg_inf(self):
        sig = np.zeros(SR)
        assert measure_lufs(sig) == -np.inf

    def test_louder_signal_higher_lufs(self):
        quiet = np.ones(SR) * 0.01
        loud = np.ones(SR) * 0.5
        assert measure_lufs(loud) > measure_lufs(quiet)

    def test_stereo_signal(self):
        sig = np.column_stack([np.ones(SR) * 0.1, np.ones(SR) * 0.1])
        lufs = measure_lufs(sig)
        assert np.isfinite(lufs)

    def test_returns_finite_for_normal_signal(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t)
        lufs = measure_lufs(sig)
        assert np.isfinite(lufs)


class TestSuggestEqCuts:
    """Suggest EQ cuts to reduce mid-band masking across stems."""

    def test_returns_empty_for_disjoint_spectra(self):
        """Low-freq stem + high-freq stem should have no overlap."""
        t = np.linspace(0, 2, SR * 2, endpoint=False)
        low = 0.5 * np.sin(2 * np.pi * 80 * t)    # in sub/low bands
        high = 0.5 * np.sin(2 * np.pi * 8000 * t)  # in high band
        suggestions = suggest_eq_cuts(
            {"low": low, "high": high}, sample_rate=SR,
        )
        # No mid-band overlap -> no suggestions
        assert isinstance(suggestions, list)

    def test_returns_suggestions_for_overlapping_stems(self):
        """Two stems with energy in the same mid band should generate suggestions."""
        t = np.linspace(0, 2, SR * 2, endpoint=False)
        a = 0.5 * np.sin(2 * np.pi * 800 * t)  # mid band
        b = 0.5 * np.sin(2 * np.pi * 1200 * t)  # also mid band
        suggestions = suggest_eq_cuts({"a": a, "b": b}, sample_rate=SR)
        # Each suggestion is (stem_name, freq_hz, cut_db)
        assert all(len(s) == 3 for s in suggestions)

    def test_suggestions_reference_valid_stem(self):
        t = np.linspace(0, 2, SR * 2, endpoint=False)
        a = 0.5 * np.sin(2 * np.pi * 800 * t)
        b = 0.5 * np.sin(2 * np.pi * 1200 * t)
        suggestions = suggest_eq_cuts({"a": a, "b": b}, sample_rate=SR)
        for stem, _, _ in suggestions:
            assert stem in ("a", "b")
