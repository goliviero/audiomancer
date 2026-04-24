"""Tests for audiomancer.modulation module."""

import numpy as np

from audiomancer.modulation import (
    apply_amplitude_mod,
    apply_filter_sweep,
    drift,
    evolving_lfo,
    lfo_sine,
    lfo_triangle,
    multi_lfo,
    random_walk,
)

SR = 44100
DUR = 2.0  # short duration for fast tests


class TestLfoSine:
    def test_shape(self):
        sig = lfo_sine(DUR, sample_rate=SR)
        assert sig.shape == (int(SR * DUR),)

    def test_range_default(self):
        sig = lfo_sine(DUR, depth=1.0, offset=0.0, sample_rate=SR)
        assert np.min(sig) >= -1.01
        assert np.max(sig) <= 1.01

    def test_offset_and_depth(self):
        sig = lfo_sine(DUR, depth=0.5, offset=2.0, sample_rate=SR)
        assert np.min(sig) >= 1.49
        assert np.max(sig) <= 2.51

    def test_phase(self):
        sig0 = lfo_sine(DUR, phase=0.0, sample_rate=SR)
        sig_pi = lfo_sine(DUR, phase=np.pi, sample_rate=SR)
        # Phase-shifted signals should differ
        assert not np.allclose(sig0, sig_pi)


class TestLfoTriangle:
    def test_shape(self):
        sig = lfo_triangle(DUR, sample_rate=SR)
        assert sig.shape == (int(SR * DUR),)

    def test_range(self):
        sig = lfo_triangle(DUR, depth=1.0, offset=0.0, sample_rate=SR)
        assert np.min(sig) >= -1.01
        assert np.max(sig) <= 1.01


class TestDrift:
    def test_shape(self):
        sig = drift(DUR, sample_rate=SR)
        assert sig.shape == (int(SR * DUR),)

    def test_clamped_range(self):
        sig = drift(DUR, depth=0.5, offset=1.0, seed=42, sample_rate=SR)
        assert np.min(sig) >= 0.49
        assert np.max(sig) <= 1.51

    def test_reproducibility(self):
        a = drift(DUR, seed=123, sample_rate=SR)
        b = drift(DUR, seed=123, sample_rate=SR)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = drift(DUR, seed=1, sample_rate=SR)
        b = drift(DUR, seed=2, sample_rate=SR)
        assert not np.allclose(a, b)


class TestEvolvingLfo:
    def test_shape(self):
        sig = evolving_lfo(DUR, seed=42, sample_rate=SR)
        assert sig.shape == (int(SR * DUR),)

    def test_clipped_range(self):
        sig = evolving_lfo(DUR, depth=1.0, offset=0.0, seed=42,
                           sample_rate=SR)
        assert np.min(sig) >= -1.51
        assert np.max(sig) <= 1.51

    def test_reproducibility(self):
        a = evolving_lfo(DUR, seed=7, sample_rate=SR)
        b = evolving_lfo(DUR, seed=7, sample_rate=SR)
        np.testing.assert_array_equal(a, b)


class TestApplyAmplitudeMod:
    def test_mono(self):
        signal = np.ones(SR)
        mod = np.full(SR, 0.5)
        result = apply_amplitude_mod(signal, mod)
        np.testing.assert_allclose(result, 0.5)

    def test_stereo(self):
        signal = np.ones((SR, 2))
        mod = np.full(SR, 2.0)
        result = apply_amplitude_mod(signal, mod)
        assert result.shape == (SR, 2)
        np.testing.assert_allclose(result, 2.0)

    def test_mod_shorter_than_signal(self):
        signal = np.ones(SR * 2)
        mod = np.ones(SR)
        result = apply_amplitude_mod(signal, mod)
        assert result.shape == signal.shape


class TestApplyFilterSweep:
    def test_output_shape_mono(self):
        signal = np.random.default_rng(0).standard_normal(SR)
        mod = np.full(SR, 1000.0)  # constant 1000 Hz cutoff
        result = apply_filter_sweep(signal, mod, sample_rate=SR)
        assert result.shape == signal.shape

    def test_output_shape_stereo(self):
        signal = np.random.default_rng(0).standard_normal((SR, 2))
        mod = np.full(SR, 1000.0)
        result = apply_filter_sweep(signal, mod, sample_rate=SR)
        assert result.shape == signal.shape


class TestMultiLfo:
    """Phase D1: multi-scale LFO stack."""

    def test_shape(self):
        env = multi_lfo(DUR, sample_rate=SR)
        assert env.shape == (int(SR * DUR),)

    def test_centered_around_one(self):
        env = multi_lfo(DUR, seed=42, sample_rate=SR)
        assert 0.8 < np.mean(env) < 1.2

    def test_deterministic_with_seed(self):
        a = multi_lfo(DUR, seed=42, sample_rate=SR)
        b = multi_lfo(DUR, seed=42, sample_rate=SR)
        assert np.allclose(a, b)

    def test_differs_with_seed(self):
        a = multi_lfo(DUR, seed=42, sample_rate=SR)
        b = multi_lfo(DUR, seed=99, sample_rate=SR)
        # Phases differ -> envelopes should NOT be identical
        assert not np.allclose(a, b)
        # Correlation should stay below strong-match threshold
        assert np.corrcoef(a, b)[0, 1] < 0.9

    def test_custom_layers(self):
        env = multi_lfo(DUR, layers=[(1.0, 0.1), (0.2, 0.05)],
                        seed=42, sample_rate=SR)
        assert env.shape == (int(SR * DUR),)
        assert 0.8 < np.mean(env) < 1.2


class TestRandomWalk:
    """Phase D2: Ornstein-Uhlenbeck bounded walk."""

    def test_shape(self):
        env = random_walk(DUR, sample_rate=SR)
        assert env.shape == (int(SR * DUR),)

    def test_mean_near_one(self):
        # Over long enough duration the mean should approach 1.0
        env = random_walk(10.0, sigma=0.05, tau=1.0, seed=42, sample_rate=SR)
        assert 0.9 < np.mean(env) < 1.1

    def test_bounded(self):
        # 3 sigma effective envelope should rarely exceed the stationary std
        # stationary_std = sigma * sqrt(tau/2), here ~= 0.05*sqrt(0.5) = 0.035
        env = random_walk(10.0, sigma=0.05, tau=1.0, seed=42, sample_rate=SR)
        assert np.max(env) < 1.5
        assert np.min(env) > 0.5

    def test_deterministic_with_seed(self):
        a = random_walk(DUR, seed=42, sample_rate=SR)
        b = random_walk(DUR, seed=42, sample_rate=SR)
        assert np.allclose(a, b)

    def test_differs_with_seed(self):
        a = random_walk(DUR, seed=42, sample_rate=SR)
        b = random_walk(DUR, seed=99, sample_rate=SR)
        assert not np.allclose(a, b)
