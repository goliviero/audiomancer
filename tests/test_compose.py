"""Tests for audiomancer.compose module."""

import numpy as np
import pytest

from audiomancer.compose import fade_envelope, make_loopable, stitch, tremolo

SR = 44100
DUR = 10.0
N = int(SR * DUR)


class TestFadeEnvelope:
    def test_shape(self):
        env = fade_envelope([(0, 0.0), (DUR, 1.0)], DUR, sample_rate=SR)
        assert env.shape == (N,)

    def test_linear_interpolation(self):
        env = fade_envelope([(0, 0.0), (DUR, 1.0)], DUR, sample_rate=SR)
        # At the midpoint, value should be ~0.5
        mid = env[N // 2]
        assert mid == pytest.approx(0.5, abs=0.01)

    def test_start_value(self):
        env = fade_envelope([(0, 0.3), (DUR, 1.0)], DUR, sample_rate=SR)
        assert env[0] == pytest.approx(0.3, abs=0.01)

    def test_end_value(self):
        env = fade_envelope([(0, 0.0), (DUR, 0.8)], DUR, sample_rate=SR)
        assert env[-1] == pytest.approx(0.8, abs=0.01)

    def test_multiple_waypoints(self):
        # Flat plateau in the middle
        env = fade_envelope([(0, 0.0), (2, 1.0), (8, 1.0), (10, 0.0)], DUR, sample_rate=SR)
        mid_idx = int(SR * 5.0)
        assert env[mid_idx] == pytest.approx(1.0, abs=0.01)

    def test_single_waypoint_repeated(self):
        env = fade_envelope([(0, 0.5), (DUR, 0.5)], DUR, sample_rate=SR)
        assert np.allclose(env, 0.5, atol=0.01)

    def test_constant_value(self):
        env = fade_envelope([(0, 0.7), (5, 0.7), (DUR, 0.7)], DUR, sample_rate=SR)
        assert np.all(np.abs(env - 0.7) < 0.01)

    def test_filter_curve_range(self):
        # Used as filter cutoff — values must be sensible Hz ranges
        env = fade_envelope([(0, 800), (5, 3000), (DUR, 800)], DUR, sample_rate=SR)
        assert env.min() >= 799
        assert env.max() <= 3001


class TestTremolo:
    def test_mono_shape_preserved(self):
        signal = np.ones(N)
        result = tremolo(signal, rate_hz=0.15, depth=0.05, seed=42, sample_rate=SR)
        assert result.shape == (N,)

    def test_stereo_shape_preserved(self):
        signal = np.ones((N, 2))
        result = tremolo(signal, rate_hz=0.15, depth=0.05, seed=42, sample_rate=SR)
        assert result.shape == (N, 2)

    def test_depth_bounded(self):
        # With depth=0.05, output should be in [0.95, 1.05] range for input=1.0
        signal = np.ones(N)
        result = tremolo(signal, rate_hz=0.15, depth=0.05, seed=42, sample_rate=SR)
        assert result.min() >= 0.94
        assert result.max() <= 1.06

    def test_not_static(self):
        # Tremolo should change the signal
        signal = np.ones(N)
        result = tremolo(signal, rate_hz=0.15, depth=0.05, seed=42, sample_rate=SR)
        assert not np.allclose(result, 1.0)

    def test_reproducible_with_seed(self):
        signal = np.ones(N)
        a = tremolo(signal, seed=1, sample_rate=SR)
        b = tremolo(signal, seed=1, sample_rate=SR)
        np.testing.assert_array_equal(a, b)


class TestStitch:
    def test_single_section_passthrough(self):
        s = np.ones(N)
        result = stitch([s], crossfade_sec=1.0, sample_rate=SR)
        np.testing.assert_array_equal(result, s)

    def test_two_sections_length(self):
        a = np.ones(N)
        b = np.ones(N) * 2
        result = stitch([a, b], crossfade_sec=1.0, sample_rate=SR)
        # Result should be shorter than sum (crossfade overlaps)
        xf = int(SR * 1.0)
        assert result.shape[0] == 2 * N - xf

    def test_three_sections(self):
        sections = [np.ones(N), np.ones(N) * 2, np.ones(N) * 3]
        result = stitch(sections, crossfade_sec=1.0, sample_rate=SR)
        xf = int(SR * 1.0)
        assert result.shape[0] == 3 * N - 2 * xf

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            stitch([], sample_rate=SR)

    def test_stereo_sections(self):
        a = np.ones((N, 2))
        b = np.ones((N, 2)) * 2
        result = stitch([a, b], crossfade_sec=1.0, sample_rate=SR)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_no_abrupt_jump(self):
        # At the stitch point, there should be no large amplitude discontinuity
        a = np.ones(N)
        b = np.ones(N) * 0.0  # goes to silence
        result = stitch([a, b], crossfade_sec=2.0, sample_rate=SR)
        # Find the boundary region
        xf = int(SR * 2.0)
        boundary = result[N - xf: N]
        diffs = np.abs(np.diff(boundary))
        # Max sample-to-sample jump should be small
        assert diffs.max() < 0.01


class TestMakeLoopable:
    def test_shape_preserved(self):
        stem = np.ones(N)
        result = make_loopable(stem, crossfade_sec=1.0, sample_rate=SR)
        assert result.shape == stem.shape

    def test_stereo_shape_preserved(self):
        stem = np.ones((N, 2))
        result = make_loopable(stem, crossfade_sec=1.0, sample_rate=SR)
        assert result.shape == stem.shape

    def test_loop_point_smoothed(self):
        # Create a stem where end ≠ start to test smoothing
        stem = np.ones(N)
        stem[-100:] = 0.5  # End is quieter
        result = make_loopable(stem, crossfade_sec=1.0, sample_rate=SR)
        # Tail is blended: at the very end, majority is fade_out × tail + fade_in × head
        # Just verify no NaN/Inf
        assert np.all(np.isfinite(result))

    def test_not_modified_in_middle(self):
        # Middle of the stem should be unchanged
        stem = np.random.default_rng(42).standard_normal(N)
        result = make_loopable(stem, crossfade_sec=1.0, sample_rate=SR)
        xf = int(SR * 1.0)
        # Middle section should be identical
        np.testing.assert_array_equal(result[xf:-xf], stem[xf:-xf])
