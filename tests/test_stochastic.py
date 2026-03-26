"""Tests for audiomancer.stochastic module."""

from unittest.mock import patch

import numpy as np

from audiomancer.stochastic import DEFAULT_EVENTS, _place_events, scatter_events

SR = 44100


def _fake_texture(name, duration_sec, seed=None, sample_rate=SR, **kw):
    """Fast mock texture — sine burst instead of real texture generation."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    return 0.3 * np.sin(2 * np.pi * 440 * t)


class TestScatterEvents:
    @patch("audiomancer.stochastic._gen_texture", side_effect=_fake_texture)
    def test_output_shape(self, mock_tex):
        result = scatter_events(10.0, density=0.5, seed=42, sample_rate=SR)
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert result.shape[0] == int(SR * 10.0)

    @patch("audiomancer.stochastic._gen_texture", side_effect=_fake_texture)
    def test_deterministic_with_seed(self, mock_tex):
        a = scatter_events(10.0, seed=123, sample_rate=SR)
        b = scatter_events(10.0, seed=123, sample_rate=SR)
        np.testing.assert_array_equal(a, b)

    @patch("audiomancer.stochastic._gen_texture", side_effect=_fake_texture)
    def test_different_seeds_differ(self, mock_tex):
        a = scatter_events(10.0, seed=1, sample_rate=SR)
        b = scatter_events(10.0, seed=2, sample_rate=SR)
        assert not np.array_equal(a, b)

    @patch("audiomancer.stochastic._gen_texture", side_effect=_fake_texture)
    def test_has_energy(self, mock_tex):
        result = scatter_events(30.0, density=1.0, seed=42, sample_rate=SR)
        assert np.max(np.abs(result)) > 0.001

    @patch("audiomancer.stochastic._gen_texture", side_effect=_fake_texture)
    def test_custom_events(self, mock_tex):
        events = [{"texture": "crystal_shimmer", "duration": 2.0,
                    "count": 1, "volume_db": -6.0, "min_gap_sec": 5.0}]
        result = scatter_events(10.0, events=events, seed=42, sample_rate=SR)
        assert result.shape[0] == int(SR * 10.0)


class TestPlaceEvents:
    def test_respects_min_gap(self):
        rng = np.random.default_rng(42)
        positions = _place_events(60.0, 5.0, 4, 10.0, rng)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert abs(positions[i] - positions[j]) >= 10.0

    def test_returns_sorted(self):
        rng = np.random.default_rng(42)
        positions = _place_events(120.0, 3.0, 5, 5.0, rng)
        assert positions == sorted(positions)

    def test_short_duration_fallback(self):
        rng = np.random.default_rng(42)
        positions = _place_events(3.0, 5.0, 3, 10.0, rng)
        assert positions == [0.0]


class TestDefaultEvents:
    def test_default_events_is_list(self):
        assert isinstance(DEFAULT_EVENTS, list)
        assert len(DEFAULT_EVENTS) >= 2

    def test_default_events_have_required_keys(self):
        for evt in DEFAULT_EVENTS:
            assert "texture" in evt
            assert "duration" in evt
