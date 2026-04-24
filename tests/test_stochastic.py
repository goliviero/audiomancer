"""Tests for audiomancer.stochastic module."""

from unittest.mock import patch

import numpy as np

from audiomancer.stochastic import (
    DEFAULT_EVENTS,
    _place_events,
    micro_events,
    micro_silence_env,
    scatter_events,
)

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


class TestMicroEvents:
    """Phase D4: typed micro-event scatter."""

    def test_harmonic_bloom_shape_and_count(self):
        dur = 300.0  # 5 min
        chord = [264.0, 297.0, 396.0]
        result = micro_events(
            dur,
            event_specs=[{"type": "harmonic_bloom", "rate_per_min": 1.0,
                          "volume_db": -24.0, "duration_range": (3.0, 5.0)}],
            chord_freqs=chord, seed=42, sample_rate=SR,
        )
        assert result.shape == (int(dur * SR), 2)
        # ~5 events (1/min over 5 min). Signal should not be silent.
        assert np.max(np.abs(result)) > 0.0

    def test_deterministic_with_seed(self):
        chord = [264.0]
        a = micro_events(
            60.0,
            event_specs=[{"type": "harmonic_bloom", "rate_per_min": 2.0,
                          "volume_db": -24.0}],
            chord_freqs=chord, seed=42, sample_rate=SR,
        )
        b = micro_events(
            60.0,
            event_specs=[{"type": "harmonic_bloom", "rate_per_min": 2.0,
                          "volume_db": -24.0}],
            chord_freqs=chord, seed=42, sample_rate=SR,
        )
        assert np.allclose(a, b)

    def test_grain_burst_needs_source(self):
        import pytest
        src = np.random.default_rng(0).standard_normal(SR * 2)
        # With source: OK
        result = micro_events(
            30.0,
            event_specs=[{"type": "grain_burst", "rate_per_min": 2.0,
                          "volume_db": -28.0}],
            source=src, seed=42, sample_rate=SR,
        )
        assert result.shape == (int(30 * SR), 2)
        # Without source: raises
        with pytest.raises(ValueError, match="grain_burst needs source"):
            micro_events(
                30.0,
                event_specs=[{"type": "grain_burst", "rate_per_min": 2.0}],
                seed=42, sample_rate=SR,
            )

    def test_overtone_whisper(self):
        result = micro_events(
            120.0,  # need >=60s for rate_per_min=1 to yield >=1 event
            event_specs=[{"type": "overtone_whisper", "rate_per_min": 2.0,
                          "volume_db": -32.0}],
            chord_freqs=[264.0], seed=42, sample_rate=SR,
        )
        assert result.shape == (int(120 * SR), 2)
        # Whisper is quiet but non-zero
        assert np.max(np.abs(result)) > 0.0
        assert np.max(np.abs(result)) < 0.1  # -32 dB = ~0.025


class TestMicroSilenceEnv:
    """Phase D4 bis: subtractive envelope for mix ducks."""

    def test_returns_envelope_in_bounds(self):
        env = micro_silence_env(60.0, rate_per_min=2.0,
                                duck_db=-12.0, seed=42, sample_rate=SR)
        assert env.shape == (int(60 * SR), 2)
        assert np.min(env) < 1.0  # some ducking occurred
        assert np.min(env) >= 0.2  # -12 dB ~= 0.25
        assert np.max(env) == 1.0  # unchanged outside ducks

    def test_zero_rate_no_duck(self):
        env = micro_silence_env(30.0, rate_per_min=0.0, sample_rate=SR)
        assert np.all(env == 1.0)
