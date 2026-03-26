"""Tests for audiomancer.envelope — amplitude shaping."""

import numpy as np
import pytest

from audiomancer import SAMPLE_RATE
from audiomancer.envelope import (
    adsr, adsr_exp, ar, segments, breathing, swell, gate_pattern,
)


SR = SAMPLE_RATE
DUR = 1.0


# ---------------------------------------------------------------------------
# ADSR
# ---------------------------------------------------------------------------

class TestADSR:
    def test_correct_length(self):
        env = adsr(DUR, sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_starts_at_zero(self):
        env = adsr(DUR, attack=0.1, sample_rate=SR)
        assert env[0] == pytest.approx(0.0, abs=0.01)

    def test_peak_at_one(self):
        env = adsr(DUR, attack=0.1, decay=0.1, sustain=0.5, sample_rate=SR)
        assert np.max(env) == pytest.approx(1.0, abs=0.02)

    def test_sustain_level(self):
        env = adsr(2.0, attack=0.1, decay=0.1, sustain=0.6, release=0.1,
                   sample_rate=SR)
        # Sample in the middle of the sustain phase
        mid = int(SR * 1.0)
        assert env[mid] == pytest.approx(0.6, abs=0.02)

    def test_ends_near_zero(self):
        env = adsr(DUR, attack=0.1, decay=0.1, sustain=0.5, release=0.3,
                   sample_rate=SR)
        assert env[-1] == pytest.approx(0.0, abs=0.02)

    def test_values_in_range(self):
        env = adsr(DUR, sample_rate=SR)
        assert np.min(env) >= -0.01
        assert np.max(env) <= 1.01


# ---------------------------------------------------------------------------
# Exponential ADSR
# ---------------------------------------------------------------------------

class TestADSRExp:
    def test_correct_length(self):
        env = adsr_exp(DUR, sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_peak_at_one(self):
        env = adsr_exp(DUR, attack=0.2, decay=0.1, sustain=0.5, sample_rate=SR)
        assert np.max(env) == pytest.approx(1.0, abs=0.05)

    def test_higher_curve_different_shape(self):
        linear = adsr_exp(DUR, curve=1.0, sample_rate=SR)
        curved = adsr_exp(DUR, curve=5.0, sample_rate=SR)
        # They should be different (different curve shapes)
        assert not np.allclose(linear, curved, atol=0.01)

    def test_values_in_range(self):
        env = adsr_exp(DUR, sample_rate=SR)
        assert np.min(env) >= -0.01
        assert np.max(env) <= 1.01


# ---------------------------------------------------------------------------
# AR envelope
# ---------------------------------------------------------------------------

class TestAR:
    def test_correct_length(self):
        env = ar(DUR, sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_symmetric_peak_in_middle(self):
        env = ar(DUR, attack=0.5, sample_rate=SR)
        peak_idx = np.argmax(env)
        mid = len(env) // 2
        assert abs(peak_idx - mid) < SR * 0.05  # Within 50ms of center

    def test_starts_and_ends_near_zero(self):
        env = ar(DUR, sample_rate=SR)
        assert env[0] == pytest.approx(0.0, abs=0.02)
        assert env[-1] == pytest.approx(0.0, abs=0.02)

    def test_peak_is_one(self):
        env = ar(DUR, sample_rate=SR)
        assert np.max(env) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Multi-segment envelope
# ---------------------------------------------------------------------------

class TestSegments:
    def test_correct_length(self):
        pts = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
        env = segments(pts, DUR, sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_starts_at_first_value(self):
        pts = [(0.0, 0.3), (0.5, 1.0), (1.0, 0.0)]
        env = segments(pts, DUR, sample_rate=SR)
        assert env[0] == pytest.approx(0.3, abs=0.02)

    def test_reaches_peak(self):
        pts = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
        env = segments(pts, DUR, sample_rate=SR)
        assert np.max(env) >= 0.95

    def test_holds_last_value(self):
        pts = [(0.0, 0.0), (0.3, 0.8)]
        env = segments(pts, 1.0, sample_rate=SR)
        # After 0.3s, should hold at 0.8
        assert env[-1] == pytest.approx(0.8, abs=0.02)

    def test_curved_different_from_linear(self):
        pts = [(0.0, 0.0), (1.0, 1.0)]
        linear = segments(pts, DUR, curve=1.0, sample_rate=SR)
        curved = segments(pts, DUR, curve=3.0, sample_rate=SR)
        assert not np.allclose(linear, curved, atol=0.01)


# ---------------------------------------------------------------------------
# Breathing
# ---------------------------------------------------------------------------

class TestBreathing:
    def test_correct_length(self):
        env = breathing(DUR, sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_values_in_range(self):
        env = breathing(DUR, floor=0.5, depth=0.3, sample_rate=SR)
        assert np.min(env) >= 0.49
        assert np.max(env) <= 0.81

    def test_oscillates(self):
        env = breathing(5.0, breath_rate=1.0, depth=0.4, sample_rate=SR)
        # Should have multiple peaks and valleys
        diff = np.diff(np.sign(np.diff(env)))
        n_extrema = np.sum(np.abs(diff) > 0)
        assert n_extrema > 5  # At least a few oscillation cycles


# ---------------------------------------------------------------------------
# Swell
# ---------------------------------------------------------------------------

class TestSwell:
    def test_correct_length(self):
        env = swell(DUR, sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_peak_is_one(self):
        env = swell(DUR, sample_rate=SR)
        assert np.max(env) == pytest.approx(1.0, abs=0.05)

    def test_starts_near_zero(self):
        env = swell(DUR, peak_time=0.5, sample_rate=SR)
        assert env[0] == pytest.approx(0.0, abs=0.02)

    def test_ends_near_zero(self):
        env = swell(DUR, peak_time=0.5, sample_rate=SR)
        assert env[-1] == pytest.approx(0.0, abs=0.02)

    def test_peak_position(self):
        env = swell(DUR, peak_time=0.7, sample_rate=SR)
        peak_idx = np.argmax(env)
        expected = int(SR * 0.7)
        assert abs(peak_idx - expected) < SR * 0.1


# ---------------------------------------------------------------------------
# Gate pattern
# ---------------------------------------------------------------------------

class TestGatePattern:
    def test_correct_length(self):
        env = gate_pattern(DUR, [1.0, 0.0], sample_rate=SR)
        assert len(env) == int(SR * DUR)

    def test_values_in_range(self):
        env = gate_pattern(DUR, [1.0, 0.0, 0.5], sample_rate=SR)
        assert np.min(env) >= -0.01
        assert np.max(env) <= 1.01

    def test_pattern_repeats(self):
        env = gate_pattern(2.0, [1.0, 0.0], step_sec=0.5,
                           smoothing_ms=0.0, sample_rate=SR)
        # Should alternate between ~1 and ~0
        mid_on = int(SR * 0.25)   # Middle of first ON step
        mid_off = int(SR * 0.75)  # Middle of first OFF step
        assert env[mid_on] > 0.8
        assert env[mid_off] < 0.2

    def test_smoothing_removes_clicks(self):
        env = gate_pattern(DUR, [1.0, 0.0], step_sec=0.1,
                           smoothing_ms=10.0, sample_rate=SR)
        # Max sample-to-sample change should be smaller than a hard step
        max_diff = np.max(np.abs(np.diff(env)))
        assert max_diff < 0.5  # Smoothed, not a hard 0→1 jump
