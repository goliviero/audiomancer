"""Tests for audiomancer.field module."""

import numpy as np
import pytest

from audiomancer.field import clean, noise_gate, process_field

SR = 44100


class TestClean:
    def test_removes_dc_offset(self):
        sig = np.ones(SR) * 0.5  # DC offset
        result = clean(sig, sample_rate=SR)
        assert abs(np.mean(result)) < 0.01

    def test_preserves_shape(self):
        sig = np.random.default_rng().random(SR) - 0.5
        result = clean(sig, sample_rate=SR)
        assert result.shape == sig.shape


class TestNoiseGate:
    def test_quiet_signal_gated(self):
        sig = np.ones(SR) * 0.001  # Very quiet (-60 dB)
        result = noise_gate(sig, threshold_db=-40.0, sample_rate=SR)
        assert np.max(np.abs(result)) < np.max(np.abs(sig))

    def test_loud_signal_passes(self):
        sig = np.ones(SR) * 0.5  # Loud
        result = noise_gate(sig, threshold_db=-40.0, sample_rate=SR)
        assert np.max(np.abs(result)) > 0


class TestProcessField:
    def test_runs_without_error(self):
        rng = np.random.default_rng(42)
        sig = rng.random(SR) - 0.5
        result = process_field(sig, sample_rate=SR)
        assert len(result) > 0
        assert not np.any(np.isnan(result))
