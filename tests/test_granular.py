"""Tests for audiomancer.synth.granular function."""

import numpy as np
import pytest

from audiomancer.synth import granular, sine

SR = 44100


class TestGranular:
    def test_output_shape(self):
        source = sine(440, 2.0, sample_rate=SR)
        out = granular(source, 3.0, sample_rate=SR, seed=42)
        assert out.ndim == 1
        assert out.shape[0] == int(SR * 3.0)

    def test_deterministic_with_seed(self):
        source = sine(440, 1.0, sample_rate=SR)
        a = granular(source, 2.0, seed=123, sample_rate=SR)
        b = granular(source, 2.0, seed=123, sample_rate=SR)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        source = sine(440, 1.0, sample_rate=SR)
        a = granular(source, 2.0, seed=1, sample_rate=SR)
        b = granular(source, 2.0, seed=2, sample_rate=SR)
        assert not np.array_equal(a, b)

    def test_has_energy(self):
        source = sine(440, 1.0, sample_rate=SR)
        out = granular(source, 2.0, seed=42, sample_rate=SR)
        assert np.max(np.abs(out)) > 0.01

    def test_amplitude_control(self):
        source = sine(440, 1.0, sample_rate=SR)
        out = granular(source, 2.0, amplitude=0.5, seed=42, sample_rate=SR)
        assert np.max(np.abs(out)) == pytest.approx(0.5, abs=0.05)

    def test_no_pitch_spread(self):
        source = sine(440, 1.0, sample_rate=SR)
        out = granular(source, 1.0, pitch_spread=0.0, seed=42, sample_rate=SR)
        assert out.shape[0] == SR

    def test_stereo_source_handled(self):
        mono = sine(440, 1.0, sample_rate=SR)
        stereo = np.column_stack([mono, mono])
        out = granular(stereo, 1.0, seed=42, sample_rate=SR)
        # Should return mono (uses first channel)
        assert out.ndim == 1

    def test_short_source(self):
        # Very short source buffer
        source = sine(440, 0.1, sample_rate=SR)
        out = granular(source, 1.0, seed=42, sample_rate=SR)
        assert out.shape[0] == SR
