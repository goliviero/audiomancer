"""Tests for audiomancer.sidechain — envelope follower + ducking."""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.sidechain import envelope_follower, sidechain_duck
from audiomancer.synth import sine

SR = SAMPLE_RATE


class TestEnvelopeFollower:
    def test_shape_mono(self):
        sig = sine(440.0, 0.5, sample_rate=SR)
        env = envelope_follower(sig, sample_rate=SR)
        assert env.shape == sig.shape

    def test_shape_stereo(self):
        mono = sine(440.0, 0.5, sample_rate=SR)
        sig = np.column_stack([mono, mono])
        env = envelope_follower(sig, sample_rate=SR)
        # Envelope is always mono
        assert env.ndim == 1
        assert env.shape[0] == sig.shape[0]

    def test_envelope_non_negative(self):
        sig = sine(440.0, 0.5, sample_rate=SR)
        env = envelope_follower(sig, sample_rate=SR)
        assert np.all(env >= 0)

    def test_envelope_bounded_by_peak(self):
        sig = sine(440.0, 0.5, amplitude=0.8, sample_rate=SR)
        env = envelope_follower(sig, sample_rate=SR)
        # Envelope should not exceed source peak (by more than epsilon)
        assert np.max(env) <= 0.85


class TestSidechainDuck:
    def test_shape(self):
        target_mono = sine(220.0, 0.5, amplitude=0.5, sample_rate=SR)
        trigger = sine(880.0, 0.5, amplitude=0.5, sample_rate=SR)
        out = sidechain_duck(target_mono, trigger, sample_rate=SR)
        assert out.shape == target_mono.shape

    def test_stereo_target(self):
        target_mono = sine(220.0, 0.5, amplitude=0.5, sample_rate=SR)
        target = np.column_stack([target_mono, target_mono])
        trigger = sine(880.0, 0.5, amplitude=0.7, sample_rate=SR)
        out = sidechain_duck(target, trigger, sample_rate=SR)
        assert out.shape == target.shape

    def test_duck_reduces_gain(self):
        """When trigger is loud, target should be reduced."""
        target = np.full(SR, 0.5)  # constant DC-ish at 0.5
        trigger = np.full(SR, 0.5)  # constant above threshold
        out = sidechain_duck(target, trigger, amount_db=-12.0,
                             threshold_db=-24.0, attack_ms=5.0,
                             release_ms=50.0, sample_rate=SR)
        # Mid-signal after attack has settled
        mid = out[SR // 2]
        assert abs(mid) < 0.5  # reduced from source 0.5

    def test_silent_trigger_no_duck(self):
        target = sine(220.0, 0.5, amplitude=0.5, sample_rate=SR)
        trigger = np.zeros_like(target)
        out = sidechain_duck(target, trigger, sample_rate=SR)
        # With silent trigger, ducking shouldn't engage -> output ~= target
        assert np.allclose(out, target, atol=0.01)
