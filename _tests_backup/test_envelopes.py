"""Tests for fractal.envelopes — amplitude shaping."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.envelopes import (
    ADSR,
    AutomationCurve,
    ExponentialFade,
    FadeInOut,
    Gate,
    SmoothFade,
    Swell,
    Tremolo,
)


class TestFadeInOut:
    def test_fade_in(self):
        env = FadeInOut(fade_in=1.0)
        curve = env.generate(SAMPLE_RATE * 2)
        assert curve[0] == 0.0
        assert curve[SAMPLE_RATE - 1] == pytest.approx(1.0, abs=1e-4)
        assert curve[-1] == 1.0

    def test_fade_out(self):
        env = FadeInOut(fade_out=1.0)
        curve = env.generate(SAMPLE_RATE * 2)
        assert curve[0] == 1.0
        assert curve[-1] == pytest.approx(0.0, abs=1e-4)

    def test_both_fades(self):
        env = FadeInOut(fade_in=0.5, fade_out=0.5)
        curve = env.generate(SAMPLE_RATE * 2)
        assert curve[0] == 0.0
        assert curve[SAMPLE_RATE] == 1.0  # middle
        assert curve[-1] == pytest.approx(0.0, abs=1e-4)

    def test_no_fades(self):
        env = FadeInOut()
        curve = env.generate(1000)
        np.testing.assert_array_equal(curve, np.ones(1000))

    def test_apply_mono(self):
        sig = np.ones(SAMPLE_RATE)
        env = FadeInOut(fade_in=0.5)
        shaped = env.apply(sig)
        assert shaped[0] == 0.0
        assert shaped[-1] == 1.0

    def test_apply_stereo(self):
        sig = np.ones((SAMPLE_RATE, 2))
        env = FadeInOut(fade_in=0.5, fade_out=0.5)
        shaped = env.apply(sig)
        assert shaped.shape == (SAMPLE_RATE, 2)
        assert shaped[0, 0] == 0.0
        assert shaped[0, 1] == 0.0


class TestADSR:
    def test_basic_shape(self):
        env = ADSR(attack=0.1, decay=0.1, sustain=0.7, release=0.2)
        curve = env.generate(SAMPLE_RATE)
        # Starts at 0
        assert curve[0] == 0.0
        # Peak near 1.0 at end of attack
        attack_end = int(SAMPLE_RATE * 0.1)
        assert curve[attack_end - 1] == pytest.approx(1.0, abs=0.01)
        # Sustain level in the middle
        mid = SAMPLE_RATE // 2
        assert curve[mid] == pytest.approx(0.7, abs=0.01)
        # Ends near 0
        assert curve[-1] == pytest.approx(0.0, abs=0.01)

    def test_sustain_clamp(self):
        env = ADSR(sustain=1.5)
        assert env.sustain == 1.0
        env2 = ADSR(sustain=-0.5)
        assert env2.sustain == 0.0

    def test_short_signal_scales(self):
        """ADSR should not crash on signals shorter than A+D+R."""
        env = ADSR(attack=1.0, decay=1.0, sustain=0.5, release=1.0)
        curve = env.generate(SAMPLE_RATE // 2)  # 0.5s < 3.0s total
        assert len(curve) == SAMPLE_RATE // 2
        assert curve[0] == 0.0

    def test_apply(self):
        sig = np.ones(SAMPLE_RATE)
        env = ADSR(attack=0.1, decay=0.05, sustain=0.8, release=0.1)
        shaped = env.apply(sig)
        assert shaped[0] == 0.0
        assert len(shaped) == SAMPLE_RATE


class TestSmoothFade:
    def test_cosine_shape(self):
        env = SmoothFade(fade_in=1.0)
        curve = env.generate(SAMPLE_RATE * 2)
        assert curve[0] == 0.0
        # At midpoint of fade, cosine gives 0.5
        mid = SAMPLE_RATE // 2
        assert curve[mid] == pytest.approx(0.5, abs=0.01)

    def test_fade_out(self):
        env = SmoothFade(fade_out=1.0)
        curve = env.generate(SAMPLE_RATE * 2)
        assert curve[-1] == pytest.approx(0.0, abs=1e-4)
        assert curve[0] == 1.0


class TestExponentialFade:
    def test_starts_at_zero(self):
        env = ExponentialFade(fade_in=1.0)
        curve = env.generate(SAMPLE_RATE * 2)
        assert curve[0] == pytest.approx(0.0, abs=1e-4)
        assert curve[SAMPLE_RATE - 1] == pytest.approx(1.0, abs=1e-3)

    def test_curve_is_convex(self):
        """Exponential fade-in should be below the linear diagonal."""
        env = ExponentialFade(fade_in=1.0, steepness=5.0)
        curve = env.generate(SAMPLE_RATE)
        linear = np.linspace(0, 1, SAMPLE_RATE)
        # Exponential starts slow, so midpoint should be below linear
        mid = SAMPLE_RATE // 2
        assert curve[mid] < linear[mid]


class TestSwell:
    def test_rise_then_hold(self):
        env = Swell(rise_time=2.0, peak=0.8)
        curve = env.generate(SAMPLE_RATE * 5)
        assert curve[0] == pytest.approx(0.0, abs=1e-3)
        # After rise, should be at peak
        assert curve[SAMPLE_RATE * 3] == pytest.approx(0.8, abs=1e-3)

    def test_linear_rise(self):
        env = Swell(rise_time=1.0, peak=1.0, curve_type="linear")
        curve = env.generate(SAMPLE_RATE * 2)
        mid = SAMPLE_RATE // 2
        assert curve[mid] == pytest.approx(0.5, abs=0.01)


class TestGate:
    def test_basic_pattern(self):
        env = Gate(on_time=0.1, off_time=0.1, smooth_ms=0.0)
        curve = env.generate(SAMPLE_RATE)
        on_samples = int(SAMPLE_RATE * 0.1)
        # First "on" block should be 1.0
        assert curve[0] == 1.0
        assert curve[on_samples - 1] == 1.0
        # First "off" block should be 0.0
        assert curve[on_samples] == 0.0

    def test_has_transitions_with_smooth(self):
        env = Gate(on_time=0.1, off_time=0.1, smooth_ms=10.0)
        curve = env.generate(SAMPLE_RATE)
        # Smooth gate should have non-binary values at transitions
        unique = np.unique(curve)
        assert len(unique) > 2


class TestTremolo:
    def test_range(self):
        env = Tremolo(rate=5.0, depth=0.5)
        curve = env.generate(SAMPLE_RATE)
        assert np.min(curve) >= 0.5 - 1e-6
        assert np.max(curve) <= 1.0 + 1e-6

    def test_full_depth(self):
        env = Tremolo(rate=5.0, depth=1.0)
        curve = env.generate(SAMPLE_RATE)
        assert np.min(curve) >= -1e-6  # can go to 0
        assert np.max(curve) <= 1.0 + 1e-6

    def test_no_depth(self):
        env = Tremolo(rate=5.0, depth=0.0)
        curve = env.generate(SAMPLE_RATE)
        np.testing.assert_allclose(curve, 1.0, atol=1e-10)

    def test_triangle_shape(self):
        env = Tremolo(rate=2.0, depth=0.5, shape="triangle")
        curve = env.generate(SAMPLE_RATE)
        assert len(curve) == SAMPLE_RATE


class TestAutomationCurve:
    def test_linear_ramp(self):
        env = AutomationCurve([(0.0, 0.0), (1.0, 1.0)])
        curve = env.generate(SAMPLE_RATE)
        assert curve[0] == pytest.approx(0.0, abs=1e-4)
        mid = SAMPLE_RATE // 2
        assert curve[mid] == pytest.approx(0.5, abs=0.01)

    def test_multiple_points(self):
        env = AutomationCurve([(0.0, 0.0), (0.5, 1.0), (1.0, 0.3)])
        curve = env.generate(SAMPLE_RATE)
        # Peak at 0.5s
        peak_idx = SAMPLE_RATE // 2
        assert curve[peak_idx] == pytest.approx(1.0, abs=0.01)
        # End at 0.3
        assert curve[-1] == pytest.approx(0.3, abs=0.01)

    def test_needs_two_points(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            AutomationCurve([(0.0, 1.0)])

    def test_auto_sorts(self):
        env = AutomationCurve([(1.0, 0.0), (0.0, 1.0)])
        curve = env.generate(SAMPLE_RATE)
        # Should sort to (0,1) -> (1,0): starts at 1, ends at 0
        assert curve[0] == pytest.approx(1.0, abs=0.01)
