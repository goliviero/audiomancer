"""Tests for fractal.modulation -- LFO, vibrato, filter sweep, param automation."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.modulation import LFO, apply_vibrato, apply_filter_sweep, apply_param_automation
from fractal.generators import sine, sawtooth
from fractal.effects import LowPassFilter


SR = SAMPLE_RATE
DUR = 0.5
N = int(SR * DUR)


# =============================================
# LFO
# =============================================

class TestLFO:
    def test_returns_correct_length(self):
        lfo = LFO(rate=2.0)
        sig = lfo.generate(N)
        assert len(sig) == N

    def test_unipolar_range(self):
        """Unipolar LFO should be in [0, depth]."""
        lfo = LFO(rate=2.0, depth=1.0, bipolar=False)
        sig = lfo.generate(N)
        assert np.min(sig) >= -0.01
        assert np.max(sig) <= 1.01

    def test_bipolar_range(self):
        """Bipolar LFO should be in [-depth, +depth]."""
        lfo = LFO(rate=2.0, depth=1.0, bipolar=True)
        sig = lfo.generate(N)
        assert np.min(sig) >= -1.01
        assert np.max(sig) <= 1.01

    def test_depth_scales_output(self):
        """Depth=0.5 should halve the output range."""
        full = LFO(rate=2.0, depth=1.0, bipolar=True).generate(N)
        half = LFO(rate=2.0, depth=0.5, bipolar=True).generate(N)
        assert np.max(np.abs(half)) < np.max(np.abs(full))

    def test_all_shapes_valid(self):
        """All shapes should produce finite signals."""
        for shape in LFO._SHAPES:
            lfo = LFO(rate=1.0, shape=shape)
            sig = lfo.generate(N, seed=42)
            assert np.isfinite(sig).all(), f"Shape '{shape}' has NaN/Inf"

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            LFO(shape="wobble")

    def test_sine_shape_is_smooth(self):
        """Sine LFO should have small sample-to-sample differences."""
        lfo = LFO(rate=2.0, shape="sine", bipolar=True)
        sig = lfo.generate(N)
        diffs = np.abs(np.diff(sig))
        assert np.max(diffs) < 0.01

    def test_square_shape_is_binary(self):
        """Square LFO should have only two values."""
        lfo = LFO(rate=2.0, shape="square", depth=1.0, bipolar=True)
        sig = lfo.generate(N)
        unique = np.unique(sig)
        assert len(unique) == 2

    def test_sample_hold_steps(self):
        """Sample-hold LFO should hold constant within each cycle."""
        lfo = LFO(rate=4.0, shape="sample_hold", bipolar=True)
        sig = lfo.generate(N, seed=42)
        # Within a cycle (SR/rate samples), values should be constant
        cycle_len = int(SR / 4.0)
        segment = sig[:cycle_len]
        assert np.all(segment == segment[0])

    def test_modulate_param_center(self):
        """modulate_param should oscillate around base_value."""
        lfo = LFO(rate=2.0, shape="sine", depth=1.0, bipolar=True)
        params = lfo.modulate_param(1000, 500, N)
        assert np.mean(params) == pytest.approx(1000, abs=50)
        assert np.max(params) <= 1501
        assert np.min(params) >= 499


# =============================================
# Vibrato
# =============================================

class TestVibrato:
    def test_returns_correct_length(self):
        sig = sine(440, DUR)
        result = apply_vibrato(sig, rate=5.0, depth_cents=20.0)
        assert len(result) == len(sig)

    def test_zero_depth_is_identity(self):
        """Zero depth vibrato should not change the signal."""
        sig = sine(440, DUR, amplitude=0.5)
        result = apply_vibrato(sig, rate=5.0, depth_cents=0.0)
        assert np.allclose(sig, result, atol=1e-10)

    def test_vibrato_changes_signal(self):
        """Non-zero vibrato should modify the signal."""
        sig = sine(440, DUR, amplitude=0.5)
        result = apply_vibrato(sig, rate=5.0, depth_cents=50.0)
        assert not np.allclose(sig, result, atol=0.01)

    def test_vibrato_preserves_energy(self):
        """Vibrato should not drastically change signal energy."""
        sig = sine(440, DUR, amplitude=0.5)
        result = apply_vibrato(sig, rate=5.0, depth_cents=20.0)
        energy_orig = np.sum(sig ** 2)
        energy_vib = np.sum(result ** 2)
        assert abs(energy_vib / energy_orig - 1.0) < 0.1  # within 10%


# =============================================
# Filter Sweep
# =============================================

class TestFilterSweep:
    def test_returns_correct_length(self):
        sig = sawtooth(220, DUR)
        result = apply_filter_sweep(sig, 5000, 200)
        assert len(result) == len(sig)

    def test_lowpass_sweep_removes_highs(self):
        """Sweeping from high to low cutoff should reduce high frequencies."""
        sig = sawtooth(220, DUR, amplitude=1.0)
        result = apply_filter_sweep(sig, 5000, 200, filter_type="lowpass")
        # Second half should have less HF content (lower cutoff)
        mid = len(result) // 2
        fft_first = np.abs(np.fft.rfft(result[:mid]))
        fft_second = np.abs(np.fft.rfft(result[mid:]))
        freq_first = np.fft.rfftfreq(mid, 1.0 / SR)
        freq_second = np.fft.rfftfreq(len(result) - mid, 1.0 / SR)
        mask_first = freq_first > 2000
        mask_second = freq_second > 2000
        assert np.sum(fft_first[mask_first]) > np.sum(fft_second[mask_second])

    def test_exponential_curve(self):
        """Exponential sweep should also work."""
        sig = sawtooth(220, DUR)
        result = apply_filter_sweep(sig, 200, 5000, curve="exponential")
        assert len(result) == len(sig)
        assert np.isfinite(result).all()

    def test_highpass_sweep(self):
        """Highpass sweep should work."""
        sig = sawtooth(220, DUR)
        result = apply_filter_sweep(sig, 200, 5000, filter_type="highpass")
        assert np.isfinite(result).all()

    def test_invalid_filter_type_raises(self):
        sig = sawtooth(220, DUR)
        with pytest.raises(ValueError):
            apply_filter_sweep(sig, 200, 5000, filter_type="bandpass")

    def test_invalid_curve_raises(self):
        sig = sawtooth(220, DUR)
        with pytest.raises(ValueError):
            apply_filter_sweep(sig, 200, 5000, curve="quadratic")


# =============================================
# Parameter Automation
# =============================================

class TestParamAutomation:
    def test_returns_correct_length(self):
        sig = sawtooth(220, DUR)
        lpf = LowPassFilter(cutoff_hz=2000)
        automation = np.linspace(5000, 200, len(sig))
        result = apply_param_automation(sig, lpf, "cutoff_hz", automation)
        assert len(result) == len(sig)

    def test_automation_resizes(self):
        """Automation array of different length should be interpolated."""
        sig = sawtooth(220, DUR)
        lpf = LowPassFilter(cutoff_hz=2000)
        automation = np.linspace(5000, 200, 100)  # much shorter
        result = apply_param_automation(sig, lpf, "cutoff_hz", automation)
        assert len(result) == len(sig)
        assert np.isfinite(result).all()
