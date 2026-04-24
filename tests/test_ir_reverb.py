"""Tests for audiomancer.ir_reverb — convolution reverb + synthetic IRs."""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.ir_reverb import (
    convolve_reverb,
    reverb_from_synthetic,
    synthetic_ir,
)
from audiomancer.synth import sine

SR = SAMPLE_RATE


class TestSyntheticIR:
    def test_preset_shapes(self):
        for space in ("room", "hall", "cathedral", "plate"):
            ir = synthetic_ir(space, seed=42, sample_rate=SR)
            assert ir.ndim == 2
            assert ir.shape[1] == 2
            assert ir.shape[0] > 0
            assert np.max(np.abs(ir)) <= 1.0

    def test_cathedral_longer_than_room(self):
        room = synthetic_ir("room", seed=42, sample_rate=SR)
        cathedral = synthetic_ir("cathedral", seed=42, sample_rate=SR)
        assert cathedral.shape[0] > room.shape[0]

    def test_invalid_space_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown space"):
            synthetic_ir("casino", seed=42, sample_rate=SR)

    def test_seed_reproducible(self):
        a = synthetic_ir("hall", seed=42, sample_rate=SR)
        b = synthetic_ir("hall", seed=42, sample_rate=SR)
        assert np.allclose(a, b)


class TestConvolveReverb:
    def test_mono_input(self):
        sig = sine(440.0, 1.0, sample_rate=SR)
        ir = synthetic_ir("room", seed=42, sample_rate=SR)
        out = convolve_reverb(sig, ir, wet=0.5, sample_rate=SR)
        assert out.ndim == 2  # auto-converted to stereo
        assert out.shape[0] == len(sig)

    def test_stereo_input(self):
        mono = sine(440.0, 1.0, sample_rate=SR)
        sig = np.column_stack([mono, mono])
        ir = synthetic_ir("hall", seed=42, sample_rate=SR)
        out = convolve_reverb(sig, ir, wet=0.3, sample_rate=SR)
        assert out.shape == sig.shape

    def test_wet_zero_is_dry(self):
        sig = sine(440.0, 0.5, sample_rate=SR)
        ir = synthetic_ir("room", seed=42, sample_rate=SR)
        out = convolve_reverb(sig, ir, wet=0.0, sample_rate=SR)
        # wet=0 means all dry: stereo signal == mono duplicated
        from audiomancer.utils import mono_to_stereo
        expected = mono_to_stereo(sig)
        assert np.allclose(out, expected, atol=1e-6)

    def test_from_synthetic_wrapper(self):
        sig = sine(440.0, 0.5, sample_rate=SR)
        out = reverb_from_synthetic(sig, space="cathedral", wet=0.4,
                                    seed=42, sample_rate=SR)
        assert out.ndim == 2
        assert out.shape[0] == len(sig)
