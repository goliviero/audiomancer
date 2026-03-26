"""Tests for audiomancer.spatial — stereo positioning and width."""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.spatial import (
    auto_pan,
    decode_mid_side,
    encode_mid_side,
    haas_width,
    pan,
    rotate,
    stereo_width,
)
from audiomancer.synth import sine

SR = SAMPLE_RATE
DUR = 0.5


# ---------------------------------------------------------------------------
# Pan
# ---------------------------------------------------------------------------

class TestPan:
    def test_center_pan_equal_channels(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = pan(sig, position=0.0)
        assert out.ndim == 2
        assert out.shape[1] == 2
        # Center = equal power on both channels
        np.testing.assert_allclose(
            np.max(np.abs(out[:, 0])),
            np.max(np.abs(out[:, 1])),
            atol=0.01,
        )

    def test_hard_left(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = pan(sig, position=-1.0)
        assert np.max(np.abs(out[:, 0])) > np.max(np.abs(out[:, 1]))
        assert np.max(np.abs(out[:, 1])) < 0.01

    def test_hard_right(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = pan(sig, position=1.0)
        assert np.max(np.abs(out[:, 1])) > np.max(np.abs(out[:, 0]))
        assert np.max(np.abs(out[:, 0])) < 0.01

    def test_stereo_input(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        out = pan(sig, position=0.3)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Auto-pan
# ---------------------------------------------------------------------------

class TestAutoPan:
    def test_output_is_stereo(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = auto_pan(sig, rate_hz=1.0, sample_rate=SR)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_no_depth_equals_center(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = auto_pan(sig, depth=0.0, sample_rate=SR)
        np.testing.assert_allclose(
            np.max(np.abs(out[:, 0])),
            np.max(np.abs(out[:, 1])),
            atol=0.01,
        )

    def test_stereo_input(self):
        sig = np.column_stack([sine(440, DUR), sine(440, DUR)])
        out = auto_pan(sig, rate_hz=0.5)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Stereo width
# ---------------------------------------------------------------------------

class TestStereoWidth:
    def test_mono_collapse(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        out = stereo_width(sig, width=0.0)
        # Width 0 = mono, both channels identical
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-10)

    def test_unchanged_at_one(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        out = stereo_width(sig, width=1.0)
        np.testing.assert_allclose(out, sig, atol=1e-10)

    def test_wider_increases_difference(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        normal = stereo_width(sig, width=1.0)
        wide = stereo_width(sig, width=2.0)
        diff_normal = np.std(normal[:, 0] - normal[:, 1])
        diff_wide = np.std(wide[:, 0] - wide[:, 1])
        assert diff_wide > diff_normal

    def test_mono_input(self):
        sig = sine(440, DUR)
        out = stereo_width(sig, width=1.0)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Mid/Side
# ---------------------------------------------------------------------------

class TestMidSide:
    def test_roundtrip(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        mid, side = encode_mid_side(sig)
        reconstructed = decode_mid_side(mid, side)
        np.testing.assert_allclose(reconstructed, sig, atol=1e-10)

    def test_mono_signal_no_side(self):
        mono = sine(440, DUR)
        stereo = np.column_stack([mono, mono])
        mid, side = encode_mid_side(stereo)
        assert np.max(np.abs(side)) < 1e-10

    def test_mono_input_encode(self):
        sig = sine(440, DUR)
        mid, side = encode_mid_side(sig)
        assert np.max(np.abs(side)) < 1e-10


# ---------------------------------------------------------------------------
# Haas width
# ---------------------------------------------------------------------------

class TestHaasWidth:
    def test_output_stereo(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = haas_width(sig, delay_ms=15.0, sample_rate=SR)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_channels_are_different(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = haas_width(sig, delay_ms=15.0, sample_rate=SR)
        assert not np.allclose(out[:, 0], out[:, 1])

    def test_stereo_input(self):
        sig = np.column_stack([sine(440, DUR), sine(440, DUR)])
        out = haas_width(sig, delay_ms=10.0)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------

class TestRotate:
    def test_output_stereo(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = rotate(sig, revolutions=1.0, sample_rate=SR)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_preserves_length(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = rotate(sig, sample_rate=SR)
        assert out.shape[0] == len(sig)

    def test_stereo_input(self):
        sig = np.column_stack([sine(440, DUR), sine(440, DUR)])
        out = rotate(sig, revolutions=0.5)
        assert out.ndim == 2
