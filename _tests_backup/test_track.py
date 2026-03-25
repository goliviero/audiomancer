"""Tests for fractal.track — Track container and panning."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.generators import sine
from fractal.signal import mono_to_stereo, is_stereo, is_mono
from fractal.effects import HighPassFilter, EQ
from fractal.track import Track, apply_pan


class TestApplyPan:
    def test_center_pan_equal_channels(self):
        """Pan=0 should produce equal levels on both channels."""
        sig = sine(440, 0.5, amplitude=0.5)
        stereo = apply_pan(sig, 0.0)
        assert is_stereo(stereo)
        rms_left = np.sqrt(np.mean(stereo[:, 0] ** 2))
        rms_right = np.sqrt(np.mean(stereo[:, 1] ** 2))
        assert abs(rms_left - rms_right) < 0.001

    def test_full_left(self):
        """Pan=-1 should have signal only in left channel."""
        sig = sine(440, 0.5, amplitude=0.5)
        stereo = apply_pan(sig, -1.0)
        rms_left = np.sqrt(np.mean(stereo[:, 0] ** 2))
        rms_right = np.sqrt(np.mean(stereo[:, 1] ** 2))
        assert rms_left > 0.3
        assert rms_right < 0.001

    def test_full_right(self):
        """Pan=+1 should have signal only in right channel."""
        sig = sine(440, 0.5, amplitude=0.5)
        stereo = apply_pan(sig, 1.0)
        rms_left = np.sqrt(np.mean(stereo[:, 0] ** 2))
        rms_right = np.sqrt(np.mean(stereo[:, 1] ** 2))
        assert rms_left < 0.001
        assert rms_right > 0.3

    def test_mono_input_returns_stereo(self):
        """Mono input should be converted to stereo."""
        sig = sine(440, 0.5)
        assert is_mono(sig)
        panned = apply_pan(sig, 0.3)
        assert is_stereo(panned)

    def test_stereo_input_preserved(self):
        """Stereo input should keep shape."""
        sig = mono_to_stereo(sine(440, 0.5))
        panned = apply_pan(sig, -0.5)
        assert is_stereo(panned)
        assert panned.shape == sig.shape

    def test_pan_clamp(self):
        """Values outside [-1, 1] should be clamped, not crash."""
        sig = sine(440, 0.5)
        left = apply_pan(sig, -5.0)
        right = apply_pan(sig, 10.0)
        assert is_stereo(left)
        assert is_stereo(right)

    def test_equal_power_center_gain(self):
        """At center, each channel should be ~0.707 of original (equal-power law)."""
        sig = np.ones(1000)
        panned = apply_pan(sig, 0.0)
        expected = np.cos(np.pi / 4)  # ~0.7071
        assert abs(panned[500, 0] - expected) < 0.001
        assert abs(panned[500, 1] - expected) < 0.001


class TestTrack:
    def test_basic_render(self):
        """Track.render() should produce stereo output."""
        sig = sine(440, 0.5, amplitude=0.5)
        track = Track(name="test", signal=sig)
        rendered = track.render()
        assert is_stereo(rendered)
        assert rendered.shape[0] == sig.shape[0]

    def test_volume_gain(self):
        """Positive volume_db should increase amplitude."""
        sig = sine(440, 0.5, amplitude=0.3)
        quiet = Track(name="quiet", signal=sig, volume_db=-12.0).render()
        loud = Track(name="loud", signal=sig, volume_db=6.0).render()
        rms_quiet = np.sqrt(np.mean(quiet ** 2))
        rms_loud = np.sqrt(np.mean(loud ** 2))
        assert rms_loud > rms_quiet * 3

    def test_mute_returns_silence(self):
        """Muted track should render silence."""
        sig = sine(440, 0.5, amplitude=0.5)
        track = Track(name="muted", signal=sig, mute=True)
        rendered = track.render()
        assert np.max(np.abs(rendered)) == 0.0

    def test_pan_left(self):
        """Panning left should put more energy in left channel."""
        sig = sine(440, 0.5, amplitude=0.5)
        track = Track(name="left", signal=sig, pan=-0.8)
        rendered = track.render()
        rms_l = np.sqrt(np.mean(rendered[:, 0] ** 2))
        rms_r = np.sqrt(np.mean(rendered[:, 1] ** 2))
        assert rms_l > rms_r * 2

    def test_effects_applied(self):
        """Effects should be applied before gain/pan."""
        sig = sine(8000, 1.0, amplitude=0.5)
        track_no_fx = Track(name="no_fx", signal=sig)
        track_with_fx = Track(name="with_fx", signal=sig,
                              effects=[HighPassFilter(cutoff_hz=10000)])
        rms_no_fx = np.sqrt(np.mean(track_no_fx.render() ** 2))
        rms_with_fx = np.sqrt(np.mean(track_with_fx.render() ** 2))
        # HPF at 10kHz should cut the 8kHz tone significantly
        assert rms_with_fx < rms_no_fx * 0.5

    def test_bus_default(self):
        """Default bus should be 'master'."""
        track = Track(name="t", signal=np.zeros(100))
        assert track.bus == "master"

    def test_does_not_mutate_input(self):
        """Rendering should not modify the original signal."""
        sig = sine(440, 0.5, amplitude=0.5)
        original = sig.copy()
        Track(name="t", signal=sig, volume_db=6.0, pan=0.5).render()
        np.testing.assert_array_equal(sig, original)
