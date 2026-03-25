"""Tests for fractal.effects — audio processing transforms."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.generators import sine, white_noise, binaural
from fractal.signal import mono_to_stereo, is_stereo
from fractal.effects import (
    BandPassFilter,
    Delay,
    Distortion,
    EffectChain,
    EQ,
    HighPassFilter,
    LowPassFilter,
    NormalizePeak,
    Reverb,
    StereoWidth,
)


class TestLowPassFilter:
    def test_attenuates_high_frequencies(self):
        """A 5000Hz tone through a 1000Hz LPF should be much quieter."""
        sig = sine(5000, 1.0, amplitude=0.5)
        filtered = LowPassFilter(cutoff_hz=1000).process(sig)
        # RMS should drop significantly
        rms_before = np.sqrt(np.mean(sig ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        assert rms_after < rms_before * 0.2

    def test_passes_low_frequencies(self):
        """A 100Hz tone through a 1000Hz LPF should be mostly unchanged."""
        sig = sine(100, 1.0, amplitude=0.5)
        filtered = LowPassFilter(cutoff_hz=1000).process(sig)
        rms_before = np.sqrt(np.mean(sig ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        assert rms_after > rms_before * 0.9

    def test_stereo(self):
        sig = mono_to_stereo(sine(5000, 0.5, amplitude=0.5))
        filtered = LowPassFilter(cutoff_hz=1000).process(sig)
        assert is_stereo(filtered)


class TestHighPassFilter:
    def test_attenuates_low_frequencies(self):
        sig = sine(50, 1.0, amplitude=0.5)
        filtered = HighPassFilter(cutoff_hz=500).process(sig)
        rms_before = np.sqrt(np.mean(sig ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        assert rms_after < rms_before * 0.2

    def test_passes_high_frequencies(self):
        sig = sine(5000, 1.0, amplitude=0.5)
        filtered = HighPassFilter(cutoff_hz=500).process(sig)
        rms_before = np.sqrt(np.mean(sig ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        assert rms_after > rms_before * 0.9


class TestBandPassFilter:
    def test_passes_in_band(self):
        sig = sine(500, 1.0, amplitude=0.5)
        filtered = BandPassFilter(low_hz=200, high_hz=1000).process(sig)
        rms_before = np.sqrt(np.mean(sig ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        assert rms_after > rms_before * 0.8

    def test_attenuates_out_of_band(self):
        sig = sine(5000, 1.0, amplitude=0.5)
        filtered = BandPassFilter(low_hz=200, high_hz=1000).process(sig)
        rms_before = np.sqrt(np.mean(sig ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        assert rms_after < rms_before * 0.2


class TestEQ:
    def test_low_shelf_boost(self):
        """Boosting low shelf should increase low-frequency energy."""
        sig = white_noise(1.0, amplitude=0.3)
        eq = EQ(bands=[{"type": "low_shelf", "freq": 200, "gain_db": 6.0}])
        processed = eq.process(sig)
        # Check that something changed
        assert not np.array_equal(sig, processed)

    def test_high_shelf_cut(self):
        sig = white_noise(1.0, amplitude=0.3)
        eq = EQ(bands=[{"type": "high_shelf", "freq": 4000, "gain_db": -6.0}])
        processed = eq.process(sig)
        rms_after = np.sqrt(np.mean(processed ** 2))
        rms_before = np.sqrt(np.mean(sig ** 2))
        assert rms_after < rms_before

    def test_peak_boost(self):
        sig = white_noise(1.0, amplitude=0.3)
        eq = EQ(bands=[{"type": "peak", "freq": 1000, "gain_db": 6.0}])
        processed = eq.process(sig)
        assert not np.array_equal(sig, processed)

    def test_multiple_bands(self):
        sig = white_noise(1.0, amplitude=0.3)
        eq = EQ(bands=[
            {"type": "low_shelf", "freq": 80, "gain_db": 3.0},
            {"type": "peak", "freq": 2500, "gain_db": 2.0},
            {"type": "high_shelf", "freq": 8000, "gain_db": -2.0},
        ])
        processed = eq.process(sig)
        assert len(processed) == len(sig)


class TestStereoWidth:
    def test_mono_passthrough(self):
        sig = sine(440, 0.5)
        result = StereoWidth(width=1.5).process(sig)
        np.testing.assert_array_equal(sig, result)

    def test_width_zero_is_mono(self):
        """Width 0 should collapse stereo to mono (L=R)."""
        left = sine(440, 0.5, amplitude=0.5)
        right = sine(550, 0.5, amplitude=0.5)
        stereo = np.column_stack([left, right])
        result = StereoWidth(width=0.0).process(stereo)
        np.testing.assert_allclose(result[:, 0], result[:, 1], atol=1e-10)

    def test_width_increases_difference(self):
        """Width > 1 should increase the L-R difference."""
        left = sine(440, 0.5, amplitude=0.5)
        right = sine(550, 0.5, amplitude=0.3)
        stereo = np.column_stack([left, right])
        original_diff = np.mean(np.abs(stereo[:, 0] - stereo[:, 1]))
        widened = StereoWidth(width=2.0).process(stereo)
        widened_diff = np.mean(np.abs(widened[:, 0] - widened[:, 1]))
        assert widened_diff > original_diff


class TestReverb:
    def test_adds_tail(self):
        """Reverb should make the signal more diffuse (higher RMS near the end)."""
        sig = np.zeros(SAMPLE_RATE)
        sig[:1000] = sine(440, 1000 / SAMPLE_RATE, amplitude=0.5)
        processed = Reverb(decay=0.5, mix=0.5).process(sig)
        # The tail (last 25%) should have more energy than dry
        tail_rms = np.sqrt(np.mean(processed[-SAMPLE_RATE // 4:] ** 2))
        dry_tail_rms = np.sqrt(np.mean(sig[-SAMPLE_RATE // 4:] ** 2))
        assert tail_rms > dry_tail_rms

    def test_stereo(self):
        sig = binaural(200, 10, 0.5)
        processed = Reverb(decay=0.3, mix=0.2).process(sig)
        assert is_stereo(processed)

    def test_room_sizes(self):
        sig = sine(440, 0.5, amplitude=0.5)
        for size in ["small", "medium", "large"]:
            processed = Reverb(decay=0.3, mix=0.3, room_size=size).process(sig)
            assert len(processed) == len(sig)


class TestDelay:
    def test_creates_echoes(self):
        sig = np.zeros(SAMPLE_RATE)
        sig[:1000] = 0.5  # short impulse
        delayed = Delay(delay_ms=200, feedback=0.5, mix=0.5).process(sig)
        # Energy at the echo point should be nonzero
        echo_start = int(0.2 * SAMPLE_RATE)
        echo_energy = np.sum(np.abs(delayed[echo_start:echo_start + 1000]))
        assert echo_energy > 0.1

    def test_mix_zero_is_dry(self):
        sig = sine(440, 0.5, amplitude=0.5)
        delayed = Delay(mix=0.0).process(sig)
        np.testing.assert_array_equal(sig, delayed)


class TestDistortion:
    def test_soft_clip(self):
        """Distortion should keep signal within original peak range."""
        sig = sine(440, 0.5, amplitude=0.8)
        distorted = Distortion(drive=5.0).process(sig)
        assert np.max(np.abs(distorted)) <= 1.0 + 1e-6

    def test_adds_harmonics(self):
        """Distorted signal should have more spectral content."""
        sig = sine(440, 1.0, amplitude=0.5)
        distorted = Distortion(drive=5.0).process(sig)
        fft_orig = np.abs(np.fft.rfft(sig))
        fft_dist = np.abs(np.fft.rfft(distorted))
        # Count significant frequency bins (above 1% of max)
        threshold = 0.01 * np.max(fft_orig)
        orig_bins = np.sum(fft_orig > threshold)
        dist_bins = np.sum(fft_dist > threshold)
        assert dist_bins > orig_bins

    def test_mix_zero(self):
        sig = sine(440, 0.5, amplitude=0.5)
        result = Distortion(drive=5.0, mix=0.0).process(sig)
        np.testing.assert_allclose(sig, result, atol=1e-10)


class TestNormalizePeak:
    def test_normalize(self):
        sig = sine(440, 0.5, amplitude=0.3)
        normalized = NormalizePeak(target_db=-1.0).process(sig)
        target = 10 ** (-1.0 / 20)
        assert np.max(np.abs(normalized)) == pytest.approx(target, rel=1e-4)


class TestEffectChain:
    def test_chain_order(self):
        """Effects should be applied in order."""
        sig = sine(440, 0.5, amplitude=0.5)
        chain = EffectChain([
            LowPassFilter(cutoff_hz=200),
            NormalizePeak(target_db=-1.0),
        ])
        result = chain.process(sig)
        # 440Hz through 200Hz LPF should be very quiet, then normalized back up
        target = 10 ** (-1.0 / 20)
        assert np.max(np.abs(result)) == pytest.approx(target, rel=1e-3)

    def test_empty_chain(self):
        sig = sine(440, 0.5, amplitude=0.5)
        result = EffectChain([]).process(sig)
        np.testing.assert_array_equal(sig, result)
