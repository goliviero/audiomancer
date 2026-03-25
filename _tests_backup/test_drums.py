"""Tests for fractal.drums -- kick, snare, hihat, clap, tom, cymbal, drum_kit."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.drums import (
    kick, snare, hihat, clap, tom, cymbal, drum_kit,
    _KIT_CONFIGS,
)


SR = SAMPLE_RATE


# =============================================
# Kick
# =============================================

class TestKick:
    def test_returns_correct_length(self):
        sig = kick(duration_sec=0.3)
        assert len(sig) == int(SR * 0.3)

    def test_amplitude_bounded(self):
        sig = kick(amplitude=0.5)
        assert np.max(np.abs(sig)) <= 0.51

    def test_drive_adds_saturation(self):
        """Higher drive should produce more harmonic content."""
        clean = kick(drive=1.0, amplitude=1.0)
        driven = kick(drive=3.0, amplitude=1.0)
        fft_clean = np.abs(np.fft.rfft(clean))
        fft_driven = np.abs(np.fft.rfft(driven))
        freq_bins = np.fft.rfftfreq(len(clean), 1.0 / SR)
        mask = freq_bins > 500
        assert np.sum(fft_driven[mask]) > np.sum(fft_clean[mask])

    def test_starts_with_energy(self):
        """Kick should have energy in the first few ms (attack)."""
        sig = kick(amplitude=1.0)
        first_ms = sig[:int(SR * 0.01)]
        assert np.max(np.abs(first_ms)) > 0.1

    def test_no_click_still_works(self):
        sig = kick(click_amount=0.0)
        assert len(sig) > 0
        assert np.isfinite(sig).all()


# =============================================
# Snare
# =============================================

class TestSnare:
    def test_returns_correct_length(self):
        sig = snare(duration_sec=0.2)
        assert len(sig) == int(SR * 0.2)

    def test_amplitude_bounded(self):
        sig = snare(amplitude=0.5)
        assert np.max(np.abs(sig)) <= 0.51

    def test_pure_noise_snare(self):
        """noise_amount=1.0 should still produce a valid signal."""
        sig = snare(noise_amount=1.0)
        assert np.isfinite(sig).all()

    def test_pure_tone_snare(self):
        """noise_amount=0.0 should produce a tonal signal."""
        sig = snare(noise_amount=0.0, ring=0.0)
        assert np.isfinite(sig).all()
        # Should have strong fundamental
        fft_sig = np.abs(np.fft.rfft(sig))
        peak_bin = np.argmax(fft_sig)
        freq_bins = np.fft.rfftfreq(len(sig), 1.0 / SR)
        assert freq_bins[peak_bin] < 400  # fundamental should be near tone_hz


# =============================================
# Hi-hat
# =============================================

class TestHihat:
    def test_returns_correct_length(self):
        sig = hihat(duration_sec=0.05)
        assert len(sig) == int(SR * 0.05)

    def test_closed_hat_decays_fast(self):
        """Closed hi-hat should have most energy in the first half."""
        sig = hihat(duration_sec=0.1, openness=0.0)
        mid = len(sig) // 2
        energy_first = np.sum(sig[:mid] ** 2)
        energy_second = np.sum(sig[mid:] ** 2)
        assert energy_first > energy_second * 5

    def test_open_hat_sustains(self):
        """Open hi-hat should sustain more than closed."""
        closed = hihat(duration_sec=0.2, openness=0.0)
        opened = hihat(duration_sec=0.2, openness=1.0)
        mid = len(closed) // 2
        # Compare energy ratio (second half / first half)
        ratio_closed = np.sum(closed[mid:] ** 2) / (np.sum(closed[:mid] ** 2) + 1e-10)
        ratio_open = np.sum(opened[mid:] ** 2) / (np.sum(opened[:mid] ** 2) + 1e-10)
        assert ratio_open > ratio_closed


# =============================================
# Clap
# =============================================

class TestClap:
    def test_returns_correct_length(self):
        sig = clap(duration_sec=0.15)
        assert len(sig) == int(SR * 0.15)

    def test_amplitude_bounded(self):
        sig = clap(amplitude=0.5)
        assert np.max(np.abs(sig)) <= 0.51

    def test_has_multiple_transients(self):
        """Clap should have multiple energy peaks (the bursts)."""
        sig = clap(duration_sec=0.15, spread=0.5)
        # Check energy in 10ms windows
        window = int(SR * 0.01)
        energies = [np.sum(sig[i:i+window] ** 2)
                    for i in range(0, len(sig) - window, window)]
        # Should have at least 2 peaks above 10% of max
        threshold = 0.1 * max(energies)
        peaks = sum(1 for e in energies if e > threshold)
        assert peaks >= 2


# =============================================
# Tom
# =============================================

class TestTom:
    def test_returns_correct_length(self):
        sig = tom(duration_sec=0.25)
        assert len(sig) == int(SR * 0.25)

    def test_pitch_affects_spectrum(self):
        """Lower tom should have lower spectral centroid."""
        low = tom(pitch_hz=80)
        high = tom(pitch_hz=200)
        fft_low = np.abs(np.fft.rfft(low))
        fft_high = np.abs(np.fft.rfft(high))
        freq_bins = np.fft.rfftfreq(len(low), 1.0 / SR)
        centroid_low = np.average(freq_bins, weights=fft_low)
        centroid_high = np.average(freq_bins, weights=fft_high)
        assert centroid_low < centroid_high


# =============================================
# Cymbal
# =============================================

class TestCymbal:
    def test_returns_correct_length(self):
        sig = cymbal(duration_sec=1.0)
        assert len(sig) == int(SR * 1.0)

    def test_brightness_affects_spectrum(self):
        """Brighter cymbal should have more high-frequency content."""
        dark = cymbal(brightness=0.0, duration_sec=0.5)
        bright = cymbal(brightness=1.0, duration_sec=0.5)
        fft_dark = np.abs(np.fft.rfft(dark))
        fft_bright = np.abs(np.fft.rfft(bright))
        freq_bins = np.fft.rfftfreq(len(dark), 1.0 / SR)
        mask = freq_bins > 8000
        assert np.sum(fft_bright[mask]) > np.sum(fft_dark[mask])


# =============================================
# Drum Kit
# =============================================

class TestDrumKit:
    def test_808_kit_has_all_pieces(self):
        kit = drum_kit("808")
        expected = {"kick", "snare", "hihat_closed", "hihat_open",
                    "clap", "tom_low", "tom_mid", "tom_high", "cymbal"}
        assert set(kit.keys()) == expected

    def test_all_kits_valid(self):
        """All kit styles should produce valid signals."""
        for style in _KIT_CONFIGS:
            kit = drum_kit(style, amplitude=0.3)
            for name, sig in kit.items():
                assert len(sig) > 0, f"{style}/{name} is empty"
                assert np.isfinite(sig).all(), f"{style}/{name} has NaN/Inf"
                assert np.max(np.abs(sig)) <= 0.31, f"{style}/{name} exceeds amplitude"

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError):
            drum_kit("dubstep")

    def test_kit_amplitude_respected(self):
        kit = drum_kit("808", amplitude=0.3)
        for name, sig in kit.items():
            assert abs(np.max(np.abs(sig)) - 0.3) < 0.02, f"{name} amplitude wrong"
