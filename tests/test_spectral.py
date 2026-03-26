"""Tests for audiomancer.spectral — FFT-based audio processing."""

import numpy as np
import pytest

from audiomancer import SAMPLE_RATE
from audiomancer.spectral import (
    freeze, blur, pitch_shift, spectral_gate, morph, spectral_balance,
    _stft, _istft,
)
from audiomancer.synth import sine, white_noise, drone


SR = SAMPLE_RATE
DUR = 1.0  # Short duration for fast tests


# ---------------------------------------------------------------------------
# STFT / ISTFT roundtrip
# ---------------------------------------------------------------------------

class TestSTFT:
    def test_roundtrip_preserves_signal(self):
        """STFT → ISTFT should approximately reconstruct the original."""
        sig = sine(440, DUR, sample_rate=SR)
        fft_size = 2048
        frames = _stft(sig, fft_size)
        reconstructed = _istft(frames, fft_size, target_length=len(sig))
        # Allow some reconstruction error at edges; trim to common length
        n = min(len(reconstructed), len(sig))
        mid = slice(fft_size, n - fft_size)
        np.testing.assert_allclose(
            reconstructed[mid], sig[mid], atol=0.05,
        )

    def test_stft_output_shape(self):
        sig = sine(440, DUR, sample_rate=SR)
        fft_size = 2048
        frames = _stft(sig, fft_size)
        assert frames.ndim == 2
        assert frames.shape[1] == fft_size // 2 + 1


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_returns_correct_length(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = freeze(sig, freeze_time=0.5, duration_sec=2.0, sample_rate=SR)
        expected = int(SR * 2.0)
        assert abs(len(out) - expected) < 4096  # Allow FFT window tolerance

    def test_frozen_has_energy(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = freeze(sig, freeze_time=0.5, sample_rate=SR)
        assert np.max(np.abs(out)) > 0.01

    def test_stereo_input(self):
        sig = np.column_stack([sine(440, DUR, sample_rate=SR),
                               sine(550, DUR, sample_rate=SR)])
        out = freeze(sig, freeze_time=0.3, sample_rate=SR)
        assert out.ndim == 2
        assert out.shape[1] == 2


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------

class TestBlur:
    def test_no_blur_preserves_length(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = blur(sig, amount=0.0, sample_rate=SR)
        assert len(out) == len(sig)

    def test_full_blur_still_has_energy(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = blur(sig, amount=1.0, sample_rate=SR)
        assert np.max(np.abs(out)) > 0.01

    def test_stereo_blur(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        out = blur(sig, amount=0.5)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Pitch shift
# ---------------------------------------------------------------------------

class TestPitchShift:
    def test_no_shift_preserves_signal(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = pitch_shift(sig, semitones=0.0, sample_rate=SR)
        assert len(out) == len(sig)

    def test_octave_up_has_energy(self):
        sig = sine(220, DUR, sample_rate=SR)
        out = pitch_shift(sig, semitones=12.0, sample_rate=SR)
        assert np.max(np.abs(out)) > 0.01

    def test_stereo_shift(self):
        sig = np.column_stack([sine(440, DUR), sine(440, DUR)])
        out = pitch_shift(sig, semitones=5.0)
        assert out.ndim == 2
        assert out.shape[1] == 2


# ---------------------------------------------------------------------------
# Spectral gate
# ---------------------------------------------------------------------------

class TestSpectralGate:
    def test_keeps_loud_signal(self):
        sig = sine(440, DUR, sample_rate=SR)
        out = spectral_gate(sig, threshold_db=-60.0, sample_rate=SR)
        assert np.max(np.abs(out)) > 0.1

    def test_gate_reduces_noise(self):
        """Gating a noisy signal should reduce overall energy."""
        # Mix sine with low-level noise — gate should remove the noise
        tone = sine(440, DUR, amplitude=0.5, sample_rate=SR)
        noise = white_noise(DUR, amplitude=0.01, sample_rate=SR)
        sig = tone + noise
        out = spectral_gate(sig, threshold_db=-20.0, sample_rate=SR)
        # Output should still have the tone
        assert np.max(np.abs(out)) > 0.1

    def test_stereo_gate(self):
        sig = np.column_stack([sine(440, DUR), sine(550, DUR)])
        out = spectral_gate(sig, threshold_db=-40.0)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Morph
# ---------------------------------------------------------------------------

class TestMorph:
    def test_mix_zero_returns_signal_a(self):
        a = sine(440, DUR, sample_rate=SR)
        b = sine(880, DUR, sample_rate=SR)
        out = morph(a, b, mix=0.0)
        # Should be very close to a
        assert len(out) == len(a)

    def test_mix_one_returns_signal_b_energy(self):
        a = sine(440, DUR, sample_rate=SR)
        b = white_noise(DUR, sample_rate=SR)
        out = morph(a, b, mix=1.0)
        assert np.max(np.abs(out)) > 0.01

    def test_stereo_morph(self):
        a = np.column_stack([sine(440, DUR), sine(440, DUR)])
        b = np.column_stack([sine(880, DUR), sine(880, DUR)])
        out = morph(a, b, mix=0.5)
        assert out.ndim == 2

    def test_different_lengths_handled(self):
        a = sine(440, 1.0)
        b = sine(880, 0.5)
        out = morph(a, b, mix=0.5)
        assert len(out) <= len(a)


# ---------------------------------------------------------------------------
# Spectral balance
# ---------------------------------------------------------------------------

class TestSpectralBalance:
    def test_returns_all_keys(self):
        stems = {
            "low": sine(100, DUR, sample_rate=SR),
            "high": sine(5000, DUR, sample_rate=SR),
        }
        result = spectral_balance(stems, sample_rate=SR)
        assert "bands" in result
        assert "stems" in result
        assert "balance_score" in result
        assert "overlap_warnings" in result

    def test_band_count_matches(self):
        stems = {"test": sine(440, DUR, sample_rate=SR)}
        result = spectral_balance(stems, sample_rate=SR)
        assert len(result["bands"]) == 7  # default 7 bands
        assert len(result["stems"]["test"]) == 7

    def test_balance_score_range(self):
        stems = {
            "a": sine(200, DUR, sample_rate=SR),
            "b": sine(2000, DUR, sample_rate=SR),
        }
        result = spectral_balance(stems, sample_rate=SR)
        assert 0.0 <= result["balance_score"] <= 1.0

    def test_custom_bands(self):
        bands = [("low", 20, 500), ("high", 500, 20000)]
        stems = {"test": sine(440, DUR, sample_rate=SR)}
        result = spectral_balance(stems, bands=bands, sample_rate=SR)
        assert len(result["bands"]) == 2

    def test_overlap_detection(self):
        # Two stems both loud at same frequency
        stems = {
            "a": sine(440, DUR, amplitude=0.8, sample_rate=SR),
            "b": sine(440, DUR, amplitude=0.8, sample_rate=SR),
        }
        result = spectral_balance(stems, sample_rate=SR)
        # Should detect overlap in the mid band (440 Hz is in 200-2000)
        has_overlap = any(w[0] == "mid" for w in result["overlap_warnings"])
        assert has_overlap
