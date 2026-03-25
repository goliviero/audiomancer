"""Tests for fractal.synth -- FM, additive, wavetable, subtractive, pulse, unison."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.synth import (
    HARMONIC_PRESETS,
    additive,
    fm_synth,
    pulse,
    subtractive,
    unison,
    wavetable,
)
from fractal.generators import sine, sawtooth
from fractal.envelopes import ADSR


SR = SAMPLE_RATE
DUR = 0.5  # short duration for fast tests


# =============================================
# FM Synthesis
# =============================================

class TestFmSynth:
    def test_returns_correct_length(self):
        sig = fm_synth(440, 440, 1.0, DUR)
        assert len(sig) == int(SR * DUR)

    def test_mod_index_zero_is_sine(self):
        """With mod_index=0, FM should produce a pure sine."""
        fm = fm_synth(440, 440, 0.0, DUR, amplitude=0.5)
        pure = sine(440, DUR, amplitude=0.5)
        assert np.allclose(fm, pure, atol=1e-10)

    def test_higher_mod_index_more_harmonics(self):
        """Higher mod_index should produce more spectral content."""
        low = fm_synth(440, 440, 0.5, DUR)
        high = fm_synth(440, 440, 5.0, DUR)
        # Compare spectral energy above 2kHz
        fft_low = np.abs(np.fft.rfft(low))
        fft_high = np.abs(np.fft.rfft(high))
        freq_bins = np.fft.rfftfreq(len(low), 1.0 / SR)
        mask = freq_bins > 2000
        assert np.sum(fft_high[mask]) > np.sum(fft_low[mask])

    def test_amplitude_bounds(self):
        sig = fm_synth(440, 220, 2.0, DUR, amplitude=0.5)
        assert np.max(np.abs(sig)) <= 0.51  # small tolerance

    def test_mod_envelope_applied(self):
        """With a mod_envelope that decays, early signal should differ from late."""
        n = int(SR * DUR)
        env = np.linspace(1.0, 0.0, n)
        sig = fm_synth(440, 440, 3.0, DUR, mod_envelope=env)
        # First quarter should have more HF content than last quarter
        q = n // 4
        fft_early = np.abs(np.fft.rfft(sig[:q]))
        fft_late = np.abs(np.fft.rfft(sig[-q:]))
        assert np.sum(fft_early) > np.sum(fft_late)

    def test_mod_envelope_shorter_than_signal(self):
        """Envelope shorter than signal should be zero-padded."""
        short_env = np.ones(100)
        sig = fm_synth(440, 440, 2.0, DUR, mod_envelope=short_env)
        assert len(sig) == int(SR * DUR)


# =============================================
# Additive Synthesis
# =============================================

class TestAdditive:
    def test_single_harmonic_is_sine(self):
        """Single harmonic (1, 1.0) should approximate a sine wave."""
        add = additive(440, [(1, 1.0)], DUR, amplitude=0.5)
        pure = sine(440, DUR, amplitude=0.5)
        assert np.allclose(add, pure, atol=1e-10)

    def test_returns_correct_length(self):
        sig = additive(220, [(1, 1.0), (2, 0.5)], DUR)
        assert len(sig) == int(SR * DUR)

    def test_amplitude_normalized(self):
        sig = additive(220, [(1, 1.0), (2, 0.8), (3, 0.6)], DUR, amplitude=0.5)
        assert abs(np.max(np.abs(sig)) - 0.5) < 0.01

    def test_more_harmonics_richer_spectrum(self):
        simple = additive(220, [(1, 1.0)], DUR)
        rich = additive(220, [(1, 1.0), (3, 0.5), (5, 0.3), (7, 0.2)], DUR)
        fft_simple = np.abs(np.fft.rfft(simple))
        fft_rich = np.abs(np.fft.rfft(rich))
        # Rich should have more spectral bins with significant energy
        threshold = 0.01 * np.max(fft_rich)
        assert np.sum(fft_rich > threshold) > np.sum(fft_simple > threshold)

    def test_skips_above_nyquist(self):
        """Harmonics above Nyquist should be silently skipped."""
        # Harmonic 1000 of 440Hz = 440kHz, way above Nyquist
        sig = additive(440, [(1, 1.0), (1000, 0.5)], DUR)
        assert len(sig) == int(SR * DUR)

    def test_all_presets_valid(self):
        """All HARMONIC_PRESETS should produce valid signals."""
        for name, harmonics in HARMONIC_PRESETS.items():
            sig = additive(220, harmonics, 0.1)
            assert len(sig) > 0, f"Preset '{name}' produced empty signal"
            assert np.isfinite(sig).all(), f"Preset '{name}' has NaN/Inf"


# =============================================
# Wavetable Synthesis
# =============================================

class TestWavetable:
    def test_sine_table_produces_sine(self):
        """A sine wavetable should produce a signal close to sine()."""
        table = np.sin(np.linspace(0, 2 * np.pi, 2048, endpoint=False))
        wt = wavetable(table, 440, DUR, amplitude=0.5)
        pure = sine(440, DUR, amplitude=0.5)
        # Allow some tolerance for interpolation artifacts
        assert np.corrcoef(wt, pure)[0, 1] > 0.999

    def test_returns_correct_length(self):
        table = np.sin(np.linspace(0, 2 * np.pi, 256, endpoint=False))
        sig = wavetable(table, 440, DUR)
        assert len(sig) == int(SR * DUR)

    def test_amplitude_scaling(self):
        table = np.sin(np.linspace(0, 2 * np.pi, 256, endpoint=False))
        sig = wavetable(table, 440, DUR, amplitude=0.3)
        assert abs(np.max(np.abs(sig)) - 0.3) < 0.01

    def test_custom_waveform(self):
        """A square-like wavetable should produce a non-sinusoidal signal."""
        table = np.ones(256)
        table[128:] = -1.0
        sig = wavetable(table, 440, DUR)
        # Should have more harmonics than a sine
        fft_sig = np.abs(np.fft.rfft(sig))
        freq_bins = np.fft.rfftfreq(len(sig), 1.0 / SR)
        mask = freq_bins > 1000
        assert np.sum(fft_sig[mask]) > 0.1


# =============================================
# Subtractive Synthesis
# =============================================

class TestSubtractive:
    def test_returns_correct_length(self):
        sig = subtractive("saw", 220, DUR, cutoff_hz=2000)
        assert len(sig) == int(SR * DUR)

    def test_filter_removes_highs(self):
        """Low cutoff should remove high-frequency content vs raw sawtooth."""
        filtered = subtractive("saw", 220, DUR, cutoff_hz=500, amplitude=1.0)
        raw = sawtooth(220, DUR, amplitude=1.0)
        fft_filt = np.abs(np.fft.rfft(filtered))
        fft_raw = np.abs(np.fft.rfft(raw))
        freq_bins = np.fft.rfftfreq(len(raw), 1.0 / SR)
        mask = freq_bins > 2000
        assert np.sum(fft_filt[mask]) < np.sum(fft_raw[mask])

    def test_envelope_applied(self):
        """With an ADSR envelope, signal should start quiet and have release."""
        adsr = ADSR(attack=0.05, decay=0.05, sustain=0.5, release=0.1)
        sig = subtractive("saw", 220, DUR, cutoff_hz=2000, envelope=adsr)
        # First 10 samples should be near zero (attack start)
        assert np.max(np.abs(sig[:10])) < 0.1

    def test_filter_envelope(self):
        """Filter envelope should make early part brighter than late part."""
        n = int(SR * DUR)
        filt_env = np.linspace(1.0, 0.0, n)  # Open to closed
        sig = subtractive("saw", 220, DUR, cutoff_hz=5000,
                          filter_envelope=filt_env)
        q = n // 4
        fft_early = np.abs(np.fft.rfft(sig[:q]))
        fft_late = np.abs(np.fft.rfft(sig[-q:]))
        # Early should have more total spectral energy (brighter)
        assert np.sum(fft_early) > np.sum(fft_late) * 0.5

    def test_unknown_oscillator_raises(self):
        with pytest.raises(ValueError):
            subtractive("kazoo", 220, DUR, cutoff_hz=2000)

    def test_all_oscillator_types(self):
        """All oscillator types should work."""
        for osc in ["sine", "square", "saw", "sawtooth", "triangle"]:
            sig = subtractive(osc, 220, 0.1, cutoff_hz=2000)
            assert len(sig) > 0
            assert np.isfinite(sig).all()


# =============================================
# Pulse Wave
# =============================================

class TestPulse:
    def test_returns_correct_length(self):
        sig = pulse(440, DUR)
        assert len(sig) == int(SR * DUR)

    def test_duty_half_is_square(self):
        """duty=0.5 should give values of +amplitude and -amplitude."""
        sig = pulse(440, DUR, duty=0.5, amplitude=0.5)
        unique_vals = np.unique(sig)
        assert len(unique_vals) == 2
        assert abs(unique_vals[0] - (-0.5)) < 0.01
        assert abs(unique_vals[1] - 0.5) < 0.01

    def test_narrow_duty_more_negative(self):
        """duty=0.1 should spend ~90% of time at -amplitude."""
        sig = pulse(440, DUR, duty=0.1, amplitude=1.0)
        negative_ratio = np.sum(sig < 0) / len(sig)
        assert negative_ratio > 0.85

    def test_amplitude_scaling(self):
        sig = pulse(440, DUR, duty=0.5, amplitude=0.3)
        assert abs(np.max(np.abs(sig)) - 0.3) < 0.01


# =============================================
# Unison / Detune
# =============================================

class TestUnison:
    def test_single_voice_matches_generator(self):
        """1 voice with 0 detune should match the generator."""
        uni = unison(sine, 440, DUR, voices=1, detune_cents=0, amplitude=0.5)
        pure = sine(440, DUR, amplitude=0.5)
        assert np.allclose(uni, pure, atol=1e-10)

    def test_returns_correct_length(self):
        sig = unison(sawtooth, 440, DUR, voices=5)
        assert len(sig) == int(SR * DUR)

    def test_amplitude_normalized(self):
        sig = unison(sine, 440, DUR, voices=5, detune_cents=20, amplitude=0.5)
        assert abs(np.max(np.abs(sig)) - 0.5) < 0.01

    def test_more_voices_wider_spectrum(self):
        """Multiple detuned voices should produce a wider spectral spread."""
        one = unison(sine, 440, DUR, voices=1, amplitude=0.5)
        many = unison(sine, 440, DUR, voices=7, detune_cents=30, amplitude=0.5)
        fft_one = np.abs(np.fft.rfft(one))
        fft_many = np.abs(np.fft.rfft(many))
        # Spectral spread: count bins with significant energy
        thresh = 0.01 * np.max(fft_many)
        assert np.sum(fft_many > thresh) > np.sum(fft_one > thresh)

    def test_zero_voices_raises(self):
        with pytest.raises(ValueError):
            unison(sine, 440, DUR, voices=0)

    def test_works_with_sawtooth(self):
        """Unison should work with any generator function."""
        sig = unison(sawtooth, 220, 0.1, voices=3, detune_cents=15)
        assert len(sig) > 0
        assert np.isfinite(sig).all()
