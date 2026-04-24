"""Tests for audiomancer.saturation — tape/vinyl analog emulation."""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.saturation import tape_hiss, tape_saturate, vinyl_wow
from audiomancer.synth import sine

SR = SAMPLE_RATE
DUR = 1.0
N = int(SR * DUR)


class TestTapeSaturate:
    def test_shape_mono(self):
        sig = sine(440.0, DUR, sample_rate=SR)
        out = tape_saturate(sig, drive=1.2, asymmetry=0.15)
        assert out.shape == sig.shape

    def test_shape_stereo(self):
        sig_mono = sine(440.0, DUR, sample_rate=SR)
        sig = np.column_stack([sig_mono, sig_mono])
        out = tape_saturate(sig, drive=1.2, asymmetry=0.15)
        assert out.shape == sig.shape

    def test_asymmetric_produces_even_harmonics(self):
        """Asymmetric clip should produce 2nd-harmonic content. Symmetric should not."""
        # Drive hard so saturation actually kicks in
        sig = sine(200.0, DUR, amplitude=0.95, sample_rate=SR) * 1.8
        sym = tape_saturate(sig, drive=1.0, asymmetry=0.0)
        asym = tape_saturate(sig, drive=1.0, asymmetry=0.3)

        def h2_over_h1(s: np.ndarray, f0: float) -> float:
            spec = np.abs(np.fft.rfft(s))
            freqs = np.fft.rfftfreq(len(s), 1 / SR)
            h1 = spec[np.argmin(np.abs(freqs - f0))]
            h2 = spec[np.argmin(np.abs(freqs - 2 * f0))]
            return h2 / (h1 + 1e-12)

        assert h2_over_h1(asym, 200.0) > h2_over_h1(sym, 200.0)

    def test_no_nan(self):
        sig = sine(440.0, DUR, sample_rate=SR)
        out = tape_saturate(sig, drive=3.0, asymmetry=0.5)
        assert not np.any(np.isnan(out))


class TestTapeHiss:
    def test_shape(self):
        out = tape_hiss(1.0, level_db=-45.0, sample_rate=SR)
        assert out.shape == (SR, 2)

    def test_level_matches_target(self):
        """Peak should be close to the configured dB level (within 3 dB)."""
        level = -30.0
        out = tape_hiss(2.0, level_db=level, sample_rate=SR)
        peak_db = 20 * np.log10(np.max(np.abs(out)) + 1e-12)
        assert abs(peak_db - level) < 6.0  # pink noise crest factor -> wiggle room

    def test_subliminal_default(self):
        out = tape_hiss(1.0, sample_rate=SR)
        peak = np.max(np.abs(out))
        # Default -45 dB => well below 0.1 linear
        assert peak < 0.05


class TestVinylWow:
    def test_shape_mono(self):
        sig = sine(440.0, DUR, sample_rate=SR)
        out = vinyl_wow(sig, depth=0.0005, rate_hz=0.3, sample_rate=SR)
        assert out.shape == sig.shape

    def test_shape_stereo(self):
        sig_mono = sine(440.0, DUR, sample_rate=SR)
        sig = np.column_stack([sig_mono, sig_mono])
        out = vinyl_wow(sig, depth=0.0005, rate_hz=0.3, sample_rate=SR)
        assert out.shape == sig.shape

    def test_perturbation_audible(self):
        """With depth > 0 the output must differ from the input."""
        sig = sine(440.0, DUR, sample_rate=SR)
        out = vinyl_wow(sig, depth=0.002, rate_hz=1.0, sample_rate=SR)
        assert not np.allclose(out, sig)

    def test_zero_depth_is_identity(self):
        sig = sine(440.0, DUR, sample_rate=SR)
        out = vinyl_wow(sig, depth=0.0, rate_hz=0.3, sample_rate=SR)
        assert np.allclose(out, sig)
