"""Tests for audiomancer.mastering module."""

import numpy as np

from audiomancer.mastering import (
    ambient_master_chain,
    limit,
    master_chain,
    mono_bass,
    soft_clip,
)

SR = 44100
DUR = 2.0
N = int(SR * DUR)


def _stereo_signal(freq=200.0):
    """Generate a stereo sine test signal."""
    t = np.linspace(0, DUR, N, endpoint=False)
    left = np.sin(2 * np.pi * freq * t) * 0.8
    right = np.sin(2 * np.pi * freq * t + 0.3) * 0.8  # slight phase offset
    return np.column_stack([left, right])


def _mono_signal(freq=200.0):
    """Generate a mono sine test signal."""
    t = np.linspace(0, DUR, N, endpoint=False)
    return np.sin(2 * np.pi * freq * t) * 0.8


class TestMonoBass:
    def test_stereo_shape_preserved(self):
        sig = _stereo_signal()
        result = mono_bass(sig, crossover_hz=100.0, sample_rate=SR)
        assert result.shape == sig.shape

    def test_mono_passthrough(self):
        sig = _mono_signal()
        result = mono_bass(sig, sample_rate=SR)
        np.testing.assert_array_equal(result, sig)

    def test_low_freq_becomes_mono(self):
        # 50 Hz signal (below 100 Hz crossover) should be nearly mono after processing
        t = np.linspace(0, DUR, N, endpoint=False)
        left = np.sin(2 * np.pi * 50 * t) * 0.5
        right = np.sin(2 * np.pi * 50 * t + 1.0) * 0.5  # different phase
        sig = np.column_stack([left, right])

        result = mono_bass(sig, crossover_hz=100.0, sample_rate=SR)
        # Below crossover, L and R should be very similar
        diff = np.abs(result[:, 0] - result[:, 1])
        assert np.mean(diff) < 0.1

    def test_high_freq_stays_stereo(self):
        # 1000 Hz signal (above crossover) should retain stereo spread
        t = np.linspace(0, DUR, N, endpoint=False)
        left = np.sin(2 * np.pi * 1000 * t) * 0.5
        right = np.sin(2 * np.pi * 1000 * t + 1.0) * 0.5
        sig = np.column_stack([left, right])

        result = mono_bass(sig, crossover_hz=100.0, sample_rate=SR)
        diff = np.abs(result[:, 0] - result[:, 1])
        assert np.mean(diff) > 0.05  # still has stereo content


class TestSoftClip:
    def test_shape_preserved_mono(self):
        sig = _mono_signal()
        result = soft_clip(sig, threshold_db=-3.0)
        assert result.shape == sig.shape

    def test_shape_preserved_stereo(self):
        sig = _stereo_signal()
        result = soft_clip(sig, threshold_db=-3.0)
        assert result.shape == sig.shape

    def test_reduces_peaks(self):
        sig = _mono_signal() * 2.0  # hot signal
        result = soft_clip(sig, threshold_db=-6.0, drive=2.0)
        assert np.max(np.abs(result)) < np.max(np.abs(sig))

    def test_gentle_on_quiet(self):
        sig = _mono_signal() * 0.1  # quiet signal
        result = soft_clip(sig, threshold_db=-3.0, drive=1.0)
        # Quiet signals should pass through nearly unchanged
        np.testing.assert_allclose(result, sig, atol=0.02)

    def test_no_nan(self):
        sig = np.zeros(N)
        result = soft_clip(sig)
        assert np.all(np.isfinite(result))


class TestLimit:
    def test_peak_below_ceiling(self):
        sig = _mono_signal()
        result = limit(sig, ceiling_dbtp=-1.0, sample_rate=SR)
        ceiling_linear = 10 ** (-1.0 / 20)
        assert np.max(np.abs(result)) <= ceiling_linear + 0.001

    def test_stereo_peak_below_ceiling(self):
        sig = _stereo_signal()
        result = limit(sig, ceiling_dbtp=-1.0, sample_rate=SR)
        ceiling_linear = 10 ** (-1.0 / 20)
        assert np.max(np.abs(result)) <= ceiling_linear + 0.001

    def test_shape_preserved(self):
        sig = _stereo_signal()
        result = limit(sig, ceiling_dbtp=-1.0, sample_rate=SR)
        assert result.shape == sig.shape

    def test_quiet_signal_stays_below_ceiling(self):
        sig = _mono_signal() * 0.1  # well below ceiling
        result = limit(sig, ceiling_dbtp=-1.0, sample_rate=SR)
        ceiling_linear = 10 ** (-1.0 / 20)
        # Peak must not exceed ceiling regardless of limiter behavior
        assert np.max(np.abs(result)) <= ceiling_linear + 0.001


class TestMasterChain:
    def test_full_chain_stereo(self):
        sig = _stereo_signal()
        result = master_chain(sig, sample_rate=SR)
        assert result.shape == sig.shape
        # Peak should be at or below -1 dBTP
        ceiling_linear = 10 ** (-1.0 / 20)
        assert np.max(np.abs(result)) <= ceiling_linear + 0.001

    def test_full_chain_mono(self):
        sig = _mono_signal()
        result = master_chain(sig, sample_rate=SR)
        assert result.ndim == 1

    def test_no_nan_or_inf(self):
        sig = _stereo_signal()
        result = master_chain(sig, sample_rate=SR)
        assert np.all(np.isfinite(result))

    def test_subsonic_removed(self):
        # 10 Hz signal should be heavily attenuated by 30 Hz highpass
        t = np.linspace(0, DUR, N, endpoint=False)
        subsonic = np.sin(2 * np.pi * 10 * t) * 0.5
        sig = np.column_stack([subsonic, subsonic])

        result = master_chain(sig, highpass_hz=30.0, sample_rate=SR)
        # RMS should be much lower
        input_rms = np.sqrt(np.mean(sig ** 2))
        output_rms = np.sqrt(np.mean(result ** 2))
        assert output_rms < input_rms * 0.3


class TestVerifyLoop:
    def test_perfect_loop_scores_high(self):
        from audiomancer.compose import make_loopable, verify_loop
        # Constant signal = perfect loop
        sig = np.ones((N, 2)) * 0.5
        looped = make_loopable(sig, crossfade_sec=0.5, sample_rate=SR)
        score, report = verify_loop(looped, crossfade_sec=0.5, sample_rate=SR)
        assert score > 0.7

    def test_report_has_required_keys(self):
        from audiomancer.compose import make_loopable, verify_loop
        sig = np.ones((N, 2)) * 0.5
        looped = make_loopable(sig, crossfade_sec=0.5, sample_rate=SR)
        _, report = verify_loop(looped, crossfade_sec=0.5, sample_rate=SR)
        assert "level_diff_db" in report
        assert "jump_amplitude" in report
        assert "correlation" in report
        assert "overall" in report


class TestSoftClipCascade:
    """Phase A2: cascaded soft-clip adds harmonic content."""

    def _h2_over_h1(self, sig: np.ndarray, freq: float) -> float:
        """Ratio of 2nd harmonic amplitude over fundamental via FFT."""
        spectrum = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1 / SR)
        h1 = spectrum[np.argmin(np.abs(freqs - freq))]
        h2 = spectrum[np.argmin(np.abs(freqs - 2 * freq))]
        return h2 / (h1 + 1e-10)

    def test_cascade_adds_harmonics(self):
        """3-stage cascade should produce more H2 content than 1 stage."""
        # Drive signal hot so saturation actually kicks in
        t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
        sig = np.sin(2 * np.pi * 200 * t) * 1.5  # driven above threshold

        single = soft_clip(sig, threshold_db=-3.0, stages=1)
        cascade = soft_clip(sig, threshold_db=-3.0, stages=3)

        h2_single = self._h2_over_h1(single, 200.0)
        h2_cascade = self._h2_over_h1(cascade, 200.0)

        # Cascade should expose more harmonic richness
        assert h2_cascade >= h2_single, (
            f"cascade H2/H1={h2_cascade} not >= single H2/H1={h2_single}"
        )

    def test_stages_1_is_legacy(self):
        """stages=1 must match the original single-tanh behavior."""
        sig = _mono_signal() * 1.5
        legacy = soft_clip(sig, threshold_db=-3.0, stages=1)
        # Reference: one tanh, no cascade
        threshold = 10 ** (-3.0 / 20)
        ref = np.tanh(sig / threshold) * threshold
        assert np.allclose(legacy, ref, atol=1e-9)


class TestAmbientMasterChain:
    """Ambient master must preserve target LUFS (no maximizer upward-gain)."""

    def test_hits_target_lufs(self):
        import pyloudnorm as pyln
        # 3s @ 48kHz, well above pyloudnorm's 0.4s integration window
        sr = 48000
        dur = 3.0
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        sig = np.column_stack([
            np.sin(2 * np.pi * 200 * t) * 0.3,
            np.sin(2 * np.pi * 200 * t + 0.3) * 0.3,
        ])
        result = ambient_master_chain(sig, target_lufs=-20.0,
                                      ceiling_dbtp=-3.0, sample_rate=sr)
        meter = pyln.Meter(sr)
        measured = meter.integrated_loudness(result)
        assert abs(measured - (-20.0)) < 0.5, (
            f"LUFS should hit -20 ± 0.5, got {measured:.2f}"
        )

    def test_peak_below_ceiling(self):
        sr = 48000
        dur = 3.0
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        # Loud signal forces peak cap to engage
        sig = np.column_stack([np.sin(2 * np.pi * 200 * t),
                               np.sin(2 * np.pi * 200 * t)])
        result = ambient_master_chain(sig, target_lufs=-6.0,
                                      ceiling_dbtp=-3.0, sample_rate=sr)
        ceiling = 10 ** (-3.0 / 20)
        assert np.max(np.abs(result)) <= ceiling + 1e-6
