"""Tests for audiomancer.effects module."""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.effects import (
    _process_board,
    chain,
    chorus,
    chorus_subtle,
    compress,
    delay,
    delay_long,
    highpass,
    lowpass,
    reverb,
    reverb_cathedral,
    reverb_hall,
)

SR = SAMPLE_RATE


def _tone(freq_hz=440.0, duration=0.5, sr=SR):
    """Generate a simple sine tone for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t)


def _stereo_tone(freq_hz=440.0, duration=0.5, sr=SR):
    """Generate a stereo sine tone."""
    mono = _tone(freq_hz, duration, sr)
    return np.column_stack([mono, mono])


# ---------------------------------------------------------------------------
# Scipy filters
# ---------------------------------------------------------------------------

class TestLowpass:
    def test_reduces_highs(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        low = np.sin(2 * np.pi * 100 * t)
        high = np.sin(2 * np.pi * 8000 * t)
        mixed = low + high
        filtered = lowpass(mixed, cutoff_hz=500, sample_rate=SR)
        fft_orig = np.abs(np.fft.rfft(mixed))
        fft_filt = np.abs(np.fft.rfft(filtered))
        high_bin = int(8000 * len(t) / SR)
        assert fft_filt[high_bin] < fft_orig[high_bin] * 0.1

    def test_stereo(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        mono = np.sin(2 * np.pi * 100 * t)
        stereo = np.column_stack([mono, mono])
        filtered = lowpass(stereo, cutoff_hz=500, sample_rate=SR)
        assert filtered.shape == (SR, 2)

    def test_preserves_length(self):
        sig = _tone(200.0, 1.0)
        out = lowpass(sig, cutoff_hz=1000)
        assert len(out) == len(sig)


class TestHighpass:
    def test_reduces_lows(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        low = np.sin(2 * np.pi * 50 * t)
        high = np.sin(2 * np.pi * 5000 * t)
        mixed = low + high
        filtered = highpass(mixed, cutoff_hz=1000, sample_rate=SR)
        fft_orig = np.abs(np.fft.rfft(mixed))
        fft_filt = np.abs(np.fft.rfft(filtered))
        low_bin = int(50 * len(t) / SR)
        assert fft_filt[low_bin] < fft_orig[low_bin] * 0.1

    def test_stereo(self):
        stereo = _stereo_tone(200.0, 0.5)
        filtered = highpass(stereo, cutoff_hz=100)
        assert filtered.shape == stereo.shape


# ---------------------------------------------------------------------------
# Pedalboard effects
# ---------------------------------------------------------------------------

class TestReverb:
    def test_returns_same_length(self):
        sig = _tone(440.0, 0.5)
        out = reverb(sig, room_size=0.5, wet_level=0.3)
        assert len(out) == len(sig)

    def test_stereo_input(self):
        sig = _stereo_tone(440.0, 0.5)
        out = reverb(sig, room_size=0.5, wet_level=0.3)
        assert out.shape == sig.shape

    def test_wet_level_affects_output(self):
        sig = _tone(440.0, 0.3)
        dry = reverb(sig, room_size=0.5, wet_level=0.0)
        wet = reverb(sig, room_size=0.5, wet_level=1.0)
        # With full wet, output should differ significantly from dry
        diff = np.mean(np.abs(wet - dry))
        assert diff > 0.001

    def test_output_dtype(self):
        sig = _tone(440.0, 0.3)
        out = reverb(sig)
        assert out.dtype == np.float64


class TestDelay:
    def test_returns_same_length(self):
        sig = _tone(440.0, 0.5)
        out = delay(sig, delay_seconds=0.1, feedback=0.2)
        assert len(out) == len(sig)

    def test_stereo_input(self):
        sig = _stereo_tone(440.0, 0.5)
        out = delay(sig, delay_seconds=0.1)
        assert out.shape == sig.shape

    def test_delay_adds_energy(self):
        # A short impulse followed by silence — delay should add echoes
        sig = np.zeros(SR)
        sig[:100] = 1.0
        out = delay(sig, delay_seconds=0.1, feedback=0.5, mix=0.5)
        # Energy after the initial impulse should be nonzero
        tail_energy = np.sum(np.abs(out[SR // 2:]))
        assert tail_energy > 0.01


class TestChorus:
    def test_returns_same_length(self):
        sig = _tone(440.0, 0.5)
        out = chorus(sig, rate_hz=1.0, depth=0.25)
        assert len(out) == len(sig)

    def test_stereo_input(self):
        sig = _stereo_tone(440.0, 0.5)
        out = chorus(sig, rate_hz=1.0)
        assert out.shape == sig.shape


class TestCompress:
    def test_returns_same_length(self):
        sig = _tone(440.0, 0.5)
        out = compress(sig, threshold_db=-20.0, ratio=4.0)
        assert len(out) == len(sig)

    def test_reduces_dynamic_range(self):
        # Loud signal should be compressed
        sig = _tone(440.0, 0.5) * 0.9
        out = compress(sig, threshold_db=-6.0, ratio=10.0)
        assert np.max(np.abs(out)) <= np.max(np.abs(sig))

    def test_stereo_input(self):
        sig = _stereo_tone(440.0, 0.5) * 0.8
        out = compress(sig, threshold_db=-10.0)
        assert out.shape == sig.shape


class TestChain:
    def test_empty_chain(self):
        sig = _tone(440.0, 0.3)
        out = chain(sig, effects=[])
        assert len(out) == len(sig)

    def test_multiple_effects(self):
        import pedalboard as pb
        sig = _tone(440.0, 0.3)
        effects = [
            pb.Reverb(room_size=0.3, wet_level=0.2),
            pb.Compressor(threshold_db=-20.0),
        ]
        out = chain(sig, effects)
        assert len(out) == len(sig)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def test_reverb_hall(self):
        sig = _tone(440.0, 0.3)
        out = reverb_hall(sig)
        assert len(out) == len(sig)
        assert out.dtype == np.float64

    def test_reverb_cathedral(self):
        sig = _tone(440.0, 0.3)
        out = reverb_cathedral(sig)
        assert len(out) == len(sig)

    def test_delay_long(self):
        sig = _tone(440.0, 0.3)
        out = delay_long(sig)
        assert len(out) == len(sig)

    def test_chorus_subtle(self):
        sig = _tone(440.0, 0.3)
        out = chorus_subtle(sig)
        assert len(out) == len(sig)

    def test_presets_stereo(self):
        sig = _stereo_tone(440.0, 0.3)
        for fn in [reverb_hall, reverb_cathedral, delay_long, chorus_subtle]:
            out = fn(sig)
            assert out.shape == sig.shape, f"{fn.__name__} changed shape"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class TestProcessBoard:
    def test_mono_roundtrip_shape(self):
        import pedalboard as pb
        board = pb.Pedalboard([])
        sig = _tone(440.0, 0.3)
        out = _process_board(board, sig, SR)
        assert out.ndim == 1
        assert len(out) == len(sig)

    def test_stereo_roundtrip_shape(self):
        import pedalboard as pb
        board = pb.Pedalboard([])
        sig = _stereo_tone(440.0, 0.3)
        out = _process_board(board, sig, SR)
        assert out.ndim == 2
        assert out.shape == sig.shape
