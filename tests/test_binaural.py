"""Tests for audiomancer.binaural module."""

import numpy as np
import pytest

from audiomancer.binaural import binaural, binaural_layered, from_preset, BANDS, CARRIERS, PRESETS

SR = 44100


class TestBinaural:
    def test_shape_stereo(self):
        sig = binaural(200.0, 10.0, 1.0, sample_rate=SR)
        assert sig.shape == (SR, 2)

    def test_left_right_different(self):
        sig = binaural(200.0, 10.0, 1.0, sample_rate=SR)
        assert not np.allclose(sig[:, 0], sig[:, 1])

    def test_amplitude(self):
        sig = binaural(200.0, 10.0, 1.0, amplitude=0.5, sample_rate=SR)
        assert np.max(np.abs(sig)) == pytest.approx(0.5, abs=0.01)

    def test_zero_beat_same_channels(self):
        sig = binaural(200.0, 0.0, 1.0, sample_rate=SR)
        np.testing.assert_allclose(sig[:, 0], sig[:, 1], atol=1e-10)


class TestBinauralLayered:
    def test_shape_stereo(self):
        sig = binaural_layered(200.0, 10.0, 1.0, sample_rate=SR)
        assert sig.shape == (SR, 2)

    def test_pink_amount_changes_signal(self):
        sig0 = binaural_layered(200.0, 10.0, 1.0, pink_amount=0.0, sample_rate=SR)
        sig1 = binaural_layered(200.0, 10.0, 1.0, pink_amount=0.5, sample_rate=SR)
        assert not np.allclose(sig0, sig1)


class TestConstants:
    def test_bands_exist(self):
        assert "alpha" in BANDS
        assert "theta" in BANDS

    def test_carriers_exist(self):
        assert "schumann" in CARRIERS
        assert CARRIERS["schumann"] == pytest.approx(7.83)


class TestPresets:
    def test_presets_dict(self):
        assert "theta_deep" in PRESETS
        assert "om_theta" in PRESETS
        assert PRESETS["theta_deep"]["beat"] == 4.0

    def test_from_preset(self):
        sig = from_preset("theta_deep", 1.0, sample_rate=SR)
        assert sig.shape == (SR, 2)

    def test_from_preset_invalid(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            from_preset("nonexistent", 1.0)
