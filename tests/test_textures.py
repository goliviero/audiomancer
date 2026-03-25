"""Tests for audiomancer.textures module."""

import numpy as np
import pytest

from audiomancer.textures import (
    evolving_drone, breathing_pad, deep_space, ocean_bed,
    crystal_shimmer, earth_hum, ethereal_wash, singing_bowl,
    noise_wash, generate, list_textures, REGISTRY,
)

SR = 44100
DUR = 3.0  # 3 seconds — enough to exercise modulation, fast enough for CI


class TestTextureOutputFormat:
    """All textures must produce stereo, normalized, finite signals."""

    @pytest.mark.parametrize("name", list(REGISTRY.keys()))
    def test_stereo_output(self, name):
        sig = generate(name, duration_sec=DUR, seed=42, sample_rate=SR)
        assert sig.ndim == 2, f"{name} should be stereo"
        assert sig.shape[1] == 2, f"{name} should have 2 channels"

    @pytest.mark.parametrize("name", list(REGISTRY.keys()))
    def test_no_nans_or_infs(self, name):
        sig = generate(name, duration_sec=DUR, seed=42, sample_rate=SR)
        assert np.all(np.isfinite(sig)), f"{name} contains NaN or Inf"

    @pytest.mark.parametrize("name", list(REGISTRY.keys()))
    def test_not_silent(self, name):
        sig = generate(name, duration_sec=DUR, seed=42, sample_rate=SR)
        rms = np.sqrt(np.mean(sig ** 2))
        assert rms > 0.001, f"{name} is essentially silent"


class TestEvolvingDrone:
    def test_custom_frequency(self):
        sig = evolving_drone(DUR, frequency=136.1, seed=1, sample_rate=SR)
        assert sig.shape[0] > 0

    def test_custom_harmonics(self):
        harmonics = [(1, 1.0), (2, 0.3)]
        sig = evolving_drone(DUR, harmonics=harmonics, seed=1, sample_rate=SR)
        assert sig.ndim == 2


class TestBreathingPad:
    def test_custom_chord(self):
        sig = breathing_pad(DUR, frequencies=[220.0, 277.18, 329.63],
                            seed=1, sample_rate=SR)
        assert sig.ndim == 2

    def test_default_chord(self):
        sig = breathing_pad(DUR, seed=1, sample_rate=SR)
        assert sig.shape[0] > 0


class TestNoiseWash:
    @pytest.mark.parametrize("color", ["pink", "brown", "white"])
    def test_all_colors(self, color):
        sig = noise_wash(DUR, color=color, seed=1, sample_rate=SR)
        assert sig.ndim == 2

    def test_invalid_color(self):
        with pytest.raises(ValueError, match="Unknown noise color"):
            noise_wash(DUR, color="red", seed=1, sample_rate=SR)


class TestRegistry:
    def test_generate_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown texture"):
            generate("nonexistent", duration_sec=DUR)

    def test_list_textures(self):
        textures = list_textures()
        assert len(textures) == len(REGISTRY)
        for t in textures:
            assert "name" in t
            assert "description" in t
            assert "role" in t
            assert "tonal" in t

    def test_all_registry_entries_have_fn(self):
        for name, entry in REGISTRY.items():
            assert callable(entry["fn"]), f"{name} has no callable fn"


class TestReproducibility:
    def test_same_seed_same_output(self):
        a = generate("evolving_drone", duration_sec=DUR, seed=42, sample_rate=SR)
        b = generate("evolving_drone", duration_sec=DUR, seed=42, sample_rate=SR)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate("deep_space", duration_sec=DUR, seed=1, sample_rate=SR)
        b = generate("deep_space", duration_sec=DUR, seed=2, sample_rate=SR)
        assert not np.allclose(a, b)
