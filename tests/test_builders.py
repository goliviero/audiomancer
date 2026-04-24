"""Tests for audiomancer.builders — REGISTRY + smoke tests per builder."""

import numpy as np
import pytest

from audiomancer import SAMPLE_RATE
from audiomancer.builders import REGISTRY, derived_seed

SR = SAMPLE_RATE
DUR = 1.0  # short for fast tests


class TestRegistry:
    def test_has_all_expected_keys(self):
        for name in ("pad_alive", "pendulum_bass", "arpege_bass",
                     "binaural_beat", "texture", "piano_processed"):
            assert name in REGISTRY, f"Missing builder: {name}"

    def test_all_callables(self):
        for name, fn in REGISTRY.items():
            assert callable(fn), f"Builder {name} is not callable"


class TestDerivedSeed:
    def test_deterministic(self):
        a = derived_seed(42, "pad")
        b = derived_seed(42, "pad")
        assert a == b

    def test_different_roles_differ(self):
        a = derived_seed(42, "pad")
        b = derived_seed(42, "bass")
        assert a != b

    def test_different_roots_differ(self):
        a = derived_seed(42, "pad")
        b = derived_seed(99, "pad")
        assert a != b


class TestPadAlive:
    def test_smoke_gentle(self):
        pad = REGISTRY["pad_alive"](
            duration=DUR, seed=42, sample_rate=SR,
            chord=[264.0, 396.0], intensity="gentle",
        )
        assert pad.ndim == 2
        assert pad.shape[0] == int(DUR * SR)
        assert np.max(np.abs(pad)) > 0


class TestTextureBuilder:
    def test_smoke_noise_wash(self):
        """Non-tonal preset (doesn't need chord_freqs)."""
        sig = REGISTRY["texture"](
            duration=DUR, seed=42, sample_rate=SR,
            texture_name="noise_wash",
        )
        assert sig.ndim == 2
        assert sig.shape[0] == int(DUR * SR)

    def test_smoke_breathing_pad_with_params(self):
        """Tonal preset with custom frequencies."""
        sig = REGISTRY["texture"](
            duration=DUR, seed=42, sample_rate=SR,
            texture_name="breathing_pad",
            frequencies=[264.0, 396.0],
        )
        assert sig.ndim == 2
        assert sig.shape[0] == int(DUR * SR)


class TestPianoProcessed:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            REGISTRY["piano_processed"](
                duration=DUR, seed=42, sample_rate=SR,
                source_path="samples/own/definitely_not_there.wav",
                preset="mid_pad",
            )

    def test_smoke_with_temp_wav(self, tmp_path):
        """Generate a short sine, save, feed through the builder."""
        from audiomancer.synth import sine
        from audiomancer.utils import export_wav
        src = sine(440.0, 2.0, amplitude=0.3, sample_rate=SR)
        path = tmp_path / "piano_test.wav"
        export_wav(src, path, sample_rate=SR)

        out = REGISTRY["piano_processed"](
            duration=DUR, seed=42, sample_rate=SR,
            source_path=str(path),
            preset="mid_pad",
        )
        assert out.ndim == 2
        assert np.max(np.abs(out)) > 0


class TestMorphTextures:
    def test_registry_has_morph(self):
        assert "morph_textures" in REGISTRY

    def test_smoke(self):
        out = REGISTRY["morph_textures"](
            duration=DUR, seed=42, sample_rate=SR,
            texture_a={"name": "noise_wash", "params": {}},
            texture_b={"name": "noise_wash", "params": {}},
        )
        assert out.ndim == 2
        assert np.max(np.abs(out)) > 0


class TestInstrumentBuilders:
    def test_registry(self):
        assert "instrument_synth" in REGISTRY
        assert "instrument_sampled" in REGISTRY

    def test_synth_didgeridoo(self):
        out = REGISTRY["instrument_synth"](
            duration=0.5, seed=42, sample_rate=SR,
            name="didgeridoo", frequency=73.0,
        )
        assert out.ndim == 2
        assert np.max(np.abs(out)) > 0

    def test_synth_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown instrument"):
            REGISTRY["instrument_synth"](
                duration=0.5, seed=42, sample_rate=SR,
                name="bagpipes",
            )

    def test_sampled_requires_file(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            REGISTRY["instrument_sampled"](
                duration=0.5, seed=42, sample_rate=SR,
                source_path="samples/own/nope.wav",
                source_hz=220.0, target_hz=330.0, mode="note",
            )

    def test_sampled_note_mode(self, tmp_path):
        from audiomancer.synth import sine
        from audiomancer.utils import export_wav

        src = sine(220.0, 1.0, sample_rate=SR)
        path = tmp_path / "inst.wav"
        export_wav(src, path, sample_rate=SR)

        out = REGISTRY["instrument_sampled"](
            duration=0.5, seed=42, sample_rate=SR,
            source_path=str(path),
            source_hz=220.0, target_hz=440.0, mode="note",
        )
        assert out.ndim == 2
        assert np.max(np.abs(out)) > 0
