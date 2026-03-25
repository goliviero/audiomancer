"""Tests for fractal.presets -- SynthPreset, preset library, get_preset."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.presets import (
    SynthPreset, SYNTH_PRESETS, DRUM_PRESETS,
    get_preset, list_presets,
)


SR = SAMPLE_RATE
DUR = 0.5  # short duration for fast tests


# =============================================
# SynthPreset.render
# =============================================

class TestSynthPresetRender:
    def test_render_with_note_name(self):
        preset = get_preset("pluck")
        sig = preset.render("C4", DUR)
        assert len(sig) > 0
        assert np.isfinite(sig).all()

    def test_render_with_frequency(self):
        preset = get_preset("pluck")
        sig = preset.render(440.0, DUR)
        assert len(sig) > 0
        assert np.isfinite(sig).all()

    def test_amplitude_respected(self):
        preset = get_preset("sub_bass")
        sig = preset.render("A2", DUR, amplitude=0.3)
        assert abs(np.max(np.abs(sig)) - 0.3) < 0.02

    def test_returns_correct_length(self):
        preset = get_preset("pluck")
        sig = preset.render("A4", DUR)
        assert len(sig) == int(SR * DUR)


# =============================================
# All presets smoke test
# =============================================

class TestAllPresets:
    def test_all_presets_render(self):
        """Every preset should produce a valid signal."""
        for name, preset in SYNTH_PRESETS.items():
            sig = preset.render("A3", 0.3, amplitude=0.3)
            assert len(sig) > 0, f"Preset '{name}' produced empty signal"
            assert np.isfinite(sig).all(), f"Preset '{name}' has NaN/Inf"

    def test_all_presets_have_required_fields(self):
        for name, preset in SYNTH_PRESETS.items():
            assert preset.name, f"Preset '{name}' missing name"
            assert preset.category, f"Preset '{name}' missing category"
            assert preset.description, f"Preset '{name}' missing description"
            assert preset.synth_type, f"Preset '{name}' missing synth_type"

    def test_preset_categories_valid(self):
        valid = {"pad", "lead", "bass", "key", "texture", "fx"}
        for name, preset in SYNTH_PRESETS.items():
            assert preset.category in valid, \
                f"Preset '{name}' has invalid category '{preset.category}'"


# =============================================
# get_preset
# =============================================

class TestGetPreset:
    def test_exact_match(self):
        preset = get_preset("blade_runner_pad")
        assert preset.name == "blade_runner_pad"

    def test_partial_match(self):
        preset = get_preset("blade")
        assert preset.name == "blade_runner_pad"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent_preset_xyz")

    def test_ambiguous_raises(self):
        """Multiple matches should raise."""
        with pytest.raises(ValueError):
            get_preset("pad")  # matches multiple *_pad presets


# =============================================
# list_presets
# =============================================

class TestListPresets:
    def test_list_all(self):
        all_presets = list_presets()
        assert len(all_presets) == len(SYNTH_PRESETS)

    def test_list_by_category(self):
        pads = list_presets("pad")
        assert len(pads) > 0
        for name in pads:
            assert SYNTH_PRESETS[name].category == "pad"

    def test_list_empty_category(self):
        result = list_presets("nonexistent")
        assert result == []


# =============================================
# Drum presets
# =============================================

class TestDrumPresets:
    def test_all_drum_presets_listed(self):
        assert "808" in DRUM_PRESETS
        assert "909" in DRUM_PRESETS
        assert "acoustic" in DRUM_PRESETS
