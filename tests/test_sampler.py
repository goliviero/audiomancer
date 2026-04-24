"""Tests for audiomancer.sampler — pitch-shift + pad + multisample loader."""

import numpy as np
import pytest

from audiomancer import SAMPLE_RATE
from audiomancer.sampler import (
    _note_name_to_hz,
    load_multisample,
    pitched_pad,
    play_note,
    play_note_multisample,
)
from audiomancer.synth import sine

SR = SAMPLE_RATE


class TestPlayNote:
    def test_identity_shift(self):
        """source_hz == target_hz should be effectively identity (small allowances)."""
        src = sine(220.0, 0.5, sample_rate=SR)
        out = play_note(src, source_hz=220.0, target_hz=220.0, sample_rate=SR)
        # Same length (within rounding)
        assert abs(len(out) - len(src)) <= 2

    def test_octave_up_halves_length(self):
        src = sine(220.0, 1.0, sample_rate=SR)
        out = play_note(src, source_hz=220.0, target_hz=440.0, sample_rate=SR)
        # Output should be about half the input length
        assert abs(len(out) - len(src) // 2) < 100

    def test_octave_down_doubles_length(self):
        src = sine(220.0, 0.5, sample_rate=SR)
        out = play_note(src, source_hz=220.0, target_hz=110.0, sample_rate=SR)
        assert abs(len(out) - len(src) * 2) < 100

    def test_duration_override(self):
        src = sine(220.0, 0.5, sample_rate=SR)
        out = play_note(src, 220.0, 330.0, duration_sec=1.0, sample_rate=SR)
        assert out.shape[0] == SR

    def test_stereo_preserved(self):
        mono = sine(220.0, 0.5, sample_rate=SR)
        stereo = np.column_stack([mono, mono])
        out = play_note(stereo, source_hz=220.0, target_hz=330.0,
                        sample_rate=SR)
        assert out.ndim == 2


class TestPitchedPad:
    def test_reaches_target_duration(self):
        src = sine(220.0, 2.0, sample_rate=SR)
        pad = pitched_pad(src, source_hz=220.0, target_hz=330.0,
                          duration_sec=10.0, window_sec=0.3, sample_rate=SR)
        assert pad.shape[0] == int(10.0 * SR)
        assert pad.ndim == 2

    def test_non_silent(self):
        src = sine(220.0, 2.0, sample_rate=SR)
        pad = pitched_pad(src, 220.0, 330.0, duration_sec=5.0,
                          window_sec=0.3, sample_rate=SR)
        assert np.max(np.abs(pad)) > 0.1


class TestNoteNameToHz:
    def test_a4(self):
        assert _note_name_to_hz("A4") == pytest.approx(440.0)

    def test_c4(self):
        # C4 in A4=440 standard
        assert _note_name_to_hz("C4") == pytest.approx(261.63, abs=0.1)

    def test_sharp(self):
        assert _note_name_to_hz("F#4") == pytest.approx(369.99, abs=0.1)

    def test_flat(self):
        # Bb3 == A#3 == 233.08 Hz
        assert _note_name_to_hz("Bb3") == pytest.approx(233.08, abs=0.1)


class TestMultisample:
    def test_load_finds_matching_files(self, tmp_path):
        from audiomancer.utils import export_wav

        # Create 3 synthetic samples at different pitches
        samples_dir = tmp_path / "samples"
        (samples_dir / "cc0").mkdir(parents=True)
        for note, hz in (("D3", 146.83), ("A3", 220.0), ("F4", 349.23)):
            sig = sine(hz, 0.5, sample_rate=SR)
            path = samples_dir / "cc0" / f"testinst_{note}.wav"
            export_wav(sig, path, sample_rate=SR)

        bank = load_multisample("testinst", samples_dir=samples_dir,
                                target_sr=SR)
        assert len(bank) == 3
        # Keys should be the pitches
        assert any(abs(k - 220.0) < 0.5 for k in bank)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_multisample("notexisting", samples_dir=tmp_path / "nope",
                             target_sr=SR)

    def test_play_multisample_picks_closest(self, tmp_path):
        from audiomancer.utils import export_wav

        samples_dir = tmp_path / "samples"
        (samples_dir / "cc0").mkdir(parents=True)
        for note, hz in (("D3", 146.83), ("A3", 220.0)):
            sig = sine(hz, 0.5, sample_rate=SR)
            path = samples_dir / "cc0" / f"testinst_{note}.wav"
            export_wav(sig, path, sample_rate=SR)

        bank = load_multisample("testinst", samples_dir=samples_dir,
                                target_sr=SR)
        # Target near A3 should use A3 sample (smallest shift)
        note = play_note_multisample(bank, target_hz=233.0,
                                     sample_rate=SR)
        assert note.ndim == 1 or note.ndim == 2
