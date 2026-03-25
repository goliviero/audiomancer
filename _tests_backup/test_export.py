"""Tests for fractal.export — file export functions."""

import numpy as np
import soundfile as sf
from pathlib import Path

from fractal.constants import SAMPLE_RATE
from fractal.export import export_wav, export_flac, export_auto
from fractal.generators import sine, binaural


class TestExportWav:
    def test_creates_file(self, tmp_path):
        sig = sine(440, 0.5)
        path = tmp_path / "test.wav"
        result = export_wav(sig, path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        sig = sine(440, 0.1)
        path = tmp_path / "sub" / "dir" / "test.wav"
        result = export_wav(sig, path)
        assert result.exists()

    def test_roundtrip_mono(self, tmp_path):
        sig = sine(440, 0.5)
        path = tmp_path / "mono.wav"
        export_wav(sig, path)
        data, sr = sf.read(str(path))
        assert sr == SAMPLE_RATE
        assert data.ndim == 1
        assert len(data) == len(sig)

    def test_roundtrip_stereo(self, tmp_path):
        sig = binaural(200, 10, 0.5)
        path = tmp_path / "stereo.wav"
        export_wav(sig, path)
        data, sr = sf.read(str(path))
        assert sr == SAMPLE_RATE
        assert data.shape == sig.shape

    def test_24bit(self, tmp_path):
        sig = sine(440, 0.1)
        path = tmp_path / "24bit.wav"
        export_wav(sig, path, bit_depth=24)
        info = sf.info(str(path))
        assert info.subtype == "PCM_24"

    def test_invalid_bit_depth(self, tmp_path):
        sig = sine(440, 0.1)
        path = tmp_path / "bad.wav"
        import pytest
        with pytest.raises(ValueError, match="Unsupported bit depth"):
            export_wav(sig, path, bit_depth=8)


class TestExportFlac:
    def test_creates_file(self, tmp_path):
        sig = sine(440, 0.5)
        path = tmp_path / "test.flac"
        result = export_flac(sig, path)
        assert result.exists()

    def test_roundtrip(self, tmp_path):
        sig = sine(440, 0.5)
        path = tmp_path / "test.flac"
        export_flac(sig, path)
        data, sr = sf.read(str(path))
        assert sr == SAMPLE_RATE
        assert len(data) == len(sig)


class TestExportAuto:
    def test_wav_extension(self, tmp_path):
        sig = sine(440, 0.1)
        path = tmp_path / "auto.wav"
        result = export_auto(sig, path)
        assert result.exists()

    def test_flac_extension(self, tmp_path):
        sig = sine(440, 0.1)
        path = tmp_path / "auto.flac"
        result = export_auto(sig, path)
        assert result.exists()

    def test_unsupported_extension(self, tmp_path):
        sig = sine(440, 0.1)
        path = tmp_path / "auto.mp3"
        import pytest
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_auto(sig, path)
