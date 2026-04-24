"""Tests for audiomancer.viz — waveform + spectrum PNG rendering.

Skipped if matplotlib is not installed.
"""

import pytest

pytest.importorskip("matplotlib")

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.synth import sine
from audiomancer.viz import plot_spectrum, plot_stem, plot_waveform

SR = SAMPLE_RATE


def test_plot_waveform_creates_png(tmp_path):
    sig = sine(440.0, 0.5, sample_rate=SR)
    path = tmp_path / "wave.png"
    out = plot_waveform(sig, path, sample_rate=SR, title="test")
    assert out == path
    assert path.exists()
    # Basic PNG magic bytes check
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_spectrum_creates_png(tmp_path):
    sig = sine(440.0, 0.5, sample_rate=SR)
    path = tmp_path / "spec.png"
    plot_spectrum(sig, path, sample_rate=SR)
    assert path.exists()


def test_plot_stem_creates_png(tmp_path):
    mono = sine(440.0, 0.5, sample_rate=SR)
    sig = np.column_stack([mono, mono])
    path = tmp_path / "stem.png"
    plot_stem(sig, path, sample_rate=SR, title="stereo stem")
    assert path.exists()
