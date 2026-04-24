"""Tests for audiomancer.instruments — 5 synthetic ethnic instruments."""

import numpy as np
import pytest

from audiomancer import SAMPLE_RATE
from audiomancer.instruments import (
    derbouka_hit,
    derbouka_pattern,
    didgeridoo,
    handpan,
    oud,
    sitar,
)

SR = SAMPLE_RATE


class TestDidgeridoo:
    def test_shape(self):
        sig = didgeridoo(73.0, 0.5, sample_rate=SR)
        assert sig.shape == (int(SR * 0.5),)

    def test_deterministic_with_seed(self):
        a = didgeridoo(73.0, 0.3, seed=42, sample_rate=SR)
        b = didgeridoo(73.0, 0.3, seed=42, sample_rate=SR)
        assert np.allclose(a, b)

    def test_non_silent(self):
        sig = didgeridoo(73.0, 0.3, seed=42, sample_rate=SR)
        assert np.max(np.abs(sig)) > 0.1


class TestHandpan:
    def test_shape(self):
        sig = handpan(146.83, 0.5, sample_rate=SR)
        assert sig.shape == (int(SR * 0.5),)

    def test_decay_is_exponential(self):
        sig = handpan(146.83, 1.0, decay=0.999, seed=42, sample_rate=SR)
        # Tail RMS should be smaller than head
        head = np.sqrt(np.mean(sig[:int(SR * 0.1)] ** 2))
        tail = np.sqrt(np.mean(sig[-int(SR * 0.1):] ** 2))
        assert tail < head

    def test_inharmonicity_changes_timbre(self):
        a = handpan(146.83, 0.5, inharmonicity=0.0, seed=42, sample_rate=SR)
        b = handpan(146.83, 0.5, inharmonicity=0.15, seed=42, sample_rate=SR)
        assert not np.allclose(a, b)


class TestOud:
    def test_shape(self):
        sig = oud(146.83, 0.5, sample_rate=SR)
        assert sig.shape == (int(SR * 0.5),)

    def test_body_resonance_affects_tone(self):
        a = oud(146.83, 0.5, body_resonance_hz=300, seed=42, sample_rate=SR)
        b = oud(146.83, 0.5, body_resonance_hz=600, seed=42, sample_rate=SR)
        assert not np.allclose(a, b)


class TestSitar:
    def test_shape(self):
        sig = sitar(196.0, 0.5, sample_rate=SR)
        assert sig.shape == (int(SR * 0.5),)

    def test_buzz_amount_zero_is_plain_karplus_ish(self):
        clean = sitar(196.0, 0.5, buzz_amount=0.0,
                      sympathetic_strings=False, seed=42, sample_rate=SR)
        buzzy = sitar(196.0, 0.5, buzz_amount=0.8,
                      sympathetic_strings=False, seed=42, sample_rate=SR)
        assert not np.allclose(clean, buzzy)


class TestDerbouka:
    def test_hit_dum(self):
        sig = derbouka_hit("dum", duration_sec=0.3, sample_rate=SR)
        assert sig.shape == (int(SR * 0.3),)
        assert np.max(np.abs(sig)) > 0

    def test_hit_tek(self):
        sig = derbouka_hit("tek", duration_sec=0.3, sample_rate=SR)
        assert sig.shape == (int(SR * 0.3),)

    def test_invalid_hit_type(self):
        with pytest.raises(ValueError):
            derbouka_hit("plop", duration_sec=0.3, sample_rate=SR)

    def test_pattern_respects_duration(self):
        sig = derbouka_pattern("D t t D", bpm=120, duration_sec=2.0,
                               seed=42, sample_rate=SR)
        assert sig.shape == (int(2 * SR),)

    def test_pattern_empty_raises(self):
        with pytest.raises(ValueError):
            derbouka_pattern("", bpm=120)

    def test_pattern_default_duration(self):
        # "D t t D" = 4 eighth notes @ 120 bpm = 4 * 0.25 = 1.0s
        sig = derbouka_pattern("D t t D", bpm=120, seed=42, sample_rate=SR)
        assert abs(sig.shape[0] - SR) < SR * 0.05
