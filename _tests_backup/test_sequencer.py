"""Tests for fractal.sequencer — Clip, Pattern, Sequencer."""

import numpy as np
import pytest

from fractal.constants import SAMPLE_RATE
from fractal.generators import sine
from fractal.signal import is_stereo, is_mono, mono_to_stereo
from fractal.sequencer import Clip, Pattern, Sequencer


class TestClip:
    def test_defaults(self):
        sig = np.zeros(100)
        clip = Clip(signal=sig)
        assert clip.start_sec == 0.0
        assert clip.track_name is None

    def test_with_offset(self):
        clip = Clip(signal=np.zeros(100), start_sec=2.5, track_name="kick")
        assert clip.start_sec == 2.5
        assert clip.track_name == "kick"


class TestPattern:
    def test_empty_render(self):
        pat = Pattern(duration_sec=2.0)
        result = pat.render()
        assert result.shape[0] == int(2.0 * SAMPLE_RATE)
        assert np.max(np.abs(result)) == 0.0

    def test_single_clip(self):
        """A clip at t=0 should be placed at the start."""
        sig = np.ones(1000)
        pat = Pattern(duration_sec=1.0)
        pat.add_clip(sig, start_sec=0.0)
        result = pat.render()
        assert result[0] == 1.0
        assert result[999] == 1.0
        assert result[1000] == 0.0

    def test_clip_at_offset(self):
        """A clip placed at t=0.5 should start at sample 22050."""
        sig = np.ones(100)
        pat = Pattern(duration_sec=2.0)
        pat.add_clip(sig, start_sec=0.5)
        result = pat.render()
        start = int(0.5 * SAMPLE_RATE)
        assert result[start] == 1.0
        assert result[start - 1] == 0.0

    def test_overlapping_clips_sum(self):
        """Two clips at the same position should sum."""
        sig_a = np.ones(500) * 0.5
        sig_b = np.ones(500) * 0.3
        pat = Pattern(duration_sec=0.5)
        pat.add_clip(sig_a)
        pat.add_clip(sig_b)
        result = pat.render()
        assert abs(result[0] - 0.8) < 0.001

    def test_clip_truncated_at_duration(self):
        """Clip extending past pattern duration should be truncated."""
        sig = np.ones(SAMPLE_RATE * 5)  # 5 seconds
        pat = Pattern(duration_sec=1.0)
        pat.add_clip(sig)
        result = pat.render()
        assert result.shape[0] == SAMPLE_RATE  # 1 second

    def test_repeat(self):
        """Pattern.repeat(3) should triple the duration and replicate clips."""
        sig = np.ones(100)
        pat = Pattern(duration_sec=1.0)
        pat.add_clip(sig)
        repeated = pat.repeat(3)
        assert repeated.duration_sec == 3.0
        assert len(repeated.clips) == 3
        # Check offsets
        assert repeated.clips[0].start_sec == 0.0
        assert repeated.clips[1].start_sec == 1.0
        assert repeated.clips[2].start_sec == 2.0

    def test_repeat_render(self):
        """Repeated pattern should have signal at each repetition start."""
        sig = np.ones(100)
        pat = Pattern(duration_sec=0.5)
        pat.add_clip(sig)
        repeated = pat.repeat(4)
        result = repeated.render()
        sr = SAMPLE_RATE
        # Signal at start of each repetition
        for i in range(4):
            idx = int(i * 0.5 * sr)
            assert result[idx] == 1.0

    def test_add_clip_returns_clip(self):
        pat = Pattern(duration_sec=1.0)
        clip = pat.add_clip(np.zeros(100), track_name="hi-hat")
        assert clip.track_name == "hi-hat"

    def test_stereo_clips(self):
        """Pattern with stereo clips should produce stereo output."""
        sig = mono_to_stereo(np.ones(100))
        pat = Pattern(duration_sec=0.5)
        pat.add_clip(sig)
        result = pat.render()
        assert is_stereo(result)

    def test_mixed_mono_stereo(self):
        """Mix of mono and stereo clips should produce stereo."""
        mono = np.ones(100)
        stereo = mono_to_stereo(np.ones(100) * 0.5)
        pat = Pattern(duration_sec=0.5)
        pat.add_clip(mono)
        pat.add_clip(stereo)
        result = pat.render()
        assert is_stereo(result)


class TestSequencer:
    def test_beats_to_sec_120bpm(self):
        """At 120 BPM, 1 beat = 0.5 seconds."""
        seq = Sequencer(tempo_bpm=120)
        assert seq.beats_to_sec(1.0) == 0.5
        assert seq.beats_to_sec(4.0) == 2.0

    def test_beats_to_sec_60bpm(self):
        """At 60 BPM, 1 beat = 1.0 second."""
        seq = Sequencer(tempo_bpm=60)
        assert seq.beats_to_sec(1.0) == 1.0

    def test_sec_to_beats(self):
        seq = Sequencer(tempo_bpm=120)
        assert seq.sec_to_beats(0.5) == 1.0
        assert seq.sec_to_beats(2.0) == 4.0

    def test_empty_render(self):
        seq = Sequencer()
        result = seq.render()
        assert result.shape[0] == 0

    def test_single_clip_at_beat_0(self):
        """Clip at beat 0 should start at sample 0."""
        seq = Sequencer(tempo_bpm=120)
        sig = np.ones(1000)
        seq.add_clip(sig, start_beat=0)
        result = seq.render()
        assert result[0] == 1.0
        assert result[999] == 1.0

    def test_clip_at_beat_2(self):
        """At 120 BPM, beat 2 = 1.0 second."""
        seq = Sequencer(tempo_bpm=120)
        sig = np.ones(100)
        seq.add_clip(sig, start_beat=2)
        result = seq.render()
        start = int(1.0 * SAMPLE_RATE)
        assert result[start] == 1.0
        assert result[start - 1] == 0.0

    def test_add_clip_sec(self):
        """add_clip_sec should place at absolute time."""
        seq = Sequencer(tempo_bpm=120)
        sig = np.ones(100)
        seq.add_clip_sec(sig, start_sec=0.5)
        result = seq.render()
        start = int(0.5 * SAMPLE_RATE)
        assert result[start] == 1.0

    def test_add_pattern(self):
        """add_pattern should insert all pattern clips at the beat offset."""
        pat = Pattern(duration_sec=1.0)
        pat.add_clip(np.ones(100), start_sec=0.0)
        pat.add_clip(np.ones(100), start_sec=0.5)

        seq = Sequencer(tempo_bpm=120)
        seq.add_pattern(pat, start_beat=0)
        result = seq.render()

        assert result[0] == 1.0
        assert result[int(0.5 * SAMPLE_RATE)] == 1.0

    def test_add_pattern_with_offset(self):
        """Pattern at beat 4 (= 2s at 120BPM) should offset all clips."""
        pat = Pattern(duration_sec=1.0)
        pat.add_clip(np.ones(100), start_sec=0.0)

        seq = Sequencer(tempo_bpm=120)
        seq.add_pattern(pat, start_beat=4)
        result = seq.render()

        start = int(2.0 * SAMPLE_RATE)
        assert result[start] == 1.0
        assert result[0] == 0.0

    def test_duration_sec(self):
        """duration_sec should reflect the end of the last clip."""
        seq = Sequencer(tempo_bpm=120)
        sig = np.ones(SAMPLE_RATE)  # 1 second
        seq.add_clip(sig, start_beat=0)  # ends at 1.0s
        assert abs(seq.duration_sec - 1.0) < 0.01

    def test_duration_beats(self):
        seq = Sequencer(tempo_bpm=120)
        sig = np.ones(SAMPLE_RATE)  # 1 second = 2 beats at 120 BPM
        seq.add_clip(sig, start_beat=0)
        assert abs(seq.duration_beats - 2.0) < 0.02

    def test_tail_sec(self):
        """render(tail_sec=1.0) should add silence at the end."""
        seq = Sequencer(tempo_bpm=120)
        sig = np.ones(100)
        seq.add_clip(sig, start_beat=0)
        result = seq.render(tail_sec=1.0)
        expected_len = 100 + int(1.0 * SAMPLE_RATE)
        assert result.shape[0] == expected_len

    def test_overlapping_clips(self):
        """Two clips at the same beat should sum."""
        seq = Sequencer(tempo_bpm=120)
        seq.add_clip(np.ones(500) * 0.4, start_beat=0)
        seq.add_clip(np.ones(500) * 0.3, start_beat=0)
        result = seq.render()
        assert abs(result[0] - 0.7) < 0.001

    def test_different_tempos_same_beats(self):
        """Same beat pattern at different tempos should produce different lengths."""
        sig = np.ones(100)
        fast = Sequencer(tempo_bpm=200)
        fast.add_clip(sig, start_beat=4)
        slow = Sequencer(tempo_bpm=60)
        slow.add_clip(sig, start_beat=4)
        assert slow.duration_sec > fast.duration_sec

    def test_add_pattern_sec(self):
        """add_pattern_sec should use absolute time offset."""
        pat = Pattern(duration_sec=1.0)
        pat.add_clip(np.ones(100), start_sec=0.0)

        seq = Sequencer(tempo_bpm=120)
        seq.add_pattern_sec(pat, start_sec=3.0)
        result = seq.render()

        start = int(3.0 * SAMPLE_RATE)
        assert result[start] == 1.0

    def test_stereo_render(self):
        """Sequencer with stereo clips should produce stereo output."""
        seq = Sequencer(tempo_bpm=120)
        sig = mono_to_stereo(np.ones(100))
        seq.add_clip(sig, start_beat=0)
        result = seq.render()
        assert is_stereo(result)
