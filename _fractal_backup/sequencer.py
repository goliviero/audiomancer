"""Sequencer — place sounds on a timeline with tempo and patterns.

Core concepts:
  - Clip: a signal placed at a time offset
  - Pattern: a group of clips with a fixed duration, loopable
  - Sequencer: manages tempo (BPM), places clips/patterns on a timeline, renders

The sequencer works in beats (tempo-relative) or seconds (absolute).
All rendering produces a single np.ndarray buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.signal import pad_to_length


@dataclass
class Clip:
    """A signal placed at a time offset.

    Attributes:
        signal: Audio data (mono or stereo).
        start_sec: Offset in seconds from the beginning.
        track_name: Optional label for routing/identification.
    """
    signal: np.ndarray
    start_sec: float = 0.0
    track_name: str | None = None


@dataclass
class Pattern:
    """A group of clips forming a repeatable section (e.g., one measure).

    Attributes:
        clips: List of Clip objects positioned within this pattern.
        duration_sec: Total duration of the pattern in seconds.
            Clips extending beyond this are truncated on render.
    """
    clips: list[Clip] = field(default_factory=list)
    duration_sec: float = 4.0

    def add_clip(self, signal: np.ndarray, start_sec: float = 0.0,
                 track_name: str | None = None) -> Clip:
        """Add a clip to this pattern.

        Args:
            signal: Audio signal.
            start_sec: Position within the pattern (seconds).
            track_name: Optional label.

        Returns:
            The created Clip.
        """
        clip = Clip(signal=signal, start_sec=start_sec, track_name=track_name)
        self.clips.append(clip)
        return clip

    def repeat(self, n: int) -> Pattern:
        """Create a new pattern that loops this one N times.

        Args:
            n: Number of repetitions.

        Returns:
            A new Pattern with duration = self.duration_sec * n.
        """
        new_clips = []
        for i in range(n):
            offset = i * self.duration_sec
            for clip in self.clips:
                new_clip = Clip(
                    signal=clip.signal,
                    start_sec=clip.start_sec + offset,
                    track_name=clip.track_name,
                )
                new_clips.append(new_clip)
        return Pattern(clips=new_clips, duration_sec=self.duration_sec * n)

    def render(self, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Render all clips into a single buffer.

        Returns:
            Mono or stereo numpy array of length = duration_sec * sample_rate.
        """
        total_samples = int(self.duration_sec * sample_rate)
        if not self.clips:
            return np.zeros(total_samples, dtype=np.float64)

        # Determine if any clip is stereo
        any_stereo = any(c.signal.ndim == 2 for c in self.clips)

        if any_stereo:
            buf = np.zeros((total_samples, 2), dtype=np.float64)
        else:
            buf = np.zeros(total_samples, dtype=np.float64)

        for clip in self.clips:
            start_idx = int(clip.start_sec * sample_rate)
            if start_idx >= total_samples:
                continue

            sig = clip.signal
            # Convert mono to stereo if needed
            if any_stereo and sig.ndim == 1:
                sig = np.column_stack([sig, sig])

            # Truncate if clip extends beyond pattern duration
            available = total_samples - start_idx
            end_idx = start_idx + min(sig.shape[0], available)
            sig_len = end_idx - start_idx

            buf[start_idx:end_idx] = buf[start_idx:end_idx] + sig[:sig_len]

        return buf


class Sequencer:
    """Timeline-based sequencer with tempo management.

    Clips and patterns are placed using beat positions, which are
    converted to seconds via the tempo (BPM).

    Usage:
        seq = Sequencer(tempo_bpm=120)
        seq.add_clip(kick, start_beat=0)
        seq.add_clip(snare, start_beat=1)
        seq.add_clip(kick, start_beat=2)
        seq.add_clip(snare, start_beat=3)
        audio = seq.render()

    Attributes:
        tempo_bpm: Tempo in beats per minute.
        sample_rate: Audio sample rate.
    """

    def __init__(self, tempo_bpm: float = 120.0,
                 sample_rate: int = SAMPLE_RATE):
        self.tempo_bpm = tempo_bpm
        self.sample_rate = sample_rate
        self._clips: list[Clip] = []

    def beats_to_sec(self, beats: float) -> float:
        """Convert beat position to seconds using current tempo.

        Args:
            beats: Number of beats (e.g., 4.0 = one bar in 4/4 time).

        Returns:
            Time in seconds.
        """
        return beats * 60.0 / self.tempo_bpm

    def sec_to_beats(self, seconds: float) -> float:
        """Convert seconds to beat position.

        Args:
            seconds: Time in seconds.

        Returns:
            Beat position.
        """
        return seconds * self.tempo_bpm / 60.0

    def add_clip(self, signal: np.ndarray, start_beat: float = 0.0,
                 track_name: str | None = None) -> Clip:
        """Place a clip at a beat position on the timeline.

        Args:
            signal: Audio signal.
            start_beat: Beat position (converted to seconds via tempo).
            track_name: Optional label.

        Returns:
            The created Clip.
        """
        clip = Clip(
            signal=signal,
            start_sec=self.beats_to_sec(start_beat),
            track_name=track_name,
        )
        self._clips.append(clip)
        return clip

    def add_clip_sec(self, signal: np.ndarray, start_sec: float = 0.0,
                     track_name: str | None = None) -> Clip:
        """Place a clip at an absolute time position (seconds).

        Args:
            signal: Audio signal.
            start_sec: Position in seconds.
            track_name: Optional label.

        Returns:
            The created Clip.
        """
        clip = Clip(signal=signal, start_sec=start_sec, track_name=track_name)
        self._clips.append(clip)
        return clip

    def add_pattern(self, pattern: Pattern, start_beat: float = 0.0) -> None:
        """Place all clips from a pattern at a beat offset.

        Args:
            pattern: Pattern to insert.
            start_beat: Beat offset for the entire pattern.
        """
        offset_sec = self.beats_to_sec(start_beat)
        for clip in pattern.clips:
            new_clip = Clip(
                signal=clip.signal,
                start_sec=clip.start_sec + offset_sec,
                track_name=clip.track_name,
            )
            self._clips.append(new_clip)

    def add_pattern_sec(self, pattern: Pattern, start_sec: float = 0.0) -> None:
        """Place all clips from a pattern at a second offset.

        Args:
            pattern: Pattern to insert.
            start_sec: Time offset in seconds.
        """
        for clip in pattern.clips:
            new_clip = Clip(
                signal=clip.signal,
                start_sec=clip.start_sec + start_sec,
                track_name=clip.track_name,
            )
            self._clips.append(new_clip)

    @property
    def duration_sec(self) -> float:
        """Total duration in seconds (end of last clip)."""
        if not self._clips:
            return 0.0
        return max(
            c.start_sec + c.signal.shape[0] / self.sample_rate
            for c in self._clips
        )

    @property
    def duration_beats(self) -> float:
        """Total duration in beats."""
        return self.sec_to_beats(self.duration_sec)

    def render(self, tail_sec: float = 0.0) -> np.ndarray:
        """Render the full timeline into a single buffer.

        Args:
            tail_sec: Extra silence at the end (useful for reverb tails, etc.).

        Returns:
            Mono or stereo numpy array.
        """
        if not self._clips:
            return np.zeros(0, dtype=np.float64)

        total_sec = self.duration_sec + tail_sec
        total_samples = int(total_sec * self.sample_rate)

        any_stereo = any(c.signal.ndim == 2 for c in self._clips)

        if any_stereo:
            buf = np.zeros((total_samples, 2), dtype=np.float64)
        else:
            buf = np.zeros(total_samples, dtype=np.float64)

        for clip in self._clips:
            start_idx = int(clip.start_sec * self.sample_rate)
            if start_idx >= total_samples:
                continue

            sig = clip.signal
            if any_stereo and sig.ndim == 1:
                sig = np.column_stack([sig, sig])

            available = total_samples - start_idx
            end_idx = start_idx + min(sig.shape[0], available)
            sig_len = end_idx - start_idx

            buf[start_idx:end_idx] = buf[start_idx:end_idx] + sig[:sig_len]

        return buf
