"""Mixer — Bus routing and Session management.

A Session is the top-level container for a Fractal project. It holds tracks,
routes them through buses, and renders the final stereo mixdown.

Signal flow:
  Track.render() -> sum by bus -> Bus effects -> sum all buses -> master effects -> output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.effects import Effect
from fractal.export import export_wav, export_flac, export_auto
from fractal.signal import normalize_peak, pad_to_length
from fractal.track import Track


@dataclass
class Bus:
    """An audio bus that groups tracks and applies shared effects.

    Attributes:
        name: Bus identifier (e.g., "drums", "synths", "fx").
        effects: Effects applied to the summed bus signal.
        volume_db: Bus gain in dB.
    """
    name: str
    effects: list[Effect] = field(default_factory=list)
    volume_db: float = 0.0


class Session:
    """Top-level session — holds tracks, buses, and master chain.

    Usage:
        session = Session()
        session.add_track("kick", kick_signal, volume_db=-6, bus="drums")
        session.add_track("pad", pad_signal, pan=-0.3)
        session.add_bus("drums", effects=[HighPassFilter(cutoff_hz=40)])
        result = session.render()
        session.export("output.wav")

    Attributes:
        sample_rate: Session sample rate.
        tracks: Named tracks.
        buses: Named buses (master always exists).
        master_effects: Effects applied to the final mixdown.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 master_effects: list[Effect] | None = None):
        self.sample_rate = sample_rate
        self.tracks: dict[str, Track] = {}
        self.buses: dict[str, Bus] = {"master": Bus(name="master")}
        self.master_effects: list[Effect] = master_effects or []

    def add_track(self, name: str, signal: np.ndarray, **kwargs) -> Track:
        """Add a track to the session.

        Args:
            name: Unique track name.
            signal: Audio signal (mono or stereo).
            **kwargs: Passed to Track (volume_db, pan, mute, solo, effects, bus).

        Returns:
            The created Track.
        """
        if name in self.tracks:
            raise ValueError(f"Track '{name}' already exists in session.")
        track = Track(name=name, signal=signal, **kwargs)
        self.tracks[name] = track
        return track

    def add_bus(self, name: str, **kwargs) -> Bus:
        """Add a bus to the session.

        Args:
            name: Unique bus name.
            **kwargs: Passed to Bus (effects, volume_db).

        Returns:
            The created Bus.
        """
        if name in self.buses:
            raise ValueError(f"Bus '{name}' already exists in session.")
        bus = Bus(name=name, **kwargs)
        self.buses[name] = bus
        return bus

    def solo(self, *track_names: str) -> None:
        """Enable solo on specific tracks, disable on all others.

        When any track is soloed, only soloed tracks are rendered.
        Call with no arguments to clear all solos.
        """
        for track in self.tracks.values():
            track.solo = track.name in track_names

    def clear_solo(self) -> None:
        """Clear solo flags on all tracks."""
        for track in self.tracks.values():
            track.solo = False

    def render(self) -> np.ndarray:
        """Render the full session mixdown.

        Signal flow:
          1. Each track: effects -> gain -> pan (Track.render)
          2. Sum tracks by bus assignment
          3. Apply bus effects + bus gain
          4. Sum all buses into master
          5. Apply master effects

        Returns:
            Stereo (n, 2) numpy array — the final mix.
        """
        if not self.tracks:
            return np.zeros((0, 2), dtype=np.float64)

        # Determine if solo mode is active
        any_solo = any(t.solo for t in self.tracks.values())

        # Render each track
        rendered: dict[str, list[np.ndarray]] = {}
        for track in self.tracks.values():
            # Solo logic: if any track is soloed, skip non-soloed tracks
            if any_solo and not track.solo:
                continue
            # Mute is handled inside Track.render()
            sig = track.render(self.sample_rate)
            bus_name = track.bus if track.bus in self.buses else "master"
            rendered.setdefault(bus_name, []).append(sig)

        # Find max length across all rendered signals
        all_sigs = [s for sigs in rendered.values() for s in sigs]
        if not all_sigs:
            return np.zeros((0, 2), dtype=np.float64)
        max_len = max(s.shape[0] for s in all_sigs)

        # Sum by bus and apply bus effects
        bus_outputs: list[np.ndarray] = []
        for bus_name, bus in self.buses.items():
            if bus_name not in rendered:
                continue
            # Sum all tracks assigned to this bus
            bus_sum = np.zeros((max_len, 2), dtype=np.float64)
            for sig in rendered[bus_name]:
                padded = pad_to_length(sig, max_len)
                bus_sum = bus_sum + padded

            # Apply bus effects
            for fx in bus.effects:
                bus_sum = fx.process(bus_sum, self.sample_rate)

            # Apply bus gain
            bus_gain = 10 ** (bus.volume_db / 20)
            bus_sum = bus_sum * bus_gain

            bus_outputs.append(bus_sum)

        # Sum all buses
        master = np.zeros((max_len, 2), dtype=np.float64)
        for bus_out in bus_outputs:
            master = master + pad_to_length(bus_out, max_len)

        # Apply master effects
        for fx in self.master_effects:
            master = fx.process(master, self.sample_rate)

        return master

    def export(self, path: str | Path, **kwargs) -> Path:
        """Render and export the session to a file.

        Args:
            path: Output file path (.wav or .flac).
            **kwargs: Passed to export function (bit_depth, etc.).

        Returns:
            The path written to.
        """
        result = self.render()
        return export_auto(result, Path(path), sample_rate=self.sample_rate, **kwargs)
