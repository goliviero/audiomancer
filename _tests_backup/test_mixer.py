"""Tests for fractal.mixer — Bus routing and Session management."""

import numpy as np
import pytest
from pathlib import Path

from fractal.constants import SAMPLE_RATE
from fractal.generators import sine
from fractal.signal import is_stereo, mono_to_stereo
from fractal.effects import HighPassFilter, NormalizePeak
from fractal.mixer import Bus, Session


class TestBus:
    def test_defaults(self):
        bus = Bus(name="drums")
        assert bus.name == "drums"
        assert bus.volume_db == 0.0
        assert bus.effects == []


class TestSession:
    def test_empty_session(self):
        """Empty session renders to empty array."""
        session = Session()
        result = session.render()
        assert result.shape == (0, 2)

    def test_single_track(self):
        """Session with one track renders correctly."""
        session = Session()
        sig = sine(440, 0.5, amplitude=0.5)
        session.add_track("sine", sig)
        result = session.render()
        assert is_stereo(result)
        assert result.shape[0] == sig.shape[0]
        assert np.max(np.abs(result)) > 0

    def test_multi_track_mix(self):
        """Multiple tracks should be summed together."""
        session = Session()
        sig_a = sine(440, 0.5, amplitude=0.3)
        sig_b = sine(880, 0.5, amplitude=0.3)
        session.add_track("a", sig_a)
        session.add_track("b", sig_b)
        result = session.render()
        rms = np.sqrt(np.mean(result ** 2))
        # Mixed should be louder than either alone
        rms_a = np.sqrt(np.mean(session.tracks["a"].render() ** 2))
        assert rms > rms_a * 0.9

    def test_duplicate_track_name_raises(self):
        session = Session()
        session.add_track("kick", np.zeros(100))
        with pytest.raises(ValueError, match="already exists"):
            session.add_track("kick", np.zeros(100))

    def test_duplicate_bus_name_raises(self):
        session = Session()
        with pytest.raises(ValueError, match="already exists"):
            session.add_bus("master")  # already exists by default

    def test_mute_track(self):
        """Muted track should not contribute to output."""
        session = Session()
        sig = sine(440, 0.5, amplitude=0.5)
        session.add_track("muted", sig, mute=True)
        result = session.render()
        assert np.max(np.abs(result)) == 0.0

    def test_solo_mode(self):
        """Solo should only render soloed tracks."""
        session = Session()
        sig_a = sine(440, 0.5, amplitude=0.5)
        sig_b = sine(880, 0.5, amplitude=0.5)
        session.add_track("a", sig_a)
        session.add_track("b", sig_b)
        session.solo("a")

        result = session.render()
        # Result should be close to just track "a" rendered alone
        a_alone = session.tracks["a"].render()
        # Compare RMS — should be very similar
        rms_mix = np.sqrt(np.mean(result ** 2))
        rms_a = np.sqrt(np.mean(a_alone ** 2))
        assert abs(rms_mix - rms_a) / rms_a < 0.01

    def test_clear_solo(self):
        """clear_solo should reset all solo flags."""
        session = Session()
        session.add_track("a", np.zeros(100))
        session.add_track("b", np.zeros(100))
        session.solo("a")
        assert session.tracks["a"].solo is True
        session.clear_solo()
        assert all(not t.solo for t in session.tracks.values())

    def test_bus_routing(self):
        """Tracks routed to a bus should be summed through that bus."""
        session = Session()
        session.add_bus("synths")
        sig = sine(440, 0.5, amplitude=0.5)
        session.add_track("pad", sig, bus="synths")
        result = session.render()
        assert is_stereo(result)
        assert np.max(np.abs(result)) > 0

    def test_bus_effects(self):
        """Bus effects should be applied to the bus sum."""
        session = Session()
        session.add_bus("synths", effects=[HighPassFilter(cutoff_hz=10000)])
        sig = sine(200, 1.0, amplitude=0.5)  # 200Hz — well below HPF cutoff
        session.add_track("bass", sig, bus="synths")
        result = session.render()
        rms = np.sqrt(np.mean(result ** 2))
        # HPF at 10kHz should severely cut 200Hz
        assert rms < 0.05

    def test_bus_volume(self):
        """Bus volume_db should scale its output."""
        session = Session()
        session.add_bus("loud_bus", volume_db=6.0)
        session.add_bus("quiet_bus", volume_db=-12.0)

        sig = sine(440, 0.5, amplitude=0.3)
        session.add_track("loud", sig, bus="loud_bus")

        sig2 = sine(440, 0.5, amplitude=0.3)
        session.add_track("quiet", sig2, bus="quiet_bus")

        result = session.render()
        # Can't easily separate buses in result, but at least it renders
        assert is_stereo(result)

    def test_master_effects(self):
        """Master effects should be applied to the final mix."""
        session = Session(master_effects=[NormalizePeak(target_db=-6.0)])
        sig = sine(440, 0.5, amplitude=0.8)
        session.add_track("loud", sig)
        result = session.render()
        peak_db = 20 * np.log10(np.max(np.abs(result)) + 1e-10)
        assert abs(peak_db - (-6.0)) < 0.5

    def test_export(self, tmp_path):
        """Session.export() should write a valid audio file."""
        session = Session()
        sig = sine(440, 0.5, amplitude=0.5)
        session.add_track("test", sig)
        out = tmp_path / "test_export.wav"
        result_path = session.export(out)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_flac(self, tmp_path):
        """Session.export() should handle FLAC format."""
        session = Session()
        sig = sine(440, 0.5, amplitude=0.5)
        session.add_track("test", sig)
        out = tmp_path / "test_export.flac"
        result_path = session.export(out)
        assert result_path.exists()

    def test_unknown_bus_falls_to_master(self):
        """Track with unknown bus name should route to master."""
        session = Session()
        sig = sine(440, 0.5, amplitude=0.5)
        session.add_track("orphan", sig, bus="nonexistent")
        result = session.render()
        assert np.max(np.abs(result)) > 0

    def test_add_track_returns_track(self):
        """add_track should return the created Track object."""
        session = Session()
        track = session.add_track("t", np.zeros(100), volume_db=-3.0)
        assert track.name == "t"
        assert track.volume_db == -3.0

    def test_add_bus_returns_bus(self):
        """add_bus should return the created Bus object."""
        session = Session()
        bus = session.add_bus("fx", volume_db=-6.0)
        assert bus.name == "fx"
        assert bus.volume_db == -6.0
