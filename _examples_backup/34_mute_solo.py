"""Example 34 -- Mute/Solo demo: compare full mix vs. soloed track.

Creates 3 tracks, exports the full mix, then solos one track and exports again.
Compare the two files to hear the difference.

Run: python examples/34_mute_solo.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise
from fractal.envelopes import SmoothFade
from fractal.mixer import Session
from fractal.effects import NormalizePeak

OUTPUT_FULL = Path("outputs/audio/34_mute_solo_full.wav")
OUTPUT_SOLO = Path("outputs/audio/34_mute_solo_solo.wav")

fade = SmoothFade(fade_in=1.0, fade_out=1.0)


def build_session():
    session = Session(master_effects=[NormalizePeak(target_db=-1.0)])
    session.add_track("bass", fade.apply(sine(110.0, 5.0, amplitude=0.4)), volume_db=-3.0)
    session.add_track("lead", fade.apply(sine(440.0, 5.0, amplitude=0.3)), volume_db=-6.0, pan=0.4)
    session.add_track("noise", fade.apply(pink_noise(5.0, amplitude=0.15)), volume_db=-18.0)
    return session


# Full mix
full = build_session()
full.export(OUTPUT_FULL)
print(f"[OK] {OUTPUT_FULL} -- full mix (3 tracks)")

# Solo lead
solo = build_session()
solo.solo("lead")
solo.export(OUTPUT_SOLO)
print(f"[OK] {OUTPUT_SOLO} -- soloed 'lead' track only")
