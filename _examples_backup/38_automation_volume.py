"""Example 38 -- Volume automation via AutomationCurve on a track.

Uses an AutomationCurve envelope to create a volume swell that ramps
from silence to full over the first half, then dips and comes back.

Run: python examples/38_automation_volume.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.envelopes import AutomationCurve
from fractal.mixer import Session
from fractal.effects import NormalizePeak

OUTPUT = Path("outputs/audio/38_automation_volume.wav")

DURATION = 12.0

# Automation: silence -> full -> dip -> back
auto = AutomationCurve(points=[
    (0.0, 0.0),      # start silent
    (3.0, 1.0),      # ramp up
    (6.0, 0.3),      # dip
    (9.0, 1.0),      # back up
    (12.0, 0.0),     # fade out
])

pad = sine(220.0, DURATION, amplitude=0.5)
pad = auto.apply(pad)

session = Session(master_effects=[NormalizePeak(target_db=-1.0)])
session.add_track("pad", pad, volume_db=-3.0)

session.export(OUTPUT)
print(f"[OK] {OUTPUT} -- volume automation: swell -> dip -> swell -> fade")
