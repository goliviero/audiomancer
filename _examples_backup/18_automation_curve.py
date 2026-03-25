"""Example 18 — Automation curve: freeform volume ride.

Draw an arbitrary volume curve with breakpoints.
The signal fades in, peaks at 3s, dips at 5s, peaks again at 7s, then fades out.

This is the most flexible envelope — lets you "automate" volume like in a DAW.

Run: python examples/18_automation_curve.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.envelopes import AutomationCurve
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/18_automation_curve.wav")

DURATION = 10.0

# Rich chord
a = sine(110.0, DURATION, amplitude=0.4)
b = sine(165.0, DURATION, amplitude=0.3)
c = sine(220.0, DURATION, amplitude=0.2)
chord = mix_signals([a, b, c])

# Freeform automation: (time_sec, gain)
auto = AutomationCurve([
    (0.0, 0.0),    # start silent
    (1.5, 0.8),    # rise
    (3.0, 1.0),    # peak
    (4.0, 0.4),    # dip
    (5.0, 0.3),    # hold low
    (7.0, 1.0),    # second peak
    (9.0, 0.6),    # ease down
    (10.0, 0.0),   # end silent
])

shaped = auto.apply(chord)
shaped = normalize_peak(shaped, target_db=-3.0)
shaped = mono_to_stereo(shaped)

export_wav(shaped, OUTPUT)
print(f"[OK] {OUTPUT} -- 10s chord with freeform automation curve (8 breakpoints)")
