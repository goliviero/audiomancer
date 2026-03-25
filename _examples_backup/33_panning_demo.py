"""Example 33 -- Panning demo: same tone at left, center, right.

Three short tones separated by silence: hard left, center, hard right.
Use headphones to hear the spatial positioning clearly.

Run: python examples/33_panning_demo.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import concat, silence, normalize_peak
from fractal.envelopes import SmoothFade
from fractal.track import Track
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/33_panning_demo.wav")

DURATION = 2.0
fade = SmoothFade(fade_in=0.1, fade_out=0.3)
gap = silence(0.5, stereo=True)

tone = sine(440.0, DURATION, amplitude=0.5)
tone = fade.apply(tone)

left = Track(name="left", signal=tone, pan=-1.0).render()
center = Track(name="center", signal=tone, pan=0.0).render()
right = Track(name="right", signal=tone, pan=1.0).render()

result = concat(left, gap, center, gap, right)
result = normalize_peak(result, target_db=-1.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- panning: left -> center -> right (use headphones)")
