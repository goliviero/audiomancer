"""Example 28 — Bandpass 'radio' effect: lo-fi telephone sound.

Takes a chord and filters it through a narrow bandpass (300-3000 Hz),
mimicking the frequency range of a telephone or AM radio.

Run: python examples/28_bandpass_radio.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.signal import mix_signals, normalize_peak, concat, silence
from fractal.effects import BandPassFilter
from fractal.envelopes import SmoothFade
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/28_bandpass_radio.wav")

DURATION = 4.0

# Rich chord
c = sine(261.63, DURATION, amplitude=0.4)
e = sine(329.63, DURATION, amplitude=0.3)
g = sine(392.00, DURATION, amplitude=0.3)
chord = mix_signals([c, e, g])

fade = SmoothFade(fade_in=0.3, fade_out=0.3)

# Full bandwidth
full = fade.apply(chord)

# 'Radio' bandpass
radio = BandPassFilter(low_hz=300, high_hz=3000).process(chord)
radio = fade.apply(radio)

gap = silence(0.5)
result = concat(full, gap, radio)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- C major chord: full bandwidth then 'radio' bandpass (300-3000Hz)")
