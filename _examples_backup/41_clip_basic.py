"""Example 41 -- Basic clip: a sound placed at a specific time.

Places a short tone at t=2s on a 5-second timeline.

Run: python examples/41_clip_basic.py
"""

from pathlib import Path

from fractal.generators import sine
from fractal.envelopes import SmoothFade
from fractal.sequencer import Sequencer
from fractal.signal import normalize_peak
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/41_clip_basic.wav")

tone = sine(440.0, 1.0, amplitude=0.5)
tone = SmoothFade(fade_in=0.05, fade_out=0.2).apply(tone)

seq = Sequencer(tempo_bpm=120)
seq.add_clip_sec(tone, start_sec=2.0)

result = seq.render(tail_sec=1.0)
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- 440Hz tone placed at t=2s on a timeline")
