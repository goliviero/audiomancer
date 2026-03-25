"""Example 44 -- Tempo demo: same pattern at 80, 120, 160 BPM.

Three versions of a kick+snare pattern concatenated: slow, medium, fast.

Run: python examples/44_tempo_demo.py
"""

from pathlib import Path

from fractal.generators import sine, white_noise
from fractal.envelopes import ADSR
from fractal.sequencer import Sequencer
from fractal.signal import normalize_peak, concat, silence
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/44_tempo_demo.wav")

# Sounds
kick = sine(60.0, 0.15, amplitude=0.7)
kick = ADSR(attack=0.005, decay=0.08, sustain=0.1, release=0.05).apply(kick)

snare = white_noise(0.12, amplitude=0.4)
snare = ADSR(attack=0.002, decay=0.05, sustain=0.2, release=0.04).apply(snare)

sections = []
for bpm in [80, 120, 160]:
    seq = Sequencer(tempo_bpm=bpm)
    for bar in range(2):  # 2 bars each
        seq.add_clip(kick, start_beat=bar * 4 + 0)
        seq.add_clip(snare, start_beat=bar * 4 + 1)
        seq.add_clip(kick, start_beat=bar * 4 + 2)
        seq.add_clip(snare, start_beat=bar * 4 + 3)
    sections.append(seq.render(tail_sec=0.3))

gap = silence(0.5)
result = concat(sections[0], gap, sections[1], gap, sections[2])
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- same pattern at 80, 120, 160 BPM")
