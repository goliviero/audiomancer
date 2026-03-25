"""Example 48 -- Drone + pattern overlay: continuous pad with rhythmic elements.

A continuous drone pad with a sequenced rhythmic pattern on top.
Shows how to combine the Sequencer with a long-form signal.

Run: python examples/48_drone_pattern_overlay.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise
from fractal.envelopes import SmoothFade, ADSR, Swell
from fractal.sequencer import Pattern, Sequencer
from fractal.signal import mix_signals, normalize_peak, mono_to_stereo
from fractal.export import export_wav

OUTPUT = Path("outputs/audio/48_drone_pattern_overlay.wav")

BPM = 90
BEAT_SEC = 60.0 / BPM
DURATION = BEAT_SEC * 16  # 4 bars

# --- Drone layer (continuous) ---
drone = sine(110.0, DURATION, amplitude=0.15)
drone = Swell(rise_time=3.0).apply(drone)
drone = SmoothFade(fade_out=2.0).apply(drone)

noise = pink_noise(DURATION, amplitude=0.1)
noise = SmoothFade(fade_in=2.0, fade_out=2.0).apply(noise)

pad = mix_signals([drone, noise], volumes_db=[-6.0, -9.0])
pad = mono_to_stereo(pad)

# --- Rhythmic pattern ---
click = sine(800.0, 0.05, amplitude=0.3)
click = ADSR(attack=0.002, decay=0.02, sustain=0.1, release=0.02).apply(click)

bar = Pattern(duration_sec=BEAT_SEC * 4)
bar.add_clip(click, start_sec=0.0)
bar.add_clip(click, start_sec=BEAT_SEC * 1.5)
bar.add_clip(click, start_sec=BEAT_SEC * 2)
bar.add_clip(click, start_sec=BEAT_SEC * 3)

seq = Sequencer(tempo_bpm=BPM)
full_pattern = bar.repeat(4)
seq.add_pattern(full_pattern, start_beat=0)
rhythm = seq.render()
rhythm = mono_to_stereo(rhythm)

# --- Combine ---
result = mix_signals([pad, rhythm], volumes_db=[0.0, -9.0])
result = normalize_peak(result, target_db=-3.0)

export_wav(result, OUTPUT)
print(f"[OK] {OUTPUT} -- drone pad + rhythmic clicks, {BPM} BPM, 4 bars")
