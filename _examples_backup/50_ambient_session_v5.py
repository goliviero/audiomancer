"""Example 50 -- Ambient session v5: the Phase 5 flagship.

Combines everything: Session + Sequencer + Envelopes + Effects.

Layers:
  1. Continuous drone pad (Session track)
  2. Pink noise bed (Session track, atmosphere bus)
  3. Binaural theta (Session track)
  4. Sequenced melodic fragments (notes placed on beats via Sequencer)
  5. Sequenced rhythmic texture (soft clicks on patterns)

Run: python examples/50_ambient_session_v5.py
"""

from pathlib import Path

from fractal.generators import sine, pink_noise, binaural
from fractal.envelopes import (
    SmoothFade, Swell, ExponentialFade, ADSR, Tremolo,
)
from fractal.sequencer import Pattern, Sequencer
from fractal.mixer import Session
from fractal.effects import (
    HighPassFilter, EQ, Reverb, StereoWidth, Delay, NormalizePeak,
)
from fractal.constants import SAMPLE_RATE
from fractal.signal import mix_signals, mono_to_stereo, normalize_peak
from fractal.export import export_wav, export_flac

OUTPUT_WAV = Path("outputs/audio/50_ambient_session_v5.wav")
OUTPUT_FLAC = Path("outputs/audio/50_ambient_session_v5.flac")

BPM = 60
BEAT_SEC = 60.0 / BPM  # 1s per beat
DURATION = 60.0

# =============================================
# Part 1: Sequenced elements
# =============================================

# Melodic fragments: slow pentatonic notes
NOTES = [220.0, 261.63, 329.63, 392.00, 440.0]  # A3 to A4 pentatonic-ish
adsr = ADSR(attack=0.05, decay=0.2, sustain=0.3, release=0.4)

melody_pat = Pattern(duration_sec=BEAT_SEC * 8)
for i, freq in enumerate(NOTES):
    note = sine(freq, BEAT_SEC * 1.5, amplitude=0.15)
    note = adsr.apply(note)
    # Place notes with gaps
    melody_pat.add_clip(note, start_sec=i * BEAT_SEC * 1.5)

# Rhythmic texture: soft clicks
click = sine(1200.0, 0.02, amplitude=0.1)
click = ADSR(attack=0.001, decay=0.01, sustain=0.05, release=0.005).apply(click)

rhythm_pat = Pattern(duration_sec=BEAT_SEC * 4)
rhythm_pat.add_clip(click, start_sec=0.0)
rhythm_pat.add_clip(click, start_sec=BEAT_SEC * 1.5)
rhythm_pat.add_clip(click, start_sec=BEAT_SEC * 3)

# Build sequencer timeline
seq = Sequencer(tempo_bpm=BPM)
# Melody enters at beat 8 (8s), repeats
for start in range(8, 48, 8):
    seq.add_pattern(melody_pat, start_beat=start)
# Rhythm enters at beat 4, loops throughout
rhythm_loop = rhythm_pat.repeat(12)
seq.add_pattern(rhythm_loop, start_beat=4)

sequenced = seq.render(tail_sec=2.0)
# Pad to full duration
if sequenced.shape[0] < int(DURATION * SAMPLE_RATE):
    import numpy as np
    pad_len = int(DURATION * SAMPLE_RATE) - sequenced.shape[0]
    sequenced = np.concatenate([sequenced, np.zeros(pad_len)])
sequenced = SmoothFade(fade_in=4.0, fade_out=6.0).apply(sequenced)

# =============================================
# Part 2: Session with continuous layers
# =============================================

session = Session(master_effects=[
    HighPassFilter(cutoff_hz=35),
    EQ(bands=[
        {"type": "peak", "freq": 150, "gain_db": 1.0},
        {"type": "high_shelf", "freq": 8000, "gain_db": -3.0},
    ]),
    NormalizePeak(target_db=-3.0),
])

# Atmosphere bus
session.add_bus("atmosphere", volume_db=-3.0, effects=[
    Reverb(decay=0.4, mix=0.25, room_size="large"),
    EQ(bands=[{"type": "high_shelf", "freq": 5000, "gain_db": -4.0}]),
])

# Layer 1: Drone pad
drone_a = sine(110.0, DURATION, amplitude=0.15)
drone_b = sine(165.0, DURATION, amplitude=0.08)
drone = mix_signals([drone_a, drone_b], volumes_db=[0.0, -6.0])
drone = Swell(rise_time=10.0, curve_type="cosine").apply(drone)
drone = SmoothFade(fade_out=10.0).apply(drone)
session.add_track("drone", drone, volume_db=-9.0)

# Layer 2: Noise bed
noise = pink_noise(DURATION, amplitude=0.3)
noise = SmoothFade(fade_in=5.0, fade_out=6.0).apply(noise)
session.add_track("noise", noise, bus="atmosphere", volume_db=-6.0,
                  effects=[
                      EQ(bands=[
                          {"type": "low_shelf", "freq": 80, "gain_db": 2.0},
                          {"type": "high_shelf", "freq": 5000, "gain_db": -5.0},
                      ]),
                      StereoWidth(width=1.3),
                  ])

# Layer 3: Binaural theta
theta = binaural(200.0, 6.0, DURATION, amplitude=0.05)
theta = Swell(rise_time=12.0, curve_type="cosine").apply(theta)
theta = SmoothFade(fade_out=10.0).apply(theta)
session.add_track("theta", theta, volume_db=-27.0)

# Layer 4: Sequenced elements (melody + rhythm)
sequenced_stereo = mono_to_stereo(sequenced)
session.add_track("sequenced", sequenced_stereo, bus="atmosphere", volume_db=-12.0,
                  effects=[Delay(delay_ms=500, feedback=0.2, mix=0.3)])

# --- Export ---
session.export(OUTPUT_WAV)
result = session.render()
export_flac(result, OUTPUT_FLAC)

print(f"[OK] {OUTPUT_WAV}")
print(f"[OK] {OUTPUT_FLAC}")
print("     60s ambient session v5 -- Session + Sequencer combined")
print("     Drone + noise + binaural + melodic fragments + rhythmic texture")
print("     Use with headphones!")
