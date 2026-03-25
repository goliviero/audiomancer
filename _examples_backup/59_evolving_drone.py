"""Example 59 — Evolving Drone.

A drone that evolves over time using LFO-modulated filter cutoff
and tremolo. Demonstrates how modulation turns a static sound
into something alive.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.synth import unison, additive, HARMONIC_PRESETS
from fractal.generators import sawtooth
from fractal.modulation import LFO, apply_filter_sweep, apply_param_automation
from fractal.effects import LowPassFilter, Reverb, NormalizePeak, EffectChain
from fractal.envelopes import SmoothFade
from fractal.music_theory import note_to_hz
from fractal.export import export_wav

DURATION = 15.0
n = int(SAMPLE_RATE * DURATION)

# Base: thick unison sawtooth drone on D2
drone = unison(sawtooth, note_to_hz("D2"), DURATION, voices=5,
               detune_cents=15, amplitude=0.7)

# Layer: additive string harmonics on D3
harmonics = additive(note_to_hz("D3"), HARMONIC_PRESETS["string"],
                     DURATION, amplitude=0.3)

# Mix layers
mix = drone + harmonics[:len(drone)]

# LFO-driven filter sweep: slow breathing effect
lfo = LFO(rate=0.15, shape="sine", depth=1.0, bipolar=False)
cutoff_curve = lfo.modulate_param(1500, 1200, n)

# Apply filter automation block by block
lpf = LowPassFilter(cutoff_hz=1500)
mix = apply_param_automation(mix, lpf, "cutoff_hz", cutoff_curve)

# Tremolo via amplitude LFO (very subtle)
trem_lfo = LFO(rate=0.3, shape="triangle", depth=0.15, bipolar=False)
trem_curve = trem_lfo.generate(n)
# Scale from 0.85 to 1.0
trem_curve = 0.85 + trem_curve * 0.15 / 0.15  # map [0, 0.15] to [0.85, 1.0]
mix = mix * trem_curve

# Smooth fades
fade = SmoothFade(fade_in=3.0, fade_out=3.0)
mix = fade.apply(mix)

# Final effects
fx = EffectChain([
    Reverb(decay=0.5, mix=0.3),
    NormalizePeak(target_db=-3.0),
])
mix = fx.process(mix, SAMPLE_RATE)

export_wav(mix, "outputs/audio/59_evolving_drone.wav")
print("Exported: outputs/audio/59_evolving_drone.wav")
