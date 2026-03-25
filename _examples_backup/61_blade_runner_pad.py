"""Example 61 — Blade Runner Pad.

The iconic Vangelis CS-80 pad sound, recreated with Fractal presets.
A Dm7 chord held for 12 seconds with slow evolution.
"""

from fractal.presets import get_preset
from fractal.music_theory import chord_hz
from fractal.signal import mix_signals
from fractal.envelopes import SmoothFade
from fractal.effects import Reverb, NormalizePeak, EffectChain
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

DUR = 12.0

# Get the preset
pad = get_preset("blade_runner_pad")

# Render each note of a Dm7 chord
chord_notes = chord_hz("D3", "min7")
voices = [pad.render(hz, DUR, amplitude=0.3) for hz in chord_notes]

# Mix all voices
mix = mix_signals(voices)

# Extra fade for smooth entry/exit
fade = SmoothFade(fade_in=2.0, fade_out=3.0)
mix = fade.apply(mix)

# Final master
fx = EffectChain([
    Reverb(decay=0.7, mix=0.3),
    NormalizePeak(target_db=-3.0),
])
mix = fx.process(mix, SAMPLE_RATE)

export_wav(mix, "outputs/audio/61_blade_runner_pad.wav")
print("Exported: outputs/audio/61_blade_runner_pad.wav")
print("  Preset: blade_runner_pad (FM, ratio=1.0, mod_index=3.5)")
print("  Chord: Dm7 (D3-F3-A3-C4)")
print("  Duration: 12s with 2s fade in, 3s fade out")
