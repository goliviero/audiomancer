"""Example 60 — Preset Showcase.

Render one note from every preset category to demonstrate the
variety of sounds available out of the box.
"""

from fractal.presets import list_presets, get_preset, SYNTH_PRESETS
from fractal.effects import NormalizePeak
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

DUR = 2.0
fx = NormalizePeak(target_db=-3.0)

# One showcase per category
showcase = {
    "pad": ("blade_runner_pad", "D4"),
    "lead": ("glass_bell", "E5"),
    "bass": ("acid_squelch", "E2"),
    "key": ("electric_piano", "C4"),
    "texture": ("ambient_drone", "A2"),
}

for category, (preset_name, note) in showcase.items():
    preset = get_preset(preset_name)
    sig = preset.render(note, DUR, amplitude=0.5)
    sig = fx.process(sig, SAMPLE_RATE)
    path = f"outputs/audio/60_preset_{category}_{preset_name}.wav"
    export_wav(sig, path)
    print(f"  [{category:>7s}] {preset_name:25s} -> {path}")

# Also list all available presets
print(f"\nAll {len(SYNTH_PRESETS)} presets:")
for cat in ["pad", "lead", "bass", "key", "texture"]:
    names = list_presets(cat)
    print(f"  {cat}: {', '.join(names)}")
