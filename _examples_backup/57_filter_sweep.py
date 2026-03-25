"""Example 57 — Filter Sweep.

Classic acid-style filter sweep: sawtooth through a lowpass filter
that opens and closes over time. Linear and exponential curves compared.
"""

from fractal.generators import sawtooth
from fractal.modulation import apply_filter_sweep
from fractal.effects import NormalizePeak
from fractal.export import export_wav
from fractal.constants import SAMPLE_RATE

DURATION = 4.0

# Raw sawtooth
raw = sawtooth(110, DURATION, amplitude=0.8)

# Linear sweep: low -> high -> low (open then close)
sweep_up = apply_filter_sweep(raw, 200, 8000, filter_type="lowpass", curve="linear")

# Exponential sweep: more musical, perceptually linear
sweep_exp = apply_filter_sweep(raw, 200, 8000, filter_type="lowpass", curve="exponential")

# Normalize
fx = NormalizePeak(target_db=-3.0)
sweep_up = fx.process(sweep_up, SAMPLE_RATE)
sweep_exp = fx.process(sweep_exp, SAMPLE_RATE)

export_wav(sweep_up, "outputs/audio/57_filter_sweep_linear.wav")
export_wav(sweep_exp, "outputs/audio/57_filter_sweep_exponential.wav")
print("Exported: outputs/audio/57_filter_sweep_linear.wav")
print("Exported: outputs/audio/57_filter_sweep_exponential.wav")
