"""Example 52 — FM Bell.

Classic FM synthesis bell using a high modulation index with
a decaying modulation envelope. The DX7 bell sound: carrier and
modulator at the same frequency, mod_index sweeps from 5 to 0.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE
from fractal.synth import fm_synth
from fractal.envelopes import ADSR
from fractal.effects import Reverb, NormalizePeak, EffectChain
from fractal.export import export_wav
from fractal.music_theory import note_to_hz

DURATION = 3.0
n_samples = int(SAMPLE_RATE * DURATION)

# Modulation envelope: fast attack, slow exponential decay
# This gives the "bell strike" character -- bright attack, mellow sustain
mod_env = np.exp(-np.linspace(0, 8, n_samples))

# Bell frequencies: carrier:modulator ratio 1:1 gives harmonic bell
# Ratio 1:1.4 gives more metallic/inharmonic bell
freq = note_to_hz("E5")

# Harmonic bell (1:1 ratio)
bell_harmonic = fm_synth(
    carrier_hz=freq,
    modulator_hz=freq,
    mod_index=5.0,
    duration_sec=DURATION,
    amplitude=0.6,
    mod_envelope=mod_env,
)

# Metallic bell (1:1.4 ratio -- inharmonic)
bell_metallic = fm_synth(
    carrier_hz=freq,
    modulator_hz=freq * 1.4,
    mod_index=4.0,
    duration_sec=DURATION,
    amplitude=0.6,
    mod_envelope=mod_env,
)

# Apply amplitude envelope (fast attack, long release)
adsr = ADSR(attack=0.005, decay=0.3, sustain=0.2, release=1.5)
env_curve = adsr.generate(n_samples, SAMPLE_RATE)

bell_harmonic = bell_harmonic * env_curve
bell_metallic = bell_metallic * env_curve

# Add reverb for space
fx = EffectChain([
    Reverb(decay=0.5, mix=0.35),
    NormalizePeak(target_db=-3.0),
])

bell_harmonic = fx.process(bell_harmonic, SAMPLE_RATE)
bell_metallic = fx.process(bell_metallic, SAMPLE_RATE)

# Export both
export_wav(bell_harmonic, "outputs/audio/52_fm_bell_harmonic.wav")
export_wav(bell_metallic, "outputs/audio/52_fm_bell_metallic.wav")

print("Exported: outputs/audio/52_fm_bell_harmonic.wav")
print("Exported: outputs/audio/52_fm_bell_metallic.wav")
