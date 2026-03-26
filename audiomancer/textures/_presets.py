"""Texture presets — ready-to-use evolving ambient generators.

Each texture outputs a stereo signal with subtle, slow-moving changes —
designed for hours-long ambient listening.
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.synth import (
    drone, chord_pad, pad, pink_noise, brown_noise, white_noise,
)
from audiomancer.modulation import (
    lfo_sine, drift, evolving_lfo,
    apply_amplitude_mod, apply_filter_sweep,
)
from audiomancer.effects import lowpass, highpass, reverb, chorus, delay
from audiomancer.utils import mono_to_stereo, normalize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subtle modulation defaults — safe for extended listening
_AMP_MOD_DEPTH = 0.08      # ±8% amplitude variation
_FILTER_DRIFT_HZ = 400     # Filter moves ±400 Hz around center
_LFO_RATE_SLOW = 0.03      # ~33s cycle
_LFO_RATE_MEDIUM = 0.07    # ~14s cycle
_DRIFT_SPEED_SLOW = 0.05
_DRIFT_SPEED_MEDIUM = 0.1


# ---------------------------------------------------------------------------
# Texture generators
# ---------------------------------------------------------------------------

def evolving_drone(duration_sec: float, frequency: float = 111.0,
                   harmonics: list[tuple[float, float]] | None = None,
                   seed: int | None = None,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Rich harmonic drone with drifting amplitude and filter."""
    raw = drone(frequency, duration_sec, harmonics=harmonics,
                amplitude=0.7, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=2000, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    amp_mod = evolving_lfo(duration_sec, rate_hz=_LFO_RATE_SLOW,
                           depth=_AMP_MOD_DEPTH, offset=1.0,
                           drift_speed=_DRIFT_SPEED_SLOW,
                           seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    seed2 = (seed + 10) if seed is not None else None
    filter_mod = drift(duration_sec, speed=_DRIFT_SPEED_SLOW,
                       depth=_FILTER_DRIFT_HZ, offset=1200,
                       seed=seed2, sample_rate=sample_rate)
    stereo = apply_filter_sweep(stereo, filter_mod, sample_rate=sample_rate)

    stereo = reverb(stereo, room_size=0.95, damping=0.4, wet_level=0.6,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def breathing_pad(duration_sec: float,
                  frequencies: list[float] | None = None,
                  seed: int | None = None,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Chord pad with slow inhale/exhale amplitude movement."""
    if frequencies is None:
        frequencies = [130.81, 164.81, 196.0]  # C3, E3, G3

    raw = chord_pad(frequencies, duration_sec, voices=4,
                    detune_cents=12.0, amplitude=0.5,
                    sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=3000, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    breath = lfo_sine(duration_sec, rate_hz=0.04, depth=0.06,
                      offset=1.0, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, breath)

    stereo = chorus(stereo, rate_hz=0.3, depth=0.1, mix=0.25,
                    sample_rate=sample_rate)
    stereo = reverb(stereo, room_size=0.85, damping=0.5, wet_level=0.55,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def deep_space(duration_sec: float, seed: int | None = None,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Dark, vast ambient texture — brown noise sculpted with drifting filter."""
    raw = brown_noise(duration_sec, amplitude=0.6, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=800, sample_rate=sample_rate)
    raw = highpass(raw, cutoff_hz=30, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    filter_mod = drift(duration_sec, speed=_DRIFT_SPEED_MEDIUM,
                       depth=300, offset=600,
                       seed=seed, sample_rate=sample_rate)
    stereo = apply_filter_sweep(stereo, filter_mod, sample_rate=sample_rate)

    seed2 = (seed + 5) if seed is not None else None
    amp_mod = drift(duration_sec, speed=_DRIFT_SPEED_SLOW,
                    depth=0.05, offset=1.0,
                    seed=seed2, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    stereo = reverb(stereo, room_size=1.0, damping=0.3, wet_level=0.7,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def ocean_bed(duration_sec: float, seed: int | None = None,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Underwater ambient — pink noise with slow filter waves."""
    raw = pink_noise(duration_sec, amplitude=0.5, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=2000, sample_rate=sample_rate)
    raw = highpass(raw, cutoff_hz=60, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    filter_mod = evolving_lfo(duration_sec, rate_hz=0.06,
                              depth=500, offset=1000,
                              drift_speed=0.04, seed=seed,
                              sample_rate=sample_rate)
    stereo = apply_filter_sweep(stereo, filter_mod, sample_rate=sample_rate)

    amp_mod = lfo_sine(duration_sec, rate_hz=0.05, depth=0.05,
                       offset=1.0, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    stereo = reverb(stereo, room_size=0.8, damping=0.6, wet_level=0.45,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def crystal_shimmer(duration_sec: float,
                    base_freq: float = 800.0,
                    seed: int | None = None,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """High-frequency cluster with chorus and delay — sparkly, ethereal."""
    freqs = [base_freq, base_freq * 1.005, base_freq * 1.498,
             base_freq * 2.003, base_freq * 2.51]
    raw = chord_pad(freqs, duration_sec, voices=2, detune_cents=6.0,
                    amplitude=0.3, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=6000, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    amp_mod = evolving_lfo(duration_sec, rate_hz=_LFO_RATE_MEDIUM,
                           depth=_AMP_MOD_DEPTH, offset=1.0,
                           drift_speed=_DRIFT_SPEED_SLOW,
                           seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    stereo = chorus(stereo, rate_hz=0.4, depth=0.2, mix=0.4,
                    sample_rate=sample_rate)
    stereo = delay(stereo, delay_seconds=0.4, feedback=0.35, mix=0.25,
                   sample_rate=sample_rate)
    stereo = reverb(stereo, room_size=0.9, damping=0.4, wet_level=0.6,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def earth_hum(duration_sec: float, frequency: float = 60.0,
              seed: int | None = None,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Very low drone + brown noise bed — primal, grounding."""
    harmonics = [(1, 1.0), (2, 0.4), (3, 0.15)]
    raw_drone = drone(frequency, duration_sec, harmonics=harmonics,
                      amplitude=0.6, sample_rate=sample_rate)
    raw_drone = lowpass(raw_drone, cutoff_hz=300, sample_rate=sample_rate)

    raw_noise = brown_noise(duration_sec, amplitude=0.15,
                            sample_rate=sample_rate)
    raw_noise = lowpass(raw_noise, cutoff_hz=200, sample_rate=sample_rate)
    raw_noise = highpass(raw_noise, cutoff_hz=20, sample_rate=sample_rate)

    combined = raw_drone + raw_noise
    stereo = mono_to_stereo(combined)

    amp_mod = drift(duration_sec, speed=_DRIFT_SPEED_SLOW,
                    depth=0.06, offset=1.0,
                    seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    stereo = reverb(stereo, room_size=0.7, damping=0.6, wet_level=0.4,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def ethereal_wash(duration_sec: float, frequency: float = 220.0,
                  seed: int | None = None,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Saw pad drenched in reverb with drifting modulation — dreamy, floating."""
    raw = pad(frequency, duration_sec, voices=5, detune_cents=15.0,
              amplitude=0.4, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=2500, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    filter_mod = drift(duration_sec, speed=_DRIFT_SPEED_MEDIUM,
                       depth=500, offset=1500,
                       seed=seed, sample_rate=sample_rate)
    stereo = apply_filter_sweep(stereo, filter_mod, sample_rate=sample_rate)

    seed2 = (seed + 7) if seed is not None else None
    amp_mod = evolving_lfo(duration_sec, rate_hz=_LFO_RATE_SLOW,
                           depth=0.06, offset=1.0,
                           drift_speed=_DRIFT_SPEED_SLOW,
                           seed=seed2, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    stereo = chorus(stereo, rate_hz=0.3, depth=0.15, mix=0.3,
                    sample_rate=sample_rate)
    stereo = reverb(stereo, room_size=1.0, damping=0.3, wet_level=0.75,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def singing_bowl(duration_sec: float, frequency: float = 256.0,
                 seed: int | None = None,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Simulated singing bowl resonance — metallic, meditative."""
    partials = [
        (1.0,    1.0),
        (2.71,   0.6),
        (5.40,   0.35),
        (8.93,   0.15),
        (13.34,  0.08),
    ]
    raw = drone(frequency, duration_sec, harmonics=partials,
                amplitude=0.5, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=5000, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    amp_mod = evolving_lfo(duration_sec, rate_hz=_LFO_RATE_MEDIUM,
                           depth=0.07, offset=1.0,
                           drift_speed=_DRIFT_SPEED_SLOW,
                           seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, amp_mod)

    stereo = reverb(stereo, room_size=0.9, damping=0.35, wet_level=0.65,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


def noise_wash(duration_sec: float, color: str = "pink",
               seed: int | None = None,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Colored noise with evolving filter — ambient bed texture."""
    generators = {"pink": pink_noise, "brown": brown_noise, "white": white_noise}
    gen = generators.get(color)
    if gen is None:
        raise ValueError(f"Unknown noise color: {color!r}")

    raw = gen(duration_sec, amplitude=0.4, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=3000, sample_rate=sample_rate)
    raw = highpass(raw, cutoff_hz=40, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    filter_mod = evolving_lfo(duration_sec, rate_hz=0.04,
                              depth=600, offset=1200,
                              drift_speed=_DRIFT_SPEED_MEDIUM,
                              seed=seed, sample_rate=sample_rate)
    stereo = apply_filter_sweep(stereo, filter_mod, sample_rate=sample_rate)

    stereo = reverb(stereo, room_size=0.7, damping=0.5, wet_level=0.4,
                    sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)
