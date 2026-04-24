"""V005 pad D2 ALIVE — same chord, but timbre morphs over time (non-static).

Problem: a constant sustain of identical sines = painful ("note continue horrible").
Fix: add spectral + spatial movement. Same D2 chord (C2 + C3 + E4 + G5), but:

  1. Filter sweep (random_walk LP cutoff 1000-4000 Hz, non-periodic)
  2. Per-voice amplitude modulation at DIFFERENT rates — voices take turns
     dominating, creating constant timbral morphing
  3. Auto-pan on the stereo field (slow, 0.04 Hz)
  4. Multi-scale volume breathing (unchanged)

30s each. Renders 3 variants with increasing amounts of movement:
    D2_alive_A : gentle (subtle filter + light voice drift)
    D2_alive_B : moderate (moderate filter + distinct voice drift)
    D2_alive_C : strong (wide filter + strong voice drift + auto-pan)

Usage:
    python scripts/23_v005_pad_alive.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable
from audiomancer.effects import chorus_subtle, lowpass, reverb
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import (
    apply_amplitude_mod,
    apply_filter_sweep,
    evolving_lfo,
    multi_lfo,
    random_walk,
)
from audiomancer.spatial import auto_pan
from audiomancer.synth import sine
from audiomancer.utils import export_wav, mono_to_stereo

SR = 48000
DUR = 30
OUT = project_root / "output" / "V005" / "variants"

# D2 voicing: C2, C3, E4, G5
CHORD = [66.0, 132.0, 330.0, 792.0]


def _voice(freq: float, duration: float, seed: int) -> np.ndarray:
    """Single voice with 3 detuned sines (mini chord_pad inlined)."""
    rng = np.random.default_rng(seed)
    offsets = np.array([-2.0, 0.0, 2.0]) + rng.uniform(-1.0, 1.0, size=3)
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    sig = np.zeros_like(t)
    for cents in offsets:
        detuned = freq * 2 ** (cents / 1200)
        sig += np.sin(2 * np.pi * detuned * t)
    return sig / 3.0


def build_alive_pad(intensity: str = "moderate", seed: int = 42,
                    duration: float = DUR,
                    chord: list[float] | None = None) -> np.ndarray:
    """Build D2 pad with spectral + spatial movement.

    intensity: 'gentle' | 'moderate' | 'strong'
    duration: total duration in seconds (default 30 for variant preview).
    chord: override the default 4-note chord (C2, C3, E4, G5).
    """
    if chord is None:
        chord = CHORD
    # --- Movement parameters by intensity ---
    if intensity == "gentle":
        filter_center = 2800
        filter_depth = 800       # LP sweep 2000-3600 Hz
        voice_mod_depth = 0.15   # +-15% voice amplitude
        use_pan = False
    elif intensity == "moderate":
        filter_center = 2500
        filter_depth = 1500      # LP sweep 1000-4000 Hz
        voice_mod_depth = 0.35   # +-35% voice amplitude
        use_pan = True
    else:  # strong
        filter_center = 2200
        filter_depth = 1800      # LP sweep 400-4000 Hz
        voice_mod_depth = 0.55   # +-55% voice amplitude
        use_pan = True

    # --- Build each voice, modulated at a DIFFERENT rate so they take turns ---
    # Rates are prime-ish in seconds to avoid beat alignment
    voice_rates = [1 / 19.0, 1 / 29.0, 1 / 13.0, 1 / 23.0]  # Hz
    voice_amps = [0.9, 0.8, 0.5, 0.35]  # C2 strongest, G5 softest

    signal = np.zeros(int(duration * SR))
    # Truncate or pad voice_rates/amps to match chord length
    n_voices = len(chord)
    voice_rates = voice_rates[:n_voices]
    voice_amps = voice_amps[:n_voices]
    for i, (freq, rate, base_amp) in enumerate(zip(chord, voice_rates, voice_amps)):
        raw = _voice(freq, duration, seed=seed + i * 17)
        # Amplitude modulation per voice — each voice has its own rhythm
        mod = evolving_lfo(
            duration, rate_hz=rate, depth=voice_mod_depth, offset=1.0,
            drift_speed=0.05,
            seed=seed + i * 31, sample_rate=SR,
        )
        signal += raw * base_amp * mod

    # Normalize before filter to avoid clipping
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * 0.85 / peak

    # --- Dynamic LP filter: random_walk around filter_center ---
    walk = random_walk(duration, sigma=0.3, tau=4.0,
                       seed=seed + 7, sample_rate=SR)
    # walk is centered on 1.0; rescale to [filter_center - depth, filter_center + depth]
    filter_curve = filter_center + filter_depth * (walk - 1.0)
    filter_curve = np.clip(filter_curve, 400, 6000)
    signal = apply_filter_sweep(signal, filter_curve, sample_rate=SR)

    stereo = mono_to_stereo(signal)

    # --- Auto-pan for spatial movement ---
    if use_pan:
        stereo = auto_pan(stereo, rate_hz=0.04, depth=0.35, center=0.0,
                          sample_rate=SR)

    # --- Space ---
    stereo = chorus_subtle(stereo, sample_rate=SR)
    stereo = reverb(stereo, room_size=0.80, damping=0.6, wet_level=0.50,
                    sample_rate=SR)

    # Multi-scale breathing (gentle overall volume movement)
    breath = multi_lfo(
        duration, layers=[(1 / 8.0, 0.02), (1 / 25.0, 0.06)],
        seed=seed, sample_rate=SR,
    )
    stereo = apply_amplitude_mod(stereo, breath)

    stereo = normalize_lufs(stereo, target_lufs=-14.0, sample_rate=SR)
    stereo = master_chain(stereo, sample_rate=SR)
    stereo = make_loopable(stereo, crossfade_sec=3.0, sample_rate=SR)
    return stereo


VARIANTS = ["gentle", "moderate", "strong"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== V005 Pad D2 ALIVE - {DUR}s x {len(VARIANTS)} ===")
    print(f"  Base chord: C2 + C3 + E4 + G5")
    print(f"  Movement: filter sweep + per-voice modulation + auto-pan")
    print()

    for intensity in VARIANTS:
        label = f"D2_alive_{intensity}"
        print(f"  Rendering {label}...")
        stem = build_alive_pad(intensity)
        path = OUT / f"V005_pad_{label}.wav"
        export_wav(stem, path, sample_rate=SR)
        peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
        print(f"    -> {path.name}  peak={peak_db:+5.1f} dBFS")

    print()
    print("Listening guide:")
    print("  D2_alive_gentle   = subtle movement (LP 2-3.6k, voices +-15%)")
    print("  D2_alive_moderate = moderate movement + auto-pan (LP 1-4k, voices +-35%)")
    print("  D2_alive_strong   = strong morphing + pan (LP 0.4-4k, voices +-55%)")


if __name__ == "__main__":
    main()
