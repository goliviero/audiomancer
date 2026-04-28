"""V007 brown noise — 10 variations de 15s pour comparaison A/B.

Exploration: trouver le timbre brown idéal pour la version 10h sleep.
Chaque variation est peak-normalisée à -3 dBFS (comparaison équitable, sans
LUFS qui demande >30s de signal pour être fiable).

Output: output/V007/variations/V007_brown_<NN>_<name>.wav
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import highpass, lowpass, reverb
from audiomancer.modulation import (
    apply_amplitude_mod,
    apply_filter_sweep,
    drift,
    evolving_lfo,
    lfo_sine,
)
from audiomancer.synth import brown_noise, sine
from audiomancer.textures import generate as gen_texture
from audiomancer.utils import export_wav, mono_to_stereo


SR = 48000
DURATION = 15.0
SEED = 42
OUT_DIR = project_root / "output" / "V007" / "variations"


def peak_normalize(stereo: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak <= 0:
        return stereo
    target_linear = 10 ** (target_dbfs / 20)
    return stereo * (target_linear / peak)


# ---------------------------------------------------------------------------
# 10 variations
# ---------------------------------------------------------------------------

def v01_sub_rumble():
    """Brown très dark — LP 600, HP 25. Pur sub, aucun FX."""
    raw = brown_noise(DURATION, amplitude=0.5, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=600, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=25, sample_rate=SR)
    return mono_to_stereo(raw)


def v02_warm_body():
    """Brown warm — LP 1500, HP 30. Corps chaud, aucun FX."""
    raw = brown_noise(DURATION, amplitude=0.5, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=1500, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=30, sample_rate=SR)
    return mono_to_stereo(raw)


def v03_full_open():
    """Brown ouvert — LP 4000, HP 40. Tire vers pink/wide-band, aucun FX."""
    raw = brown_noise(DURATION, amplitude=0.5, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=4000, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=40, sample_rate=SR)
    return mono_to_stereo(raw)


def v04_v007_actuel():
    """Preset noise_wash brown — config V007 actuelle (LP 3000, sweep, reverb 0.7)."""
    return gen_texture("noise_wash", duration_sec=DURATION,
                       seed=SEED, sample_rate=SR, color="brown")


def v05_deep_space():
    """Preset deep_space — brown LP 800 + drift + reverb 1.0/0.7."""
    return gen_texture("deep_space", duration_sec=DURATION,
                       seed=SEED, sample_rate=SR)


def v06_cathedral_hall():
    """Brown LP 3500 + reverb 0.95/0.6/damping 0.3 — cathédrale."""
    raw = brown_noise(DURATION, amplitude=0.4, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=3500, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=50, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    stereo = reverb(stereo, room_size=0.95, damping=0.3, wet_level=0.6,
                    sample_rate=SR)
    return stereo


def v07_breathing_brown():
    """Brown LP 2200 + slow amp breath (0.04Hz, ±8%) + reverb 0.5/0.25."""
    raw = brown_noise(DURATION, amplitude=0.45, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=2200, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=30, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    breath = lfo_sine(DURATION, rate_hz=0.04, depth=0.08, offset=1.0,
                      sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, breath)
    stereo = reverb(stereo, room_size=0.5, damping=0.5, wet_level=0.25,
                    sample_rate=SR)
    return stereo


def v08_earth_hum():
    """Preset earth_hum — sub-bass 60Hz drone + brown noise bed."""
    return gen_texture("earth_hum", duration_sec=DURATION,
                       seed=SEED, sample_rate=SR, frequency=60.0)


def v09_brown_plus_sub():
    """Brown LP 1800 + sine 60Hz subliminal -18dB. Grounding sleep."""
    raw = brown_noise(DURATION, amplitude=0.5, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=1800, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=30, sample_rate=SR)

    sub = sine(60.0, DURATION, amplitude=0.6, sample_rate=SR)
    sub_gain = 10 ** (-18 / 20)
    combined = raw + sub * sub_gain
    return mono_to_stereo(combined)


def v10_slow_sweep():
    """Brown LP base + wide slow filter sweep (800-3500Hz, 0.02Hz = 50s cycle)."""
    raw = brown_noise(DURATION, amplitude=0.5, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=35, sample_rate=SR)
    stereo = mono_to_stereo(raw)

    # Sweep 800-3500Hz : center 2150, depth 1350
    filter_mod = evolving_lfo(DURATION, rate_hz=0.02, depth=1350,
                              offset=2150, drift_speed=0.05,
                              seed=SEED, sample_rate=SR)
    stereo = apply_filter_sweep(stereo, filter_mod, sample_rate=SR)
    stereo = reverb(stereo, room_size=0.6, damping=0.5, wet_level=0.3,
                    sample_rate=SR)
    return stereo


VARIATIONS = [
    ("01_sub_rumble", v01_sub_rumble, "LP 600 - pur sub, 0 FX"),
    ("02_warm_body", v02_warm_body, "LP 1500 - corps chaud, 0 FX"),
    ("03_full_open", v03_full_open, "LP 4000 - ouvert pink-feel, 0 FX"),
    ("04_v007_actuel", v04_v007_actuel, "noise_wash preset (config V007)"),
    ("05_deep_space", v05_deep_space, "deep_space preset (LP 800 + reverb 1.0)"),
    ("06_cathedral", v06_cathedral_hall, "LP 3500 + reverb 0.95 cathedral"),
    ("07_breathing", v07_breathing_brown, "LP 2200 + breath 0.04Hz + light reverb"),
    ("08_earth_hum", v08_earth_hum, "earth_hum preset (60Hz drone + brown)"),
    ("09_brown_sub", v09_brown_plus_sub, "LP 1800 + sub 60Hz sine -18dB"),
    ("10_slow_sweep", v10_slow_sweep, "wide filter sweep 800-3500Hz / 50s"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise — 10 variations x {DURATION:.0f}s @ {SR}Hz ===")
    print(f"Output: {OUT_DIR.relative_to(project_root)}\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brown_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        print(f"  [OK] {name:18s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS)")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
