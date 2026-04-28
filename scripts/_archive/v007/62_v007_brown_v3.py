"""V007 brown noise v3 — focus WARM (autour de v2/02 LP 1500, en plus warm).

Constat: brown noise pur (-6dB/oct) charge le sub -> sonne "trop deep, trop bas".
Pour reduire la sensation deep/sombre sans perdre le caractere brown:
- Monter le LP cutoff (1700-2700) -> plus d'aigus preserves
- Monter le HP cutoff (40 -> 80-100) -> coupe le sub mud
- Blend avec pink noise -> pink a un -3dB/oct, plus de midrange present

Toutes 6 streams (densite v2/02), DRY (no reverb), peak -3 dBFS.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import highpass, lowpass
from audiomancer.synth import brown_noise, pink_noise
from audiomancer.utils import export_wav, mono_to_stereo


SR = 48000
DURATION = 15.0
OUT_DIR = project_root / "output" / "V007" / "v3"


def dense_brown(n_streams: int, duration: float = DURATION,
                sample_rate: int = SR) -> np.ndarray:
    summed = np.zeros(int(duration * sample_rate))
    for _ in range(n_streams):
        summed += brown_noise(duration, amplitude=1.0, sample_rate=sample_rate)
    summed = highpass(summed, cutoff_hz=10.0, sample_rate=sample_rate)
    return summed


def dense_pink(n_streams: int, duration: float = DURATION,
               sample_rate: int = SR) -> np.ndarray:
    summed = np.zeros(int(duration * sample_rate))
    for _ in range(n_streams):
        summed += pink_noise(duration, amplitude=1.0, sample_rate=sample_rate)
    return summed


def peak_normalize(stereo: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak <= 0:
        return stereo
    target_linear = 10 ** (target_dbfs / 20)
    return stereo * (target_linear / peak)


# ---------------------------------------------------------------------------
# Variations v3 — autour de LP 1500 v2/02, exploration warm
# ---------------------------------------------------------------------------

def v01_warm_1700():
    """6 streams + LP 1700 HP 40 — un cran au-dessus du v2/02."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=1700, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=40, sample_rate=SR)
    return mono_to_stereo(raw)


def v02_warm_2000():
    """6 streams + LP 2000 HP 40."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=2000, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=40, sample_rate=SR)
    return mono_to_stereo(raw)


def v03_warm_2300():
    """6 streams + LP 2300 HP 40 — vers le sweet spot warm."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=2300, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=40, sample_rate=SR)
    return mono_to_stereo(raw)


def v04_warm_2700():
    """6 streams + LP 2700 HP 40 — la limite haute brown (apres tire vers pink)."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=2700, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=40, sample_rate=SR)
    return mono_to_stereo(raw)


def v05_warm_1800_hp80():
    """6 streams + LP 1800 + HP 80 — coupe le sub mud, presence accrue."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=1800, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=80, sample_rate=SR)
    return mono_to_stereo(raw)


def v06_warm_2200_hp80():
    """6 streams + LP 2200 + HP 80 — moins de sub, plus de body."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=2200, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=80, sample_rate=SR)
    return mono_to_stereo(raw)


def v07_warm_2500_hp100():
    """6 streams + LP 2500 + HP 100 — presence max, le plus 'in your face'."""
    raw = dense_brown(6)
    raw = lowpass(raw, cutoff_hz=2500, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=100, sample_rate=SR)
    return mono_to_stereo(raw)


def v08_brown_pink_blend():
    """6 brown LP 1700 (75%) + 4 pink LP 4000 (25%) — boost mid via pink."""
    raw_brown = dense_brown(6)
    raw_brown = lowpass(raw_brown, cutoff_hz=1700, sample_rate=SR)
    raw_brown = highpass(raw_brown, cutoff_hz=40, sample_rate=SR)

    raw_pink = dense_pink(4)
    raw_pink = lowpass(raw_pink, cutoff_hz=4000, sample_rate=SR)
    raw_pink = highpass(raw_pink, cutoff_hz=80, sample_rate=SR)

    # Egalize peaks before blending
    bp = np.max(np.abs(raw_brown))
    pp = np.max(np.abs(raw_pink))
    if bp > 0:
        raw_brown = raw_brown / bp
    if pp > 0:
        raw_pink = raw_pink / pp

    blended = 0.75 * raw_brown + 0.25 * raw_pink
    return mono_to_stereo(blended)


def v09_brown_pink_blend_warm():
    """6 brown LP 2000 (70%) + 4 pink LP 5000 (30%) — encore plus de presence."""
    raw_brown = dense_brown(6)
    raw_brown = lowpass(raw_brown, cutoff_hz=2000, sample_rate=SR)
    raw_brown = highpass(raw_brown, cutoff_hz=40, sample_rate=SR)

    raw_pink = dense_pink(4)
    raw_pink = lowpass(raw_pink, cutoff_hz=5000, sample_rate=SR)
    raw_pink = highpass(raw_pink, cutoff_hz=80, sample_rate=SR)

    bp = np.max(np.abs(raw_brown))
    pp = np.max(np.abs(raw_pink))
    if bp > 0:
        raw_brown = raw_brown / bp
    if pp > 0:
        raw_pink = raw_pink / pp

    blended = 0.70 * raw_brown + 0.30 * raw_pink
    return mono_to_stereo(blended)


VARIATIONS = [
    ("01_warm_1700", v01_warm_1700, "LP 1700 HP 40"),
    ("02_warm_2000", v02_warm_2000, "LP 2000 HP 40"),
    ("03_warm_2300", v03_warm_2300, "LP 2300 HP 40"),
    ("04_warm_2700", v04_warm_2700, "LP 2700 HP 40 (limite haute brown)"),
    ("05_warm_1800_hp80", v05_warm_1800_hp80, "LP 1800 HP 80 - coupe sub mud"),
    ("06_warm_2200_hp80", v06_warm_2200_hp80, "LP 2200 HP 80 - body+presence"),
    ("07_warm_2500_hp100", v07_warm_2500_hp100,
     "LP 2500 HP 100 - max presence, le plus 'in your face'"),
    ("08_brown_pink_blend", v08_brown_pink_blend,
     "75% brown LP 1700 + 25% pink LP 4000"),
    ("09_brown_pink_warm", v09_brown_pink_blend_warm,
     "70% brown LP 2000 + 30% pink LP 5000"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise v3 — {len(VARIATIONS)} variations "
          f"x {DURATION:.0f}s @ {SR}Hz ===")
    print("Focus: + warm que v2/02 (LP 1500). Toutes DRY, 6 streams.\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brownv3_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        print(f"  [OK] {name:24s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS)")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
