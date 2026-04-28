"""V007 brown noise v4 — exploration fine autour de v3/01 (LP 1700 HP 40).

Constat: v3/05 (HP 80) et v3/06 (HP 80) sont trop presents/lourds.
-> on garde HP bas (30-50) pour eviter cette sensation.
-> on explore autour de LP 1700 (1550-1850) en finesse.
-> on ajoute la superposition 1700 + 2700 demandee.

Toutes 6 streams (sauf v08), DRY, peak -3 dBFS.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import highpass, lowpass
from audiomancer.synth import brown_noise
from audiomancer.utils import export_wav, mono_to_stereo


SR = 48000
DURATION = 15.0
OUT_DIR = project_root / "output" / "V007" / "v4"


def dense_brown(n_streams: int, duration: float = DURATION,
                sample_rate: int = SR) -> np.ndarray:
    summed = np.zeros(int(duration * sample_rate))
    for _ in range(n_streams):
        summed += brown_noise(duration, amplitude=1.0, sample_rate=sample_rate)
    summed = highpass(summed, cutoff_hz=10.0, sample_rate=sample_rate)
    return summed


def peak_normalize(stereo: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak <= 0:
        return stereo
    target_linear = 10 ** (target_dbfs / 20)
    return stereo * (target_linear / peak)


def filtered_brown(n_streams: int, lp_hz: float, hp_hz: float) -> np.ndarray:
    raw = dense_brown(n_streams)
    raw = lowpass(raw, cutoff_hz=lp_hz, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=hp_hz, sample_rate=SR)
    return raw


def normalized(mono: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(mono))
    return mono / peak if peak > 0 else mono


# ---------------------------------------------------------------------------
# Variations v4 — autour de v3/01, et superposition 1700+2700
# ---------------------------------------------------------------------------

def v01_super_70_30():
    """Superpose 70% brown LP 1700 + 30% brown LP 2700 (HP 40).

    Le 1700 domine (= base v3/01), le 2700 ajoute un peu d'air discret.
    """
    bed = filtered_brown(n_streams=6, lp_hz=1700, hp_hz=40)
    air = filtered_brown(n_streams=6, lp_hz=2700, hp_hz=40)
    blended = 0.70 * normalized(bed) + 0.30 * normalized(air)
    return mono_to_stereo(blended)


def v02_super_50_50():
    """Superpose 50% brown LP 1700 + 50% brown LP 2700 (HP 40).

    Air plus marque. Vers le warm sans perdre le bed dark.
    """
    bed = filtered_brown(n_streams=6, lp_hz=1700, hp_hz=40)
    air = filtered_brown(n_streams=6, lp_hz=2700, hp_hz=40)
    blended = 0.50 * normalized(bed) + 0.50 * normalized(air)
    return mono_to_stereo(blended)


def v03_lp_1550():
    """6 streams LP 1550 HP 40 — entre v2/02 et v3/01."""
    return mono_to_stereo(filtered_brown(6, 1550, 40))


def v04_lp_1650():
    """6 streams LP 1650 HP 40 — un poil sous v3/01."""
    return mono_to_stereo(filtered_brown(6, 1650, 40))


def v05_lp_1750():
    """6 streams LP 1750 HP 40 — un poil au-dessus v3/01."""
    return mono_to_stereo(filtered_brown(6, 1750, 40))


def v06_lp_1850():
    """6 streams LP 1850 HP 40 — vers v3/02 (LP 2000)."""
    return mono_to_stereo(filtered_brown(6, 1850, 40))


def v07_hp_30():
    """6 streams LP 1700 HP 30 — comme v3/01 mais HP plus bas (plus de sub)."""
    return mono_to_stereo(filtered_brown(6, 1700, 30))


def v08_dense_8streams():
    """8 streams LP 1700 HP 40 — meme spectre que v3/01 mais + smoothed."""
    return mono_to_stereo(filtered_brown(8, 1700, 40))


VARIATIONS = [
    ("01_super_1700_2700_70_30", v01_super_70_30,
     "70% LP 1700 + 30% LP 2700 (air discret)"),
    ("02_super_1700_2700_50_50", v02_super_50_50,
     "50% LP 1700 + 50% LP 2700 (air marque)"),
    ("03_lp_1550", v03_lp_1550, "LP 1550 HP 40 - entre v2/02 et v3/01"),
    ("04_lp_1650", v04_lp_1650, "LP 1650 HP 40 - un poil sous v3/01"),
    ("05_lp_1750", v05_lp_1750, "LP 1750 HP 40 - un poil au-dessus v3/01"),
    ("06_lp_1850", v06_lp_1850, "LP 1850 HP 40 - vers v3/02"),
    ("07_lp_1700_hp_30", v07_hp_30, "LP 1700 HP 30 - + sub vs v3/01"),
    ("08_lp_1700_8streams", v08_dense_8streams,
     "LP 1700 HP 40 + 8 streams - + smoothed"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise v4 — {len(VARIATIONS)} variations "
          f"x {DURATION:.0f}s @ {SR}Hz ===")
    print("Focus: autour de v3/01 (LP 1700 HP 40) + superposition 1700+2700.\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brownv4_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        print(f"  [OK] {name:28s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS)")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
