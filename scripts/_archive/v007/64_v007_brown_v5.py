"""V007 brown noise v5 — superpositions deep + aere.

Base: v4/02 = 50% brown LP 1700 + 50% brown LP 2700 (validee).
Exploration: 3 bandes (deep+body+air), ecarts elargis, ratios varies.

Toutes 6 streams par layer, HP 40, DRY, peak -3 dBFS.
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
HP_BASE = 40
N_STREAMS = 6
OUT_DIR = project_root / "output" / "V007" / "v5"


def dense_brown(n_streams: int, duration: float = DURATION,
                sample_rate: int = SR) -> np.ndarray:
    summed = np.zeros(int(duration * sample_rate))
    for _ in range(n_streams):
        summed += brown_noise(duration, amplitude=1.0, sample_rate=sample_rate)
    summed = highpass(summed, cutoff_hz=10.0, sample_rate=sample_rate)
    return summed


def normalized(mono: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(mono))
    return mono / peak if peak > 0 else mono


def filtered_layer(lp_hz: float) -> np.ndarray:
    raw = dense_brown(N_STREAMS)
    raw = lowpass(raw, cutoff_hz=lp_hz, sample_rate=SR)
    raw = highpass(raw, cutoff_hz=HP_BASE, sample_rate=SR)
    return normalized(raw)


def peak_normalize(stereo: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak <= 0:
        return stereo
    return stereo * (10 ** (target_dbfs / 20) / peak)


def superpose(layers: list[tuple[float, float]]) -> np.ndarray:
    """layers = list of (lp_hz, weight). Returns mono blend."""
    blended = np.zeros(int(DURATION * SR))
    for lp_hz, weight in layers:
        blended += weight * filtered_layer(lp_hz)
    return blended


# ---------------------------------------------------------------------------
# Variations v5 — superpositions deep + aere
# ---------------------------------------------------------------------------

def v01_3band_bed_body_air():
    """3 bandes: 40% LP 1500 (deep) + 30% LP 2200 (body) + 30% LP 3000 (air)."""
    return mono_to_stereo(superpose([
        (1500, 0.40),
        (2200, 0.30),
        (3000, 0.30),
    ]))


def v02_3band_balanced():
    """3 bandes egales: LP 1500 + LP 2300 + LP 3200 (33/33/33)."""
    return mono_to_stereo(superpose([
        (1500, 0.34),
        (2300, 0.33),
        (3200, 0.33),
    ]))


def v03_wider_gap_1500_3000():
    """2 bandes ecart elargi: 50% LP 1500 + 50% LP 3000.

    Plus deep que v4/02 (bed 1500 vs 1700) ET plus aere (air 3000 vs 2700).
    """
    return mono_to_stereo(superpose([
        (1500, 0.50),
        (3000, 0.50),
    ]))


def v04_even_wider_1500_3500():
    """2 bandes ecart maximum: 50% LP 1500 + 50% LP 3500.

    Sous-bass + air marque, pas de body intermediaire (V-shape).
    """
    return mono_to_stereo(superpose([
        (1500, 0.50),
        (3500, 0.50),
    ]))


def v05_centered_higher_1800_2800():
    """2 bandes shiftees up: 50% LP 1800 + 50% LP 2800.

    Globalement + warm que v4/02.
    """
    return mono_to_stereo(superpose([
        (1800, 0.50),
        (2800, 0.50),
    ]))


def v06_air_dominant_40_60():
    """v4/02 inverse: 40% LP 1700 + 60% LP 2700 (air dominant)."""
    return mono_to_stereo(superpose([
        (1700, 0.40),
        (2700, 0.60),
    ]))


def v07_bed_dominant_60_40():
    """v4/02 inverse: 60% LP 1700 + 40% LP 2700 (bed dominant)."""
    return mono_to_stereo(superpose([
        (1700, 0.60),
        (2700, 0.40),
    ]))


def v08_3band_deep_focus():
    """3 bandes orientees deep: 50% LP 1500 + 30% LP 2200 + 20% LP 3200.

    Body principal sur le bas, air discret en cerise.
    """
    return mono_to_stereo(superpose([
        (1500, 0.50),
        (2200, 0.30),
        (3200, 0.20),
    ]))


VARIATIONS = [
    ("01_3band_bed_body_air", v01_3band_bed_body_air,
     "3 bandes: 40% 1500 + 30% 2200 + 30% 3000"),
    ("02_3band_balanced", v02_3band_balanced,
     "3 bandes egales: 1500 + 2300 + 3200"),
    ("03_wider_gap_1500_3000", v03_wider_gap_1500_3000,
     "50/50 LP 1500 + 3000 (+ deep + aere que v4/02)"),
    ("04_even_wider_1500_3500", v04_even_wider_1500_3500,
     "50/50 LP 1500 + 3500 (V-shape)"),
    ("05_centered_higher", v05_centered_higher_1800_2800,
     "50/50 LP 1800 + 2800 (+ warm global)"),
    ("06_air_60_40", v06_air_dominant_40_60,
     "40/60 LP 1700+2700 (air dominant)"),
    ("07_bed_60_40", v07_bed_dominant_60_40,
     "60/40 LP 1700+2700 (bed dominant)"),
    ("08_3band_deep_focus", v08_3band_deep_focus,
     "50/30/20 LP 1500+2200+3200 (deep + cerise air)"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise v5 — {len(VARIATIONS)} variations "
          f"x {DURATION:.0f}s @ {SR}Hz ===")
    print("Focus: deep + aere. Superpositions 2-3 bandes autour de v4/02.\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brownv5_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        print(f"  [OK] {name:26s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS)")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
