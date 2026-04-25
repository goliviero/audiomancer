"""V007 brown noise v6 — base LP 1550 ('avion') + superpositions par pas de 500.

Base validee: LP 1550 HP 40, 6 streams brown ('on se croirait en avion').
Air layers explores: LP 2500, 3000, 3500, 4000 (pas de 500).

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
BED_LP = 1550
OUT_DIR = project_root / "output" / "V007" / "v6"


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
# Variations v6 — base 1550 + sweep air layers 2500/3000/3500/4000
# ---------------------------------------------------------------------------

def v01_pure_1550():
    """Reference: LP 1550 seul. Le 'avion' validee."""
    return mono_to_stereo(filtered_layer(BED_LP))


def v02_1550_2500_50_50():
    """50/50 LP 1550 + LP 2500."""
    return mono_to_stereo(superpose([(BED_LP, 0.50), (2500, 0.50)]))


def v03_1550_3000_50_50():
    """50/50 LP 1550 + LP 3000."""
    return mono_to_stereo(superpose([(BED_LP, 0.50), (3000, 0.50)]))


def v04_1550_3500_50_50():
    """50/50 LP 1550 + LP 3500."""
    return mono_to_stereo(superpose([(BED_LP, 0.50), (3500, 0.50)]))


def v05_1550_4000_50_50():
    """50/50 LP 1550 + LP 4000."""
    return mono_to_stereo(superpose([(BED_LP, 0.50), (4000, 0.50)]))


def v06_1550_3000_70_30():
    """Bed dominant: 70% LP 1550 + 30% LP 3000 (avion + voile d'air)."""
    return mono_to_stereo(superpose([(BED_LP, 0.70), (3000, 0.30)]))


def v07_1550_3500_70_30():
    """Bed dominant: 70% LP 1550 + 30% LP 3500 (avion + air plus haut)."""
    return mono_to_stereo(superpose([(BED_LP, 0.70), (3500, 0.30)]))


def v08_3layers_1550_2500_3500():
    """3 bandes: 50% LP 1550 + 30% LP 2500 + 20% LP 3500."""
    return mono_to_stereo(superpose([
        (BED_LP, 0.50),
        (2500, 0.30),
        (3500, 0.20),
    ]))


def v09_3layers_1550_3000_4000():
    """3 bandes ecart elargi: 50% LP 1550 + 30% LP 3000 + 20% LP 4000."""
    return mono_to_stereo(superpose([
        (BED_LP, 0.50),
        (3000, 0.30),
        (4000, 0.20),
    ]))


def v10_full_stack():
    """4 bandes stackees: 1550 + 2500 + 3000 + 3500 + 4000.

    Densite spectrale max, ratios decroissants (le bed reste dominant).
    """
    return mono_to_stereo(superpose([
        (BED_LP, 0.40),
        (2500, 0.20),
        (3000, 0.18),
        (3500, 0.13),
        (4000, 0.09),
    ]))


VARIATIONS = [
    ("01_pure_1550", v01_pure_1550, "LP 1550 seul (reference avion)"),
    ("02_1550_2500_50_50", v02_1550_2500_50_50, "50/50 LP 1550 + 2500"),
    ("03_1550_3000_50_50", v03_1550_3000_50_50, "50/50 LP 1550 + 3000"),
    ("04_1550_3500_50_50", v04_1550_3500_50_50, "50/50 LP 1550 + 3500"),
    ("05_1550_4000_50_50", v05_1550_4000_50_50, "50/50 LP 1550 + 4000"),
    ("06_1550_3000_70_30", v06_1550_3000_70_30,
     "70/30 LP 1550 + 3000 (avion + voile)"),
    ("07_1550_3500_70_30", v07_1550_3500_70_30,
     "70/30 LP 1550 + 3500 (avion + air haut)"),
    ("08_3layers_1550_2500_3500", v08_3layers_1550_2500_3500,
     "50/30/20 LP 1550 + 2500 + 3500"),
    ("09_3layers_1550_3000_4000", v09_3layers_1550_3000_4000,
     "50/30/20 LP 1550 + 3000 + 4000 (gap large)"),
    ("10_full_stack", v10_full_stack,
     "5 bandes: 1550+2500+3000+3500+4000 (40/20/18/13/9)"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise v6 — {len(VARIATIONS)} variations "
          f"x {DURATION:.0f}s @ {SR}Hz ===")
    print(f"Base: LP {BED_LP} ('avion'). Air layers: 2500, 3000, 3500, 4000.\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brownv6_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        print(f"  [OK] {name:30s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS)")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
