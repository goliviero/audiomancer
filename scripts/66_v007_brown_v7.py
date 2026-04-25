"""V007 brown noise v7 — STEREO decorrele + sub layer + multi-layer cacophony.

Reference description: 'many STEREO brown noise samples smoothed and deepened
to varying amounts and then layered together into a sleepy cacophony of deep
rumbling sounds.'

Cles:
1. Chaque layer est STEREO decorrele (L et R = streams independants).
   -> Image stereo large, immersion 'cacophony'.
2. Plusieurs layers avec LP cutoffs varies (= 'deepened to varying amounts').
3. Layer tres grave (LP 500) explore a differents dB.

Base validee: v6/08 = 50/30/20 LP 1550 + 2500 + 3500 (mono).
On part de la et on ajoute stereo + sub variations + cacophonie.

Toutes 6 streams par canal par layer, HP 40, DRY, peak -3 dBFS.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import highpass, lowpass
from audiomancer.synth import brown_noise
from audiomancer.utils import export_wav


SR = 48000
DURATION = 15.0
HP_BASE = 40
N_STREAMS = 6
OUT_DIR = project_root / "output" / "V007" / "v7"


def dense_brown_mono(n_streams: int, duration: float = DURATION,
                     sample_rate: int = SR) -> np.ndarray:
    """Sum N independent brown streams (mono) + detrend."""
    summed = np.zeros(int(duration * sample_rate))
    for _ in range(n_streams):
        summed += brown_noise(duration, amplitude=1.0, sample_rate=sample_rate)
    summed = highpass(summed, cutoff_hz=10.0, sample_rate=sample_rate)
    return summed


def stereo_layer(lp_hz: float, n_streams: int = N_STREAMS,
                 hp_hz: float = HP_BASE) -> np.ndarray:
    """Generate a STEREO decorrelated layer: L and R are independent brown streams.

    Returns ndarray (n, 2) peak-normalized to 1.0.
    """
    left = dense_brown_mono(n_streams)
    right = dense_brown_mono(n_streams)
    left = lowpass(left, cutoff_hz=lp_hz, sample_rate=SR)
    right = lowpass(right, cutoff_hz=lp_hz, sample_rate=SR)
    left = highpass(left, cutoff_hz=hp_hz, sample_rate=SR)
    right = highpass(right, cutoff_hz=hp_hz, sample_rate=SR)
    stereo = np.column_stack([left, right])
    peak = np.max(np.abs(stereo))
    return stereo / peak if peak > 0 else stereo


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def superpose_stereo(layers: list[tuple[float, float]]) -> np.ndarray:
    """layers = list of (lp_hz, db_offset). Returns stereo (n, 2)."""
    blended = np.zeros((int(DURATION * SR), 2))
    for lp_hz, db_offset in layers:
        weight = db_to_linear(db_offset)
        blended += weight * stereo_layer(lp_hz)
    return blended


def peak_normalize(stereo: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak <= 0:
        return stereo
    return stereo * (10 ** (target_dbfs / 20) / peak)


# ---------------------------------------------------------------------------
# Variations v7 — stereo decorrele + sub layer + cacophony
# ---------------------------------------------------------------------------

def v01_v6_08_stereo():
    """v6/08 mais en STEREO decorrele.

    Meme template (LP 1550 / 2500 / 3500 a 50/30/20 = ~0/-4.4/-8 dB) mais
    chaque layer a L et R independants.
    """
    return superpose_stereo([
        (1550, 0.0),
        (2500, -4.4),
        (3500, -8.0),
    ])


def v02_sub_layer_minus_12db():
    """v01 + LP 500 a -12 dB. Sub tres discret, juste un voile sous le bed."""
    return superpose_stereo([
        (500, -12.0),
        (1550, 0.0),
        (2500, -4.4),
        (3500, -8.0),
    ])


def v03_sub_layer_minus_6db():
    """v01 + LP 500 a -6 dB. Sub present mais pas dominant."""
    return superpose_stereo([
        (500, -6.0),
        (1550, 0.0),
        (2500, -4.4),
        (3500, -8.0),
    ])


def v04_sub_layer_0db():
    """v01 + LP 500 a 0 dB. Sub egal au bed -> rumble dominant."""
    return superpose_stereo([
        (500, 0.0),
        (1550, 0.0),
        (2500, -4.4),
        (3500, -8.0),
    ])


def v05_deep_dominant():
    """Sub + bed renforces vs air: 0/0 sur bas, -10/-14 sur aigus.

    Pour les amateurs de deep rumble.
    """
    return superpose_stereo([
        (500, 0.0),
        (1550, 0.0),
        (2500, -10.0),
        (3500, -14.0),
    ])


def v06_balanced_4layers():
    """4 layers a poids egaux: LP 500 / 1550 / 2500 / 3500 tous a 0 dB.

    Sub bien present, body et air aussi -> spectre tres riche.
    """
    return superpose_stereo([
        (500, 0.0),
        (1550, 0.0),
        (2500, 0.0),
        (3500, 0.0),
    ])


def v07_5layer_cacophony():
    """5 layers stereo, deepening gradient: LP 500/1100/1700/2500/3500.

    Approche fidele de 'sleepy cacophony of deep rumbling'.
    """
    return superpose_stereo([
        (500, -3.0),
        (1100, -1.0),
        (1700, 0.0),
        (2500, -3.0),
        (3500, -7.0),
    ])


def v08_6layer_max_cacophony():
    """6 layers stereo (cacophonie max): LP 400/800/1300/1900/2700/3500."""
    return superpose_stereo([
        (400, -4.0),
        (800, -2.0),
        (1300, 0.0),
        (1900, -1.0),
        (2700, -4.0),
        (3500, -8.0),
    ])


VARIATIONS = [
    ("01_v6_08_stereo", v01_v6_08_stereo,
     "v6/08 en stereo decorrele (1550/2500/3500 a 0/-4.4/-8 dB)"),
    ("02_sub_minus12db", v02_sub_layer_minus_12db,
     "v01 + sub LP 500 a -12 dB (voile sub)"),
    ("03_sub_minus6db", v03_sub_layer_minus_6db,
     "v01 + sub LP 500 a -6 dB (sub present)"),
    ("04_sub_0db", v04_sub_layer_0db,
     "v01 + sub LP 500 a 0 dB (sub = bed, rumble dominant)"),
    ("05_deep_dominant", v05_deep_dominant,
     "Sub+bed forts, aigus -10/-14 dB (deep heavy)"),
    ("06_balanced_4layers", v06_balanced_4layers,
     "4 layers a 0 dB: 500/1550/2500/3500 (spectre riche)"),
    ("07_5layer_cacophony", v07_5layer_cacophony,
     "5 layers stereo gradient: 500/1100/1700/2500/3500"),
    ("08_6layer_max_cacophony", v08_6layer_max_cacophony,
     "6 layers stereo: 400/800/1300/1900/2700/3500 (max cacophony)"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise v7 — {len(VARIATIONS)} variations "
          f"x {DURATION:.0f}s @ {SR}Hz ===")
    print("Stereo decorrele + sub layer + cacophony.\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brownv7_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        # Cross-correlation L/R as stereo width indicator
        l, r = stereo[:, 0], stereo[:, 1]
        if np.std(l) > 0 and np.std(r) > 0:
            corr = np.corrcoef(l, r)[0, 1]
        else:
            corr = 1.0
        width_label = "wide" if corr < 0.3 else "mid" if corr < 0.7 else "narrow"
        print(f"  [OK] {name:28s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS, "
              f"L/R corr={corr:+.2f} -> {width_label})")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
