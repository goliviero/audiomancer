"""V007 brown noise v2 — approche MindAmend "Super Deep Smoothed".

Principe (description MindAmend) :
- Densité = sum de N streams brown noise indépendants. La loi des grands
  nombres lisse les "swells" perçus quand les basses se groupent aléatoirement.
- Deep = heavy lowpass (filtrage agressif des aigus).
- DRY = aucun reverb (pas de "loin", pas de hall).
- Optionnel: compression slow attack/release pour aplatir les variations
  d'enveloppe résiduelles.

Output: output/V007/v2/V007_brownv2_<NN>_<name>.wav
Toutes peak-normalisées à -3 dBFS pour comparaison directe.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import compress, highpass, lowpass
from audiomancer.synth import brown_noise
from audiomancer.utils import export_wav, mono_to_stereo


SR = 48000
DURATION = 15.0
OUT_DIR = project_root / "output" / "V007" / "v2"


def dense_brown(n_streams: int, duration: float = DURATION,
                sample_rate: int = SR) -> np.ndarray:
    """Sum N independent brown noise streams pour densité (MindAmend principle).

    Plus n_streams est grand, plus l'enveloppe est lisse (variance / sqrt(N)).
    """
    summed = np.zeros(int(duration * sample_rate))
    for _ in range(n_streams):
        summed += brown_noise(duration, amplitude=1.0, sample_rate=sample_rate)
    # Detrend: remove slow DC drift inherent to cumsum-based brown noise
    # (agressive HP at 10Hz suffit a tuer le trend sans toucher l'audible)
    summed = highpass(summed, cutoff_hz=10.0, sample_rate=sample_rate)
    return summed


def peak_normalize(stereo: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak <= 0:
        return stereo
    target_linear = 10 ** (target_dbfs / 20)
    return stereo * (target_linear / peak)


# ---------------------------------------------------------------------------
# Variations v2 — toutes DRY, focus densite + profondeur
# ---------------------------------------------------------------------------

def v01_dense_dark():
    """6 streams + LP 700 — deep dark dense, dry."""
    raw = dense_brown(n_streams=6)
    raw = lowpass(raw, cutoff_hz=700, sample_rate=SR)
    return mono_to_stereo(raw)


def v02_dense_warm():
    """6 streams + LP 1500 — warm dense, dry. Sweet spot probable."""
    raw = dense_brown(n_streams=6)
    raw = lowpass(raw, cutoff_hz=1500, sample_rate=SR)
    return mono_to_stereo(raw)


def v03_dense_full():
    """6 streams + LP 3000 — open dense (equivalent V007 actuel sans reverb)."""
    raw = dense_brown(n_streams=6)
    raw = lowpass(raw, cutoff_hz=3000, sample_rate=SR)
    return mono_to_stereo(raw)


def v04_super_dense_dark():
    """10 streams + LP 600 — extreme density + super deep, dry."""
    raw = dense_brown(n_streams=10)
    raw = lowpass(raw, cutoff_hz=600, sample_rate=SR)
    return mono_to_stereo(raw)


def v05_super_dense_warm():
    """10 streams + LP 1200 — extreme density warm."""
    raw = dense_brown(n_streams=10)
    raw = lowpass(raw, cutoff_hz=1200, sample_rate=SR)
    return mono_to_stereo(raw)


def v06_smoothed_compressed():
    """6 streams + LP 1000 + slow compression — replique MindAmend.

    Compression: threshold -15dB, ratio 3:1, attack 50ms, release 800ms.
    Catch les swells residuels sans casser le timbre.
    """
    raw = dense_brown(n_streams=6)
    raw = lowpass(raw, cutoff_hz=1000, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    stereo = compress(stereo, threshold_db=-15.0, ratio=3.0,
                      attack_ms=50.0, release_ms=800.0, sample_rate=SR)
    return stereo


def v07_mindamend_replica():
    """8 streams + LP 800 + slow compression — la version "Super Deep Smoothed".

    Plus dense que v06, plus deep, compression encore plus douce.
    """
    raw = dense_brown(n_streams=8)
    raw = lowpass(raw, cutoff_hz=800, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    stereo = compress(stereo, threshold_db=-12.0, ratio=2.5,
                      attack_ms=80.0, release_ms=1200.0, sample_rate=SR)
    return stereo


def v08_extreme_dark_smoothed():
    """12 streams + LP 500 + compression — version maximale deep+smooth.

    Pour les amateurs de pur sub. Equivalent "ultra deep smoothed".
    """
    raw = dense_brown(n_streams=12)
    raw = lowpass(raw, cutoff_hz=500, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    stereo = compress(stereo, threshold_db=-12.0, ratio=2.5,
                      attack_ms=80.0, release_ms=1200.0, sample_rate=SR)
    return stereo


VARIATIONS = [
    ("01_dense_dark", v01_dense_dark, "6 streams LP 700 dry"),
    ("02_dense_warm", v02_dense_warm, "6 streams LP 1500 dry"),
    ("03_dense_full", v03_dense_full, "6 streams LP 3000 dry"),
    ("04_super_dense_dark", v04_super_dense_dark, "10 streams LP 600 dry"),
    ("05_super_dense_warm", v05_super_dense_warm, "10 streams LP 1200 dry"),
    ("06_smoothed_comp", v06_smoothed_compressed,
     "6 streams LP 1000 + comp 3:1"),
    ("07_mindamend_replica", v07_mindamend_replica,
     "8 streams LP 800 + comp 2.5:1 - replique MindAmend"),
    ("08_extreme_smoothed", v08_extreme_dark_smoothed,
     "12 streams LP 500 + comp - ultra deep"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V007 brown noise v2 — {len(VARIATIONS)} variations "
          f"x {DURATION:.0f}s @ {SR}Hz ===")
    print("Approche: dense (multi-stream sum) + deep (LP) + DRY (no reverb)")
    print(f"Output: {OUT_DIR.relative_to(project_root)}\n")

    for name, fn, desc in VARIATIONS:
        try:
            stereo = fn()
        except (ValueError, RuntimeError) as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        stereo = peak_normalize(stereo, target_dbfs=-3.0)
        out_path = OUT_DIR / f"V007_brownv2_{name}.wav"
        export_wav(stereo, out_path, sample_rate=SR, bit_depth=24)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
        print(f"  [OK] {name:24s} - {desc}")
        print(f"       -> {out_path.name}  (peak={peak_db:+.1f} dBFS)")

    print(f"\nDone. {len(VARIATIONS)} variations in {OUT_DIR}")


if __name__ == "__main__":
    main()
