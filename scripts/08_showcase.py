"""Showcase — audition de toutes les capacités actuelles d'audiomancer.

Génère des clips courts (15s) dans output/showcase/.

Usage:
    python scripts/08_showcase.py              # tout générer
    python scripts/08_showcase.py drones       # seulement les drones
    python scripts/08_showcase.py textures layers
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

import audiomancer.quick as q
from audiomancer.layers import normalize_lufs
from audiomancer.textures import REGISTRY
from audiomancer.utils import export_wav, fade_in, fade_out

DUR = 15.0
OUT = project_root / "output" / "showcase"


def export(signal: np.ndarray, name: str, category: str) -> None:
    folder = OUT / category
    folder.mkdir(parents=True, exist_ok=True)
    sig = fade_in(fade_out(signal, 2.0), 2.0)
    sig = normalize_lufs(sig, target_lufs=-14.0)
    path = folder / f"{name}.wav"
    export_wav(sig, path)
    print(f"  -> {path.relative_to(project_root)}")


# ===========================================================================
# DRONES — 12 fréquences différentes
# Solfège, fréquences sacrées, notes standard
# ===========================================================================

DRONES = [
    # (freq_hz, nom_fichier, harmonics_preset, cutoff_hz)
    # --- Graves ---
    (40.0,   "drone_40hz_gamma",        q.HARMONICS_WARM,   400),
    (60.0,   "drone_60hz_earth_hum",    q.HARMONICS_DARK,   600),
    (111.0,  "drone_111hz_holy",        q.HARMONICS_WARM,   1500),
    (136.1,  "drone_136hz_om",          q.HARMONICS_WARM,   1800),
    (174.0,  "drone_174hz_solfege_ut",  q.HARMONICS_WARM,   2000),
    # --- Medium ---
    (256.0,  "drone_256hz_c4_pure",     q.HARMONICS_BRIGHT, 3000),
    (285.0,  "drone_285hz_solfege_re",  q.HARMONICS_WARM,   2500),
    (396.0,  "drone_396hz_solfege_mi",  q.HARMONICS_WARM,   3000),
    (432.0,  "drone_432hz_sacred_a",    q.HARMONICS_BRIGHT, 3000),
    # --- Aigus ---
    (528.0,  "drone_528hz_solfege_sol", q.HARMONICS_WARM,   4000),
    (639.0,  "drone_639hz_solfege_la",  q.HARMONICS_BRIGHT, 4500),
    (741.0,  "drone_741hz_solfege_si",  q.HARMONICS_DARK,   5000),
    # --- Bol tibétain (partiels inharmoniques) ---
    (256.0,  "drone_256hz_singing_bowl", q.HARMONICS_BOWL,  6000),
]


def demo_drones():
    print(f"\n[drones] {len(DRONES)} fréquences")
    for freq, name, harmonics, cutoff in DRONES:
        print(f"  {freq} Hz — {name}...")
        sig = q.drone(freq, DUR, harmonics=harmonics, cutoff_hz=cutoff, seed=42)
        export(sig, name, "drones")


# ===========================================================================
# BINAURAL
# ===========================================================================

def demo_binaural():
    print("\n[binaural]")

    presets = [
        ("theta_deep",    "theta_4hz"),
        ("alpha_relax",   "alpha_10hz"),
        ("delta_sleep",   "delta_2hz"),
        ("om_theta",      "om_theta_136hz"),
        ("solfeggio_528", "solfeggio_528hz"),
    ]
    for preset, name in presets:
        sig = q.binaural(preset, DUR, volume_db=0.0)
        export(sig, name, "binaural")

    # Custom
    sig = q.binaural_custom(111.0, 4.0, DUR, volume_db=0.0)
    export(sig, "custom_111hz_4hz_theta", "binaural")

    # Drone + binaural intégré
    drone_sig = q.drone(136.1, DUR, seed=1)
    bbeat = q.binaural("om_theta", DUR, volume_db=-8.0)
    combined = q.mix([(drone_sig, 0.0), (bbeat, 0.0)], target_lufs=-14.0)
    export(combined, "drone_136hz_with_binaural", "binaural")


# ===========================================================================
# TEXTURES — les 9 presets évolutifs
# ===========================================================================

def demo_textures():
    print(f"\n[textures] {len(REGISTRY)} presets")
    for name in REGISTRY:
        print(f"  {name}...")
        sig = q.texture(name, DUR, seed=42)
        export(sig, name, "textures")


# ===========================================================================
# LAYERS — combos prêts pour Akasha
# ===========================================================================

def demo_layers():
    print("\n[layers]")

    # --- Holy Frequency stack ---
    print("  holy_frequency_stack (111 Hz + theta + noise)...")
    sig = q.mix([
        (q.drone(111.0, DUR, seed=1), 0.0),
        (q.binaural_custom(111.0, 4.0, DUR, volume_db=-8.0), 0.0),
        (q.texture("noise_wash", DUR, seed=2), -12.0),
    ])
    export(sig, "01_holy_frequency_stack", "layers")

    # --- Om meditation ---
    print("  om_meditation (136 Hz Om + theta + ocean)...")
    sig = q.mix([
        (q.drone(136.1, DUR, seed=3), 0.0),
        (q.binaural("om_theta", DUR, volume_db=-8.0), 0.0),
        (q.texture("ocean_bed", DUR, seed=4), -10.0),
    ])
    export(sig, "02_om_meditation", "layers")

    # --- Space meditation ---
    print("  space_meditation (deep_space + crystal shimmer)...")
    sig = q.mix([
        (q.texture("deep_space", DUR, seed=5), 0.0),
        (q.texture("crystal_shimmer", DUR, seed=6), -14.0),
        (q.binaural("theta_deep", DUR, volume_db=-12.0), 0.0),
    ])
    export(sig, "03_space_meditation", "layers")

    # --- Earth & Sky ---
    print("  earth_sky (earth_hum + ethereal_wash)...")
    sig = q.mix([
        (q.texture("earth_hum", DUR, seed=7), 0.0),
        (q.texture("ethereal_wash", DUR, seed=8), -6.0),
    ])
    export(sig, "04_earth_sky", "layers")

    # --- Solfège 528 ---
    print("  solfege_528hz (drone 528 + binaural + breathing pad)...")
    sig = q.mix([
        (q.drone(528.0, DUR, harmonics=q.HARMONICS_WARM, cutoff_hz=4000, seed=9), 0.0),
        (q.binaural("solfeggio_528", DUR, volume_db=-8.0), 0.0),
        (q.pad([528.0, 660.0, 792.0], DUR), -10.0),
    ])
    export(sig, "05_solfege_528hz_love_freq", "layers")

    # --- Sleep (delta + dark) ---
    print("  sleep_delta (delta 2Hz + dark pad + brown noise)...")
    sig = q.mix([
        (q.drone(111.0, DUR, harmonics=q.HARMONICS_DARK, cutoff_hz=600, seed=10), 0.0),
        (q.binaural("delta_sleep", DUR, volume_db=-8.0), 0.0),
        (q.texture("ocean_bed", DUR, seed=11), -8.0),
    ])
    export(sig, "06_sleep_delta", "layers")

    # --- Alpha focus ---
    print("  alpha_focus (alpha 10Hz + breathing pad + ocean)...")
    sig = q.mix([
        (q.binaural("alpha_relax", DUR, volume_db=0.0), 0.0),
        (q.pad([261.63, 329.63, 392.0], DUR), -8.0),        # C majeur
        (q.texture("ocean_bed", DUR, seed=12), -12.0),
    ])
    export(sig, "07_alpha_focus", "layers")

    # --- Akasha style complet ---
    print("  akasha_full_stack (Om + theta + pad + texture)...")
    sig = q.mix([
        (q.drone(136.1, DUR, seed=13), 0.0),
        (q.binaural("om_theta", DUR, volume_db=-8.0), 0.0),
        (q.texture("breathing_pad", DUR, seed=14), -10.0),
        (q.texture("noise_wash", DUR, seed=15), -18.0),
    ])
    export(sig, "08_akasha_full_stack", "layers")

    # --- Sacred geometry (528 + 432 dual) ---
    print("  sacred_dual (432 Hz + 528 Hz drones layered)...")
    sig = q.mix([
        (q.drone(432.0, DUR, harmonics=q.HARMONICS_WARM, cutoff_hz=3000, seed=16), 0.0),
        (q.drone(528.0, DUR, harmonics=q.HARMONICS_DARK, cutoff_hz=2500, seed=17), -8.0),
        (q.binaural("solfeggio_528", DUR, volume_db=-12.0), 0.0),
    ])
    export(sig, "09_sacred_dual_432_528", "layers")

    # --- Minimal (une texture, un binaural) ---
    print("  minimal (ethereal_wash + alpha)...")
    sig = q.mix([
        (q.texture("ethereal_wash", DUR, seed=18), 0.0),
        (q.binaural("alpha_relax", DUR, volume_db=-10.0), 0.0),
    ])
    export(sig, "10_minimal_ethereal_alpha", "layers")


# ===========================================================================

CATEGORIES = {
    "drones":   demo_drones,
    "binaural": demo_binaural,
    "textures": demo_textures,
    "layers":   demo_layers,
}


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(CATEGORIES.keys())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT}")
    print(f"Duration: {DUR}s per clip | Categories: {', '.join(selected)}")

    for cat in selected:
        if cat not in CATEGORIES:
            print(f"  Unknown category: {cat!r}. Available: {', '.join(CATEGORIES)}")
            continue
        CATEGORIES[cat]()

    files = list(OUT.rglob("*.wav"))
    print(f"\nDone — {len(files)} clips total in {OUT}/")


if __name__ == "__main__":
    main()
