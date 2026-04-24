"""Generic mix renderer — config-driven.

Builds every stem in the config's MIX.volumes_db, loads external samples
(firecrack etc.), adds micro-events, applies density_profile, mixes +
masters + loop-seals.

Usage:
    python scripts/render_mix.py --config V005
    python scripts/render_mix.py --config V005 --preview    # 30s
    python scripts/render_mix.py --config V005 --vary       # random variations
    python scripts/render_mix.py --config V005 --duration 120
"""

import argparse
import importlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.builders import REGISTRY, derived_seed
from audiomancer.compose import (
    apply_pre_fade,
    density_profile,
    make_loopable,
    verify_loop,
)
from audiomancer.field import clean
from audiomancer.layers import mix as mix_layers
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import ambient_master_chain, master_chain
from audiomancer.stochastic import micro_events
from audiomancer.utils import export_wav, load_audio, normalize


def _load_firecrack(spec: dict, duration: float,
                    sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Load + clean a firecrack-style layer. Returns (stereo_section, mono_full)."""
    path = project_root / spec["path"]
    sig, _ = load_audio(path, target_sr=sample_rate)
    start = int(spec.get("offset_sec", 0.0) * sample_rate)
    end = start + int(duration * sample_rate)
    if end > len(sig):
        # Tile if source is too short
        sig_tail = sig[start:]
        section = np.tile(sig_tail, (end - start) // len(sig_tail) + 1, 0)[:end - start]
    else:
        section = sig[start:end]
    section = clean(section, sample_rate=sample_rate)
    section = normalize(section, target_db=-1.0)
    mono_full = sig.mean(axis=1) if sig.ndim == 2 else sig
    return section, mono_full


def main():
    parser = argparse.ArgumentParser(
        description="Render a full mix from a config file."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--duration", type=int,
                        help="Override config duration")
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    parser.add_argument("--vary", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        cfg_mod = importlib.import_module(f"configs.{args.config}")
    except ModuleNotFoundError as e:
        parser.error(f"Config not found: configs/{args.config}.py ({e})")

    meta = cfg_mod.META
    stems_cfg = cfg_mod.STEMS
    mix_cfg = cfg_mod.MIX

    sr = meta["sample_rate"]
    duration = 30 if args.preview else (args.duration or meta["duration"])
    root_seed = (int(np.random.default_rng().integers(0, 100000))
                 if args.vary else args.seed)

    print(f"=== {args.config} mix - {duration}s ===")
    print(f"  Root seed: {root_seed}{' (random)' if args.vary else ''}")
    print()

    # --- Build stems ---
    stems_raw = {}
    volumes_db = mix_cfg.get("volumes_db", {})
    for stem_name in volumes_db:
        if stem_name in stems_cfg:
            print(f"  [stem] {stem_name}...")
            sc = stems_cfg[stem_name]
            builder = REGISTRY[sc["builder"]]
            seed = derived_seed(root_seed, stem_name)
            stems_raw[stem_name] = builder(
                duration=duration, seed=seed, sample_rate=sr, **sc["params"])

    # --- Per-stem density envelopes (breathing / sparse / arc on specific stems) ---
    stem_envs = mix_cfg.get("stem_envelopes", {})
    for stem_name, profile_name in stem_envs.items():
        if stem_name not in stems_raw:
            continue
        print(f"  [env] {stem_name}: {profile_name}")
        env = density_profile(
            duration, profile=profile_name,
            seed=derived_seed(root_seed, f"env_{stem_name}"),
            sample_rate=sr,
        )
        env = env[:stems_raw[stem_name].shape[0]]
        if stems_raw[stem_name].ndim == 2:
            env = env[:, np.newaxis]
        stems_raw[stem_name] = stems_raw[stem_name] * env

    # --- Load external samples (firecrack etc.) ---
    firecrack_cfg = mix_cfg.get("firecrack")
    fire_mono = None
    if firecrack_cfg and "firecrack" in volumes_db:
        print(f"  [sample] firecrack ({firecrack_cfg['path']})...")
        fire_section, fire_mono = _load_firecrack(firecrack_cfg, duration, sr)
        stems_raw["firecrack"] = fire_section

    # --- Micro-events layer ---
    events_specs = mix_cfg.get("micro_events", [])
    if events_specs:
        print("  [events] scattering micro-events...")
        events_layer = micro_events(
            duration,
            event_specs=[dict(s, duration_range=tuple(s.get("duration_range", (3.0, 6.0))))
                         for s in events_specs],
            chord_freqs=mix_cfg.get("chord_freqs_for_events"),
            source=fire_mono,
            seed=derived_seed(root_seed, "micro_events"),
            sample_rate=sr,
        )
    else:
        events_layer = None

    # --- Match lengths ---
    all_arrays = list(stems_raw.values())
    if events_layer is not None:
        all_arrays.append(events_layer)
    min_len = min(a.shape[0] for a in all_arrays)
    stems_raw = {k: v[:min_len] for k, v in stems_raw.items()}
    if events_layer is not None:
        events_layer = events_layer[:min_len]

    # --- Mix ---
    mix_inputs = []
    mix_vols = []
    for name, vol in volumes_db.items():
        if name in stems_raw:
            mix_inputs.append(stems_raw[name])
            mix_vols.append(vol)
    if events_layer is not None:
        mix_inputs.append(events_layer)
        mix_vols.append(0.0)

    print(f"  [mix] {len(mix_inputs)} layers...")
    stem = mix_layers(mix_inputs, volumes_db=mix_vols)

    # --- Density profile ---
    dp_name = mix_cfg.get("density_profile", "flat")
    if dp_name != "flat":
        print(f"  [density] profile={dp_name}...")
        profile = density_profile(
            duration, profile=dp_name,
            seed=derived_seed(root_seed, "density"),
            sample_rate=sr,
        )
        stem = stem * profile[:min_len, np.newaxis]

    # --- Master (default loudness-maximizer, or ambient-safe) ---
    master_mode = meta.get("master_mode", "default")
    if master_mode == "ambient":
        print(f"  [master] ambient ({meta['target_lufs']} LUFS, "
              f"ceiling {meta.get('ceiling_dbtp', -3.0)} dBTP)")
        stem = ambient_master_chain(
            stem, target_lufs=meta["target_lufs"],
            ceiling_dbtp=meta.get("ceiling_dbtp", -3.0),
            sample_rate=sr,
        )
    else:
        stem = normalize_lufs(stem, target_lufs=meta["target_lufs"],
                              sample_rate=sr)
        stem = master_chain(stem, sample_rate=sr)

    # --- Loop seal + optional pre-fade ---
    stem = make_loopable(stem, crossfade_sec=5.0, sample_rate=sr)
    pre_fade = meta.get("pre_fade_sec", 0.0)
    if pre_fade > 0:
        stem = apply_pre_fade(stem, fade_sec=pre_fade, sample_rate=sr)

    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=sr)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  [loop] {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    out_dir = project_root / "output" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_preview" if args.preview else ""
    path = out_dir / f"{args.config}_mix{suffix}.wav"
    bit_depth = meta.get("bit_depth", 16)
    export_wav(stem, path, sample_rate=sr, bit_depth=bit_depth)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"\n  -> {path.name}  ({stem.shape[0] / sr:.0f}s, peak={peak_db:.1f} dBFS, "
          f"{bit_depth}-bit)")


if __name__ == "__main__":
    main()
