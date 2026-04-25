"""Generic stem renderer — config-driven.

Usage:
    python scripts/render_stem.py --config V005 --stem warm_pad
    python scripts/render_stem.py --config V005 --stem warm_pad --preview
    python scripts/render_stem.py --config V005 --stem warm_pad --vary --duration 120

Looks up configs/<config>.py for META + STEMS dict. Each stem names a builder
registered in audiomancer/builders.py (REGISTRY) plus its params.

Output: output/<config>/<config>_<stem>[<suffix>].wav
"""

import argparse
import importlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.builders import REGISTRY, derived_seed
from audiomancer.compose import apply_pre_fade, make_loopable, verify_loop
from audiomancer.layers import normalize_lufs
from audiomancer.mastering import ambient_master_chain, master_chain
from audiomancer.utils import export_wav


def main():
    parser = argparse.ArgumentParser(
        description="Render a stem from a config file."
    )
    parser.add_argument("--config", required=True,
                        help="Config name (e.g. V005) — resolves to configs/<name>.py")
    parser.add_argument("--stem", required=True,
                        help="Stem name within the config's STEMS dict")
    parser.add_argument("--duration", type=int,
                        help="Override config duration (seconds)")
    parser.add_argument("--preview", action="store_true",
                        help="Quick 30s preview")
    parser.add_argument("--vary", action="store_true",
                        help="Random root seed (different micro-variations)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Root seed (overridden by --vary)")
    args = parser.parse_args()

    try:
        cfg_mod = importlib.import_module(f"configs.{args.config}")
    except ModuleNotFoundError as e:
        parser.error(f"Config not found: configs/{args.config}.py ({e})")

    meta = cfg_mod.META
    stems = cfg_mod.STEMS

    if args.stem not in stems:
        parser.error(
            f"Stem {args.stem!r} not in config {args.config}. "
            f"Available: {list(stems)}"
        )

    stem_cfg = stems[args.stem]
    builder_name = stem_cfg["builder"]
    params = stem_cfg["params"]

    if builder_name not in REGISTRY:
        parser.error(
            f"Unknown builder {builder_name!r}. "
            f"Available: {list(REGISTRY)}"
        )

    builder = REGISTRY[builder_name]
    sr = meta["sample_rate"]
    duration = 30 if args.preview else (args.duration or meta["duration"])
    root_seed = (int(np.random.default_rng().integers(0, 100000))
                 if args.vary else args.seed)
    stem_seed = derived_seed(root_seed, args.stem)

    print(f"=== {args.config} / {args.stem} ===")
    print(f"  Builder: {builder_name}")
    print(f"  Duration: {duration}s  |  seed: {stem_seed}"
          f"{' (random root)' if args.vary else ''}")
    print(f"  Params: {params}")
    print()

    # Build
    raw = builder(duration=duration, seed=stem_seed, sample_rate=sr, **params)

    # Master (default or ambient-safe) -> loop seal -> optional pre-fade
    master_mode = meta.get("master_mode", "default")
    if master_mode == "ambient":
        stem = ambient_master_chain(
            raw, target_lufs=meta["target_lufs"],
            ceiling_dbtp=meta.get("ceiling_dbtp", -3.0),
            sample_rate=sr,
        )
    else:
        stem = normalize_lufs(raw, target_lufs=meta["target_lufs"],
                              sample_rate=sr)
        stem = master_chain(stem, sample_rate=sr)
    stem = make_loopable(
        stem, crossfade_sec=5.0, sample_rate=sr,
        boundary_continuity=meta.get("loop_boundary_continuity", False),
    )
    pre_fade = meta.get("pre_fade_sec", 0.0)
    if pre_fade > 0:
        stem = apply_pre_fade(stem, fade_sec=pre_fade, sample_rate=sr)

    # Loop check
    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=sr)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    out_dir = project_root / "output" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_preview" if args.preview else ""
    path = out_dir / f"{args.config}_{args.stem}{suffix}.wav"
    bit_depth = meta.get("bit_depth", 16)
    export_wav(stem, path, sample_rate=sr, bit_depth=bit_depth)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"  -> {path.name}  ({stem.shape[0] / sr:.0f}s, peak={peak_db:.1f} dBFS, "
          f"{bit_depth}-bit)")


if __name__ == "__main__":
    main()
