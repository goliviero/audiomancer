"""Progressive loopable stem — 5 minutes with organic progression.

Generates a 5-minute stem designed to be looped by ffmpeg for Akasha videos.
Unlike static textures, this stem has a clear arc: awakening → deepening →
fullness → return — then loops back seamlessly.

"Rhythmic variation" here means macro-rhythm:
  - Volume swells over 20-60s (not beats)
  - Layers entering/exiting with crossfade
  - Filter opening and closing
  - Tremolo: 0.15 Hz ≈ one breath every ~7s

Musical structure (300 seconds):
  0:00 – 1:30  AWAKENING  — Drone alone, quiet, filter closed
  1:30 – 3:00  DEEPENING  — Pad enters, filter opens, tremolo grows
  3:00 – 4:00  FULLNESS   — Full stack at peak density
  4:00 – 5:00  RETURN     — Texture exits, pad fades, filter closes, back to start

Loop point: end state = start state (same volume, same filter).
5s crossfade seals the loop with make_loopable().

Output: output/stems/progressive_stem_om_5min.wav
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

import audiomancer.quick as q
from audiomancer.compose import fade_envelope, tremolo, stitch, make_loopable
from audiomancer.modulation import apply_amplitude_mod, apply_filter_sweep
from audiomancer.layers import mix, normalize_lufs
from audiomancer.utils import export_wav

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DURATION = 300          # 5 minutes
SR = 44100
SEED = 42
OUT = project_root / "output" / "stems"

# Musical constants — easy to change
FREQ = 136.1            # Om frequency (swap for 111.0 Holy, 432.0, 528.0, etc.)
LOOP_CROSSFADE = 5.0    # seconds to seal the loop point


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_stem() -> np.ndarray:
    print(f"  Frequency: {FREQ} Hz")
    print(f"  Duration: {DURATION}s ({DURATION//60} min)")
    print()

    # -----------------------------------------------------------------
    # 1. Generate raw layers (full duration)
    # -----------------------------------------------------------------
    print("[1/5] Generating layers...")

    print("  drone...")
    drone_raw = q.drone(FREQ, DURATION,
                        harmonics=q.HARMONICS_WARM,
                        cutoff_hz=2500,
                        seed=SEED)

    print("  pad (A minor: 136 / 165 / 220 Hz)...")
    pad_raw = q.pad([136.1, 165.0, 220.0], DURATION,
                    voices=4, detune_cents=10.0, dark=True)

    print("  texture (noise_wash)...")
    texture_raw = q.texture("noise_wash", DURATION, seed=SEED + 1)

    print("  binaural (Om theta, -14 dB)...")
    bbeat = q.binaural("om_theta", DURATION, volume_db=-14.0)

    # -----------------------------------------------------------------
    # 2. Volume envelopes per layer
    # Design: end values = start values → loop-friendly
    # -----------------------------------------------------------------
    print("[2/5] Applying volume envelopes...")

    # Drone: quiet → full (60s) → sustain → fade back (last 30s)
    drone_vol = fade_envelope([
        (0,   0.25),   # start quiet (loop entry point)
        (60,  1.0),    # full volume at 1:00
        (240, 1.0),    # sustain until 4:00
        (270, 0.5),    # start fade
        (300, 0.25),   # back to start level (seamless loop)
    ], DURATION, sample_rate=SR)
    drone_raw = apply_amplitude_mod(drone_raw, drone_vol)

    # Pad: silent → enter at 1:30 → exit at 4:30
    pad_vol = fade_envelope([
        (0,   0.0),
        (80,  0.0),    # still silent at 1:20
        (100, 0.8),    # full entry by 1:40
        (240, 0.8),    # sustain until 4:00
        (270, 0.0),    # faded out by 4:30
        (300, 0.0),
    ], DURATION, sample_rate=SR)
    pad_raw = apply_amplitude_mod(pad_raw, pad_vol)

    # Texture: only present during section C (3:00–4:00)
    tex_vol = fade_envelope([
        (0,   0.0),
        (165, 0.0),    # still off at 2:45
        (190, 0.7),    # full by 3:10
        (230, 0.7),    # sustain
        (260, 0.0),    # gone by 4:20
        (300, 0.0),
    ], DURATION, sample_rate=SR)
    texture_raw = apply_amplitude_mod(texture_raw, tex_vol)

    # -----------------------------------------------------------------
    # 3. Tremolo on drone (slow breathing — felt, not heard)
    # Depth grows over the piece then recedes
    # -----------------------------------------------------------------
    print("[3/5] Applying tremolo...")

    # Split drone into 3 phases with different tremolo depths
    # Phase 1 (0-90s): barely there (depth=0.02)
    # Phase 2 (90-210s): medium (depth=0.05)
    # Phase 3 (210-300s): medium → fades out (depth=0.04)
    # Simplest approach: single tremolo + depth envelope
    # depth_mod = tremolo depth modulated by an envelope
    # We use a single tremolo call but modulate the result with a depth envelope

    tremolo_depth_curve = fade_envelope([
        (0,   0.98),   # nearly no tremolo at start
        (90,  0.95),   # ±5% during body
        (210, 0.96),
        (300, 0.98),   # back to minimal at loop point
    ], DURATION, sample_rate=SR)

    # Apply tremolo (using rate that evolves slightly)
    drone_raw = tremolo(drone_raw, rate_hz=0.15, depth=0.05,
                        seed=SEED + 2, sample_rate=SR)
    # Then apply depth curve as a second modulation
    drone_raw = apply_amplitude_mod(drone_raw, tremolo_depth_curve)

    # -----------------------------------------------------------------
    # 4. Filter sweep on drone (opens and closes with the progression)
    # -----------------------------------------------------------------
    print("[4/5] Filter sweep on drone...")

    filter_curve = fade_envelope([
        (0,   800),    # filter nearly closed at start
        (90,  2000),   # opens during build
        (150, 3000),   # fully open at peak entry
        (210, 2500),   # slight close during return prep
        (270, 1200),   # closing
        (300, 800),    # back to start (seamless loop)
    ], DURATION, sample_rate=SR)

    drone_raw = apply_filter_sweep(drone_raw, filter_curve, sample_rate=SR)

    # -----------------------------------------------------------------
    # 5. Mix, seal loop, export
    # -----------------------------------------------------------------
    print("[5/5] Mixing and sealing loop...")

    stem = mix(
        [drone_raw, pad_raw, texture_raw, bbeat],
        volumes_db=[0.0, -4.0, -8.0, 0.0],
    )

    stem = normalize_lufs(stem, target_lufs=-14.0)
    stem = make_loopable(stem, crossfade_sec=LOOP_CROSSFADE, sample_rate=SR)

    return stem


def main():
    print("=== Progressive Stem — Om 136 Hz ===")
    print()

    stem = build_stem()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "progressive_stem_om_5min.wav"
    export_wav(stem, path)

    print()
    print(f"=> {path}")
    print(f"   Duration: {stem.shape[0] / SR:.1f}s")
    print(f"   Peak: {np.max(np.abs(stem)):.3f}")
    print()
    print("Loop with ffmpeg:")
    print(f"  ffmpeg -stream_loop -1 -i {path.name} -t 10800 output_3h.wav")
    print()
    print("Tip: test the loop in VLC with 'Repeat one' — should be seamless.")


if __name__ == "__main__":
    main()
