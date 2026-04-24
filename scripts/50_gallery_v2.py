"""Gallery v2 — showcase of the new capabilities added post-V005.

10 musical clips that exercise:
    Karplus-Strong, Paulstretch, saturation (tape/hiss/wow),
    micro_events, density_profile, multi_lfo, texture+piano_processed builders,
    random_walk modulation.

Output: output/gallery_v2/<label>.wav  (15-30s each, -14 LUFS, 48 kHz).
Total runtime ~1 min.

Usage:
    python scripts/50_gallery_v2.py
    python scripts/50_gallery_v2.py --only 04_ambient_scene   # single clip
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.builders import REGISTRY, derived_seed
from audiomancer.compose import density_profile, make_loopable
from audiomancer.effects import reverb_cathedral, reverb_hall
from audiomancer.layers import mix, normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import apply_amplitude_mod, multi_lfo
from audiomancer.saturation import tape_hiss, tape_saturate, vinyl_wow
from audiomancer.spectral import paulstretch
from audiomancer.stochastic import micro_events
from audiomancer.synth import karplus_strong, sine
from audiomancer.utils import export_wav, fade_in, fade_out, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "gallery_v2"


def _finalize(sig: np.ndarray, target_lufs: float = -14.0,
              fade_sec: float = 1.0) -> np.ndarray:
    """Common tail: fades + LUFS + master. No loop seal (these are demos)."""
    if sig.ndim == 1:
        sig = mono_to_stereo(sig)
    sig = fade_in(sig, fade_sec, sample_rate=SR)
    sig = fade_out(sig, fade_sec, sample_rate=SR)
    sig = normalize_lufs(sig, target_lufs=target_lufs, sample_rate=SR)
    sig = master_chain(sig, sample_rate=SR)
    return sig


# ---------------------------------------------------------------------------
# Clip builders
# ---------------------------------------------------------------------------

def clip_01_karplus_cmaj_arpeggio() -> np.ndarray:
    """Karplus-Strong plucked Cmaj arpeggio with natural decay.
    One note every ~1.5s, 8 notes over 12s. Each note has its own pluck seed.
    """
    notes = [130.81, 164.81, 196.0, 261.63, 329.63, 392.0, 261.63, 196.0]  # Cmaj
    duration = 12.0
    note_gap = 1.5  # seconds between plucks
    note_dur = 4.0  # decay tail overlap allowed

    n_samples = int(duration * SR)
    out = np.zeros(n_samples)
    for i, freq in enumerate(notes):
        start = int(i * note_gap * SR)
        if start >= n_samples:
            break
        pluck = karplus_strong(freq, note_dur, decay=0.9985,
                               brightness=0.55, amplitude=0.7,
                               seed=i * 17 + 42, sample_rate=SR)
        end = min(start + len(pluck), n_samples)
        out[start:end] += pluck[:end - start]

    stereo = mono_to_stereo(out)
    stereo = reverb_hall(stereo, sample_rate=SR)
    return _finalize(stereo, fade_sec=0.3)


def clip_02_paulstretch_15x() -> np.ndarray:
    """Short harmonic-rich source stretched 15x into an ethereal drone.
    Source: Karplus Cmaj triad, 2s. Stretched = 30s.
    """
    triad = np.zeros(int(2.0 * SR))
    for f in (130.81, 164.81, 196.0):
        triad += karplus_strong(f, 2.0, decay=0.998, brightness=0.5,
                                amplitude=0.5, seed=int(f), sample_rate=SR)
    triad = triad / np.max(np.abs(triad)) * 0.8
    stretched = paulstretch(triad, stretch_factor=15.0, window_sec=0.35,
                            seed=42, sample_rate=SR)
    stereo = mono_to_stereo(stretched)
    stereo = reverb_cathedral(stereo, sample_rate=SR)
    return _finalize(stereo, fade_sec=2.0)


def clip_03_warm_tape_pad() -> np.ndarray:
    """pad_alive + full analog warmth chain: tape_saturate + tape_hiss + vinyl_wow.
    Demonstrates saturation.py capabilities on a clean pad.
    """
    duration = 20.0
    pad = REGISTRY["pad_alive"](
        duration=duration, seed=42, sample_rate=SR,
        chord=[132.0, 196.0, 329.63, 528.0],  # Cmaj open
        intensity="gentle",
    )
    # Subtle tape saturation
    pad = tape_saturate(pad, drive=1.05, asymmetry=0.12)
    # Add slow pitch flutter
    pad = vinyl_wow(pad, depth=0.0004, rate_hz=0.2, seed=7, sample_rate=SR)
    # Add subliminal hiss
    hiss = tape_hiss(duration, level_db=-42.0, sample_rate=SR)
    min_len = min(pad.shape[0], hiss.shape[0])
    pad = pad[:min_len] + hiss[:min_len]
    return _finalize(pad, fade_sec=2.0)


def clip_04_ambient_scene() -> np.ndarray:
    """Complex scene: pad_alive + karplus chimes + micro_events + density_profile.
    Shows how the new primitives combine into a crafted ambient.
    """
    duration = 25.0
    root_seed = 42

    # Base pad
    pad = REGISTRY["pad_alive"](
        duration=duration, seed=derived_seed(root_seed, "pad"),
        sample_rate=SR,
        chord=[132.0, 196.0, 329.63], intensity="moderate",
    )
    # Sparse karplus "bell" plucks over the scene
    n = int(duration * SR)
    chimes = np.zeros(n)
    rng = np.random.default_rng(derived_seed(root_seed, "chimes"))
    chime_freqs = [523.25, 659.25, 784.0, 1046.5]  # C5, E5, G5, C6
    for _ in range(6):
        pos = rng.uniform(2.0, duration - 4.0)
        freq = float(rng.choice(chime_freqs))
        pluck = karplus_strong(freq, 4.0, decay=0.9985, brightness=0.7,
                               amplitude=0.4,
                               seed=int(pos * 1000), sample_rate=SR)
        start = int(pos * SR)
        end = min(start + len(pluck), n)
        chimes[start:end] += pluck[:end - start]
    chimes_stereo = mono_to_stereo(chimes)
    chimes_stereo = reverb_cathedral(chimes_stereo, sample_rate=SR)

    # Micro-events (harmonic blooms + overtone whispers)
    events = micro_events(
        duration,
        event_specs=[
            {"type": "harmonic_bloom", "rate_per_min": 4.0,
             "volume_db": -24.0, "duration_range": (2.0, 5.0)},
            {"type": "overtone_whisper", "rate_per_min": 2.0,
             "volume_db": -30.0, "duration_range": (2.0, 4.0)},
        ],
        chord_freqs=[132.0, 196.0, 329.63, 528.0],
        seed=derived_seed(root_seed, "events"),
        sample_rate=SR,
    )

    # Mix
    min_len = min(pad.shape[0], chimes_stereo.shape[0], events.shape[0])
    combined = mix(
        [pad[:min_len], chimes_stereo[:min_len], events[:min_len]],
        volumes_db=[-6.0, -10.0, 0.0],
    )

    # Apply density profile (random_walk evolution)
    profile = density_profile(duration, profile="random_walk",
                              seed=derived_seed(root_seed, "density"),
                              sample_rate=SR)
    combined = combined * profile[:min_len, np.newaxis]

    return _finalize(combined, fade_sec=2.0)


def clip_05_density_profile_ab() -> np.ndarray:
    """Same pad rendered with 3 density profiles stitched back-to-back.
    Demonstrates the audible difference between 'flat', 'arc', and 'random_walk'.
    """
    section = 8.0
    profiles = ["flat", "arc", "random_walk"]
    chunks = []

    for p_name in profiles:
        pad = REGISTRY["pad_alive"](
            duration=section, seed=42, sample_rate=SR,
            chord=[132.0, 196.0, 329.63], intensity="gentle",
        )
        profile = density_profile(section, profile=p_name, seed=42,
                                  sample_rate=SR)
        pad = pad * profile[:len(pad), np.newaxis]
        pad = fade_in(pad, 0.3, sample_rate=SR)
        pad = fade_out(pad, 0.3, sample_rate=SR)
        chunks.append(pad)

    combined = np.concatenate(chunks, axis=0)
    return _finalize(combined, fade_sec=0.5)


def clip_06_karplus_vs_sine() -> np.ndarray:
    """A/B: same chord as pure sines vs as Karplus plucks.
    First 8s sines, then 8s karplus.
    """
    freqs = [261.63, 329.63, 392.0]  # Cmaj
    section = 8.0

    # Sine version
    sine_mix = np.zeros(int(section * SR))
    for f in freqs:
        sine_mix += sine(f, section, amplitude=0.3, sample_rate=SR)
    sine_mix = mono_to_stereo(sine_mix)
    sine_mix = reverb_hall(sine_mix, sample_rate=SR)
    sine_mix = fade_in(sine_mix, 0.3, sample_rate=SR)
    sine_mix = fade_out(sine_mix, 0.3, sample_rate=SR)

    # Karplus version — stagger plucks for rhythmic feel
    karp_mix = np.zeros(int(section * SR))
    for i, f in enumerate(freqs):
        pluck_1 = karplus_strong(f, section, decay=0.999, brightness=0.55,
                                 amplitude=0.35, seed=i * 17,
                                 sample_rate=SR)
        # Stagger each note by 0.1s
        offset = int(i * 0.1 * SR)
        karp_mix[offset:offset + len(pluck_1)] += pluck_1[
            : len(karp_mix) - offset]
    karp_stereo = mono_to_stereo(karp_mix)
    karp_stereo = reverb_hall(karp_stereo, sample_rate=SR)
    karp_stereo = fade_in(karp_stereo, 0.3, sample_rate=SR)
    karp_stereo = fade_out(karp_stereo, 0.3, sample_rate=SR)

    combined = np.concatenate([sine_mix, karp_stereo], axis=0)
    return _finalize(combined, fade_sec=0.5)


def clip_07_piano_presets_ab() -> np.ndarray:
    """Same synthetic "piano" (karplus chord) through the 3 piano_presets.
    bass_drone (8s) + mid_pad (8s) + sparse_notes (8s).
    """
    from audiomancer.piano_presets import (
        preset_bass_drone, preset_mid_pad, preset_sparse_notes,
    )

    # Fake piano: quick Karplus triad, 4s
    triad = np.zeros(int(4.0 * SR))
    for f in (196.0, 246.94, 293.66):  # G3 B3 D4
        triad += karplus_strong(f, 4.0, decay=0.999, brightness=0.65,
                                amplitude=0.4, seed=int(f),
                                sample_rate=SR)
    triad = triad / np.max(np.abs(triad)) * 0.7

    bd = preset_bass_drone(triad, duration=8.0, sample_rate=SR)
    mp = preset_mid_pad(triad, duration=8.0, sample_rate=SR)
    sn = preset_sparse_notes(triad, duration=8.0, sample_rate=SR)

    # Normalize each so the A/B is fair
    for clip in (bd, mp, sn):
        peak = np.max(np.abs(clip))
        if peak > 0:
            clip *= 0.8 / peak

    combined = np.concatenate([bd, mp, sn], axis=0)
    return _finalize(combined, fade_sec=0.5)


def clip_08_texture_builder_trio() -> np.ndarray:
    """3 textures via the new 'texture' builder: crystal_shimmer, earth_hum, ethereal_wash.
    Each 7s, stitched back-to-back.
    """
    section = 7.0
    crystal = REGISTRY["texture"](
        duration=section, seed=42, sample_rate=SR,
        texture_name="crystal_shimmer", base_freq=528.0,
    )
    earth = REGISTRY["texture"](
        duration=section, seed=42, sample_rate=SR,
        texture_name="earth_hum", frequency=55.0,
    )
    ethereal = REGISTRY["texture"](
        duration=section, seed=42, sample_rate=SR,
        texture_name="ethereal_wash", frequency=392.0,
    )

    for clip in (crystal, earth, ethereal):
        peak = np.max(np.abs(clip))
        if peak > 0:
            clip *= 0.7 / peak

    combined = np.concatenate([crystal, earth, ethereal], axis=0)
    return _finalize(combined, fade_sec=0.5)


def clip_09_multi_lfo_movement() -> np.ndarray:
    """Same chord pad with vs without multi_lfo modulation.
    First 10s static, then 10s with multi_lfo.
    """
    from audiomancer.synth import chord_pad

    section = 10.0
    chord = [264.0, 330.0, 396.0]  # Cmaj close

    # Static
    static = chord_pad(chord, section, voices=4, detune_cents=6.0,
                       amplitude=0.5, sample_rate=SR)
    static_s = mono_to_stereo(static)
    static_s = reverb_hall(static_s, sample_rate=SR)
    static_s = fade_in(static_s, 0.3, sample_rate=SR)
    static_s = fade_out(static_s, 0.3, sample_rate=SR)

    # With multi_lfo
    live = chord_pad(chord, section, voices=4, detune_cents=6.0,
                     amplitude=0.5, sample_rate=SR)
    env = multi_lfo(section,
                    layers=[(1 / 3.0, 0.08), (1 / 7.0, 0.12), (1 / 20.0, 0.15)],
                    seed=42, sample_rate=SR)
    live_s = mono_to_stereo(live)
    live_s = reverb_hall(live_s, sample_rate=SR)
    live_s = apply_amplitude_mod(live_s, env)
    live_s = fade_in(live_s, 0.3, sample_rate=SR)
    live_s = fade_out(live_s, 0.3, sample_rate=SR)

    combined = np.concatenate([static_s, live_s], axis=0)
    return _finalize(combined, fade_sec=0.5)


def clip_10_full_ambient_3h_style() -> np.ndarray:
    """Taste of a V006-style production: pad + paulstretched chimes + warmth chain.
    20s loopable — what a "minute" of a real 3h YouTube track feels like.
    """
    duration = 20.0

    # Warm pad layer
    pad = REGISTRY["pad_alive"](
        duration=duration, seed=42, sample_rate=SR,
        chord=[132.0, 196.0, 329.63, 528.0], intensity="moderate",
    )

    # Paulstretched chimes from short karplus plucks
    chime_src = np.zeros(int(3.0 * SR))
    for f in (523.25, 659.25, 784.0):
        chime_src += karplus_strong(f, 3.0, decay=0.9985, brightness=0.7,
                                    amplitude=0.4, seed=int(f), sample_rate=SR)
    chime_stretched = paulstretch(chime_src, stretch_factor=duration / 3.0,
                                  window_sec=0.3, seed=42, sample_rate=SR)
    chimes = mono_to_stereo(chime_stretched)
    chimes = reverb_cathedral(chimes, sample_rate=SR)

    # Mix
    min_len = min(pad.shape[0], chimes.shape[0])
    combined = mix([pad[:min_len], chimes[:min_len]], volumes_db=[-6.0, -12.0])

    # Density profile + saturation chain
    profile = density_profile(duration, profile="random_walk", seed=42,
                              sample_rate=SR)
    combined = combined * profile[:min_len, np.newaxis]

    combined = tape_saturate(combined, drive=1.08, asymmetry=0.12)
    combined = vinyl_wow(combined, depth=0.0004, rate_hz=0.22, seed=3,
                        sample_rate=SR)
    hiss = tape_hiss(duration, level_db=-42.0, sample_rate=SR)
    hlen = min(combined.shape[0], hiss.shape[0])
    combined = combined[:hlen] + hiss[:hlen]

    # Loop-seal for that "could be part of a 3h" feel
    combined = make_loopable(combined, crossfade_sec=2.0, sample_rate=SR)
    return _finalize(combined, fade_sec=1.5)


CLIPS = {
    "01_karplus_cmaj_arpeggio": (clip_01_karplus_cmaj_arpeggio,
                                 "Karplus plucked Cmaj arpeggio (8 notes)"),
    "02_paulstretch_15x": (clip_02_paulstretch_15x,
                           "Karplus triad -> paulstretch 15x -> 30s ethereal"),
    "03_warm_tape_pad": (clip_03_warm_tape_pad,
                         "pad_alive + tape_saturate + vinyl_wow + tape_hiss"),
    "04_ambient_scene": (clip_04_ambient_scene,
                         "Complex: pad + karplus chimes + micro_events + density"),
    "05_density_profile_ab": (clip_05_density_profile_ab,
                              "flat / arc / random_walk (8s each, back-to-back)"),
    "06_karplus_vs_sine_ab": (clip_06_karplus_vs_sine,
                              "Cmaj sines (8s) then Cmaj karplus (8s)"),
    "07_piano_presets_ab": (clip_07_piano_presets_ab,
                            "bass_drone / mid_pad / sparse_notes (8s each)"),
    "08_texture_builder_trio": (clip_08_texture_builder_trio,
                                "crystal / earth / ethereal textures (7s each)"),
    "09_multi_lfo_movement": (clip_09_multi_lfo_movement,
                              "static pad (10s) then multi_lfo pad (10s)"),
    "10_full_ambient_3h_style": (clip_10_full_ambient_3h_style,
                                 "pad + paulstretched chimes + warmth chain (20s)"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Gallery v2 — post-V005 capabilities showcase."
    )
    parser.add_argument("--only", default=None,
                        help="Render only one clip by label")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    selected = {args.only: CLIPS[args.only]} if args.only else CLIPS

    print(f"=== Gallery v2 — {len(selected)} clip(s) ===")
    print(f"  Target: -14 LUFS, 48 kHz WAV, output/gallery_v2/")
    print()

    for label, (fn, desc) in selected.items():
        print(f"  [{label}] {desc}")
        sig = fn()
        path = OUT / f"{label}.wav"
        export_wav(sig, path, sample_rate=SR)
        peak_db = 20 * np.log10(np.max(np.abs(sig)) + 1e-10)
        print(f"    -> {path.name}  ({sig.shape[0] / SR:.1f}s, peak={peak_db:.1f} dBFS)")

    print()
    print(f"Done — {len(selected)} clip(s) in {OUT}/")


if __name__ == "__main__":
    main()
