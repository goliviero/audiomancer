#!/usr/bin/env python3
"""Gallery — showcase complet de toutes les capacités d'audiomancer.

15 sections couvrant les 15 modules :
  01  Waveforms         — sine, square, saw, triangle
  02  Noise Colors      — white / pink / brown
  03  Drones            — harmonics, filter sweep, Om
  04  Chord Pads        — voices, detune, supersaw
  05  Binaural Beats    — theta, alpha, delta, solfeggio
  06  Effects Chain     — reverb, delay, chorus, compress, highpass
  07  Modulation        — LFO, drift, evolving LFO, filter sweep
  08  Texture Bank      — 9 presets évolutifs
  09  Spectral          — freeze, blur, pitch shift, morph
  10  Spatial           — pan, auto-pan, width, Haas, rotate
  11  Harmony           — scales, just intonation, planetary, solfeggio
  12  Envelopes         — ADSR, AR, breathing, swell, gate
  13  Mastering         — highpass / mono-bass / soft-clip / limiter (before/after)
  14  Composition       — progressive arc, breathing pad, filter progression
  15  HERO              — full production mix toutes couches assemblées

Usage:
    python scripts/11_gallery.py
    => output/gallery/ (PNG + WAV, < 8 MB total)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiomancer import SAMPLE_RATE
from audiomancer.synth import (
    sine, square, sawtooth, triangle,
    white_noise, pink_noise, brown_noise,
    drone, pad, chord_pad,
)
from audiomancer.binaural import binaural, from_preset
from audiomancer.effects import (
    lowpass, highpass, reverb, reverb_cathedral, reverb_hall,
    delay, chorus, compress, chorus_subtle,
)
from audiomancer.modulation import (
    lfo_sine, lfo_triangle, drift, evolving_lfo,
    apply_amplitude_mod, apply_filter_sweep,
)
from audiomancer.textures import REGISTRY, generate as gen_texture
from audiomancer.compose import fade_envelope, tremolo, make_loopable, verify_loop
from audiomancer.layers import mix, layer, crossfade, normalize_lufs
from audiomancer.utils import (
    normalize, fade_in, fade_out, mono_to_stereo, stereo_to_mono,
    export_wav, silence, concat,
)
from audiomancer.spectral import freeze, blur, pitch_shift, spectral_gate, morph
from audiomancer.spatial import (
    pan, auto_pan, stereo_width, encode_mid_side, decode_mid_side,
    haas_width, rotate,
)
from audiomancer.harmony import (
    note_to_hz, scale, just_chord, harmonic_series, subharmonic_series,
    drone_cluster, fibonacci_freqs, SOLFEGGIO, PLANETARY, SCALES,
    just_intonation, pythagorean,
)
from audiomancer.envelope import (
    adsr, adsr_exp, ar, segments, breathing, swell, gate_pattern,
)
from audiomancer.mastering import mono_bass, soft_clip, limit, master_chain
import audiomancer.quick as q


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR = SAMPLE_RATE
OUT = Path("output/gallery")
OUT.mkdir(parents=True, exist_ok=True)

WAV_DUR = 4.0       # clip duration in seconds
WAV_CLIP = 3.0      # max clip length saved to disk

CMAP = LinearSegmentedColormap.from_list(
    "audiomancer",
    ["#060612", "#1a0a3a", "#3a1a6a", "#6a2a9a", "#9a4aba", "#da7afa", "#ffc0ff"],
)

plt.rcParams.update({
    "figure.facecolor": "#080818",
    "axes.facecolor": "#0c0c20",
    "axes.edgecolor": "#2a2a4a",
    "text.color": "#e0e0f0",
    "axes.labelcolor": "#aaa",
    "xtick.color": "#666",
    "ytick.color": "#666",
    "grid.color": "#1a1a3a",
    "grid.linestyle": ":",
    "font.family": "monospace",
    "font.size": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "axes.grid": True,
})

generated = []


def _harm(n: int) -> list[tuple[float, float]]:
    return [(i, 1.0 / i) for i in range(1, n + 1)]


def save_fig(name: str, fig=None):
    path = OUT / f"{name}.png"
    (fig or plt.gcf()).savefig(str(path), dpi=150)
    plt.close(fig or plt.gcf())
    generated.append(path)
    print(f"  [PNG] {path.name}")


def save_wav(signal: np.ndarray, name: str, sr: int = SR, dur: float = WAV_CLIP):
    sig = signal[:int(sr * dur)]
    if sig.ndim == 2:
        sig = np.mean(sig, axis=1)
    sig = normalize(sig, target_db=-1.0)
    path = OUT / f"{name}.wav"
    export_wav(sig, path, sample_rate=sr, bit_depth=16)
    generated.append(path)
    print(f"  [WAV] {path.name}")


def save_wav_stereo(signal: np.ndarray, name: str, sr: int = SR, dur: float = WAV_CLIP):
    sig = signal[:int(sr * dur)]
    if sig.ndim == 1:
        sig = mono_to_stereo(sig)
    path = OUT / f"{name}.wav"
    sig_norm = sig / (np.max(np.abs(sig)) + 1e-10) * 0.9
    export_wav(sig_norm, path, sample_rate=sr, bit_depth=16)
    generated.append(path)
    print(f"  [WAV] {path.name}")


def specgram(signal, ax, title="", sr=SR, fmax=6000):
    mono = stereo_to_mono(signal) if signal.ndim == 2 else signal
    f, t, Sxx = spectrogram(mono, fs=sr, nperseg=2048, noverlap=1792)
    ax.pcolormesh(t, f / 1000, 10 * np.log10(Sxx + 1e-10),
                  cmap=CMAP, shading="gouraud", vmin=-90, vmax=0)
    ax.set_title(title, fontsize=8, pad=3, color="#e0e0f0")
    ax.set_ylim(0, fmax / 1000)
    ax.set_ylabel("kHz", fontsize=7)


def waveplot(signal, ax, title="", color="#9a4aba", zoom_sec=None):
    t = np.linspace(0, len(signal) / SR, len(signal))
    if zoom_sec:
        n = int(SR * zoom_sec)
        t, signal = t[:n], signal[:n]
    if signal.ndim == 2:
        ax.plot(t, signal[:, 0], color=color, lw=0.3, alpha=0.8, label="L")
        ax.plot(t, signal[:, 1], color="#4a9aba", lw=0.3, alpha=0.8, label="R")
    else:
        ax.plot(t, signal, color=color, lw=0.4, alpha=0.85)
    ax.set_title(title, fontsize=8, pad=3, color="#e0e0f0")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.15, 1.15)


# ===========================================================================
# SECTIONS
# ===========================================================================

def section_waveforms():
    print("[01] Waveforms")
    dur = 0.015
    waves = [
        ("Sine 440 Hz",     sine(440, dur),     "#9a4aba"),
        ("Square 440 Hz",   square(440, dur),   "#da7afa"),
        ("Sawtooth 440 Hz", sawtooth(440, dur), "#4a9aba"),
        ("Triangle 440 Hz", triangle(440, dur), "#7adaba"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))
    for ax, (name, sig, c) in zip(axes, waves):
        waveplot(sig, ax, name, color=c)
    fig.suptitle("01 — BASIC WAVEFORMS", fontsize=11, color="#fff", y=1.04)
    plt.tight_layout()
    save_fig("01_waveforms", fig)


def section_noise():
    print("[02] Noise Colors")
    dur = 3.0
    noises = [
        ("White Noise — flat spectrum",   white_noise(dur), "#da7afa"),
        ("Pink Noise — -3 dB/oct (1/f)",  pink_noise(dur),  "#ff9a6a"),
        ("Brown Noise — -6 dB/oct (1/f²)",brown_noise(dur), "#9a6a4a"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 7))
    for i, (name, sig, c) in enumerate(noises):
        waveplot(sig[:SR // 2], axes[i, 0], f"{name.split(' — ')[0]} — waveform", color=c, zoom_sec=0.1)
        specgram(sig, axes[i, 1], f"{name.split(' — ')[0]} — spectrum")
    fig.suptitle("02 — NOISE COLORS", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("02_noise_colors", fig)


def section_drones():
    print("[03] Drones & Harmonics")
    dur = WAV_DUR
    d_warm = q.drone(136.1, dur, harmonics=q.HARMONICS_WARM, cutoff_hz=3000, seed=1)
    d_dark = q.drone(111.0, dur, harmonics=q.HARMONICS_DARK, cutoff_hz=2000, seed=2)
    d_bright = q.drone(256.0, dur, harmonics=q.HARMONICS_BRIGHT, cutoff_hz=5000, seed=3)

    # Filter sweep on Om drone
    sweep = fade_envelope([(0, 400), (1, 3000), (3, 1000), (4, 400)], dur)
    d_swept = apply_filter_sweep(q.drone(136.1, dur, harmonics=_harm(10), seed=4), sweep)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    specgram(d_warm, axes[0, 0], "Om 136 Hz — WARM (H-series 1/n)")
    specgram(d_dark, axes[0, 1], "Holy 111 Hz — DARK (low rolloff)")
    specgram(d_bright, axes[1, 0], "C 256 Hz — BRIGHT (upper harmonics)")
    specgram(d_swept, axes[1, 1], "Om 136 Hz — filter sweep 400→3kHz→400")
    fig.suptitle("03 — DRONES: HARMONICS & FILTER SWEEP", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("03_drones", fig)

    # WAV: Om drone with cathedral reverb + filter sweep
    d = d_swept
    d = mono_to_stereo(d)
    d = reverb_cathedral(d)
    d = fade_in(d, 0.3)
    d = fade_out(d, 0.5)
    save_wav_stereo(d, "03_om_filter_sweep")


def section_pads():
    print("[04] Chord Pads")
    dur = WAV_DUR
    # A minor just intonation (Om family)
    om_ji = [136.1, 136.1 * 6/5, 136.1 * 3/2]
    # C major just intonation
    c_ji = [261.63, 261.63 * 5/4, 261.63 * 3/2]
    # Fibonacci-based chord from C3
    fib = fibonacci_freqs(note_to_hz("C3"), n=5)[:3]

    p1 = chord_pad(om_ji, dur, voices=4, detune_cents=10)
    p2 = chord_pad(c_ji, dur, voices=6, detune_cents=18)
    p3 = chord_pad(fib, dur, voices=5, detune_cents=12)
    # Supersaw
    freqs_saw = [note_to_hz("D3"), note_to_hz("F3"), note_to_hz("A3")]
    p4 = chord_pad(freqs_saw, dur, voices=8, detune_cents=30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    specgram(p1, axes[0, 0], "A minor JI — 4 voices ±10¢")
    specgram(p2, axes[0, 1], "C major JI — 6 voices ±18¢")
    specgram(p3, axes[1, 0], "Fibonacci chord (C3) — 5 voices")
    specgram(p4, axes[1, 1], "Supersaw D-F-A — 8 voices ±30¢")
    fig.suptitle("04 — CHORD PADS: Just Intonation, Fibonacci, Supersaw", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("04_chord_pads", fig)

    # WAV: lush Om pad with breathing + reverb
    breath = breathing(dur, breath_rate=0.1, depth=0.25, floor=0.75)
    p = apply_amplitude_mod(p1, breath)
    p = mono_to_stereo(p)
    p = auto_pan(p, rate_hz=0.05, depth=0.4)
    p = reverb_cathedral(p)
    p = fade_in(p, 0.5)
    p = fade_out(p, 0.5)
    save_wav_stereo(p, "04_om_pad_breathing")


def section_binaural():
    print("[05] Binaural Beats")
    dur = 3.0
    presets = [
        ("om_theta",        "Om Theta (6 Hz) — deep meditation"),
        ("theta_deep",      "Theta Deep (4 Hz) — hypnagogic"),
        ("alpha_relax",     "Alpha (10 Hz) — relaxed focus"),
        ("delta_sleep",     "Delta (2 Hz) — deep sleep"),
        ("solfeggio_528",   "Solfeggio 528 Hz (love frequency)"),
        ("solfeggio_432",   "Sacred A=432 Hz"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 5))
    colors = ["#9a4aba", "#da7afa", "#4a9aba", "#2a6a4a", "#f0a060", "#60d0a0"]
    for ax, (preset, label), c in zip(axes.flat, presets, colors):
        sig = from_preset(preset, dur)
        t = np.linspace(0, dur, len(sig))
        ax.plot(t, sig[:, 0], color=c, lw=0.4, alpha=0.8, label="L")
        ax.plot(t, sig[:, 1], color="#fff", lw=0.4, alpha=0.4, label="R")
        ax.set_title(label, fontsize=7.5, pad=3, color="#e0e0f0")
        ax.set_xlim(0, 0.4)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("05 — BINAURAL BEATS — 6 presets (zoomed 0.4s)", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("05_binaural_beats", fig)


def section_effects():
    print("[06] Effects Chain")
    dur = 3.0
    src = drone(220, dur, harmonics=_harm(6))
    src_s = mono_to_stereo(src)

    sigs = [
        ("Dry Signal",         src_s),
        ("Highpass 300 Hz",    mono_to_stereo(highpass(src, 300))),
        ("Cathedral Reverb",   reverb_cathedral(src_s)),
        ("Delay 400ms / 50%",  delay(src_s, delay_seconds=0.4, feedback=0.5, mix=0.35)),
        ("Chorus subtle",      chorus_subtle(src_s)),
        ("Compress -12dB 4:1", compress(src_s, threshold_db=-12, ratio=4.0)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, (name, sig) in zip(axes.flat, sigs):
        specgram(sig, ax, name)
    fig.suptitle("06 — EFFECTS CHAIN — 220 Hz Drone", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("06_effects_chain", fig)

    # WAV: delay + reverb combo
    wet = delay(src_s, delay_seconds=0.35, feedback=0.45, mix=0.3)
    wet = reverb_hall(wet)
    save_wav_stereo(wet, "06_delay_reverb")


def section_modulation():
    print("[07] Modulation")
    dur = 12.0
    t = np.linspace(0, dur, int(SR * dur))

    lfo_s = lfo_sine(dur, rate_hz=0.12, depth=0.4, offset=1.0)
    lfo_t = lfo_triangle(dur, rate_hz=0.08, depth=0.35, offset=1.0)
    dr = drift(dur, speed=0.15, depth=0.4, offset=1.0, seed=42)
    ev = evolving_lfo(dur, rate_hz=0.1, depth=0.35, offset=1.0, drift_speed=0.04, seed=7)

    # Tremolo on drone: compare static vs evolving
    d = drone(136.1, 5.0, harmonics=_harm(8))
    d_tremolo = tremolo(d, rate_hz=0.12, depth=0.08, seed=3)
    sweep = evolving_lfo(5.0, rate_hz=0.08, depth=2000, offset=2500, seed=9)
    d_swept = apply_filter_sweep(drone(136.1, 5.0, harmonics=_harm(8)), sweep)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # Top row: 4 modulation sources
    ax_lfo_s = fig.add_subplot(gs[0, 0])
    ax_lfo_t = fig.add_subplot(gs[0, 1])
    mods = [(lfo_s, "#9a4aba", "LFO Sine 0.12 Hz"),
            (lfo_t, "#4a9aba", "LFO Triangle 0.08 Hz"),
            (dr, "#da7afa", "Brownian Drift (speed=0.15)"),
            (ev, "#7adaba", "Evolving LFO (rate drifts)")]
    tts = np.linspace(0, dur, len(lfo_s))
    ax_lfo_s.plot(tts, lfo_s, color="#9a4aba", lw=0.8)
    ax_lfo_s.plot(tts, lfo_t, color="#4a9aba", lw=0.8)
    ax_lfo_s.set_title("LFO: Sine vs Triangle", fontsize=8, pad=3, color="#e0e0f0")
    ax_lfo_s.set_xlim(0, dur)
    ax_lfo_t.plot(tts, dr, color="#da7afa", lw=0.8)
    ax_lfo_t.plot(tts, ev, color="#7adaba", lw=0.8)
    ax_lfo_t.set_title("Drift vs Evolving LFO", fontsize=8, pad=3, color="#e0e0f0")
    ax_lfo_t.set_xlim(0, dur)

    # Middle row: applied modulation (tremolo)
    ax_trem = fig.add_subplot(gs[1, 0])
    ax_swp = fig.add_subplot(gs[1, 1])
    t5 = np.linspace(0, 5, len(d))
    ax_trem.plot(t5, d, color="#444", lw=0.3, alpha=0.6, label="dry")
    ax_trem.plot(t5, d_tremolo, color="#9a4aba", lw=0.3, alpha=0.8, label="tremolo")
    ax_trem.set_title("Tremolo 0.12 Hz depth=0.08", fontsize=8, pad=3, color="#e0e0f0")
    ax_trem.legend(fontsize=6)
    ax_trem.set_xlim(0, 5)
    specgram(mono_to_stereo(d_swept), ax_swp, "Evolving Filter Sweep on Drone")

    # Bottom row: filter curve overlaid
    ax_curve = fig.add_subplot(gs[2, :])
    t12 = np.linspace(0, dur, len(ev))
    ax_curve.fill_between(t12, 1.0, ev / ev.max(), color="#9a4aba", alpha=0.15)
    ax_curve.plot(t12, lfo_s / lfo_s.max(), color="#9a4aba", lw=0.7, label="LFO Sine (norm)")
    ax_curve.plot(t12, dr / dr.max(), color="#da7afa", lw=0.7, label="Drift (norm)")
    ax_curve.plot(t12, ev / ev.max(), color="#7adaba", lw=0.7, label="Evolving (norm)")
    ax_curve.set_title("Modulation Sources Comparison (normalized)", fontsize=8, pad=3, color="#e0e0f0")
    ax_curve.set_xlim(0, dur)
    ax_curve.legend(fontsize=6, loc="upper right")
    fig.suptitle("07 — MODULATION: LFO / Drift / Evolving LFO / Tremolo", fontsize=11, color="#fff")
    save_fig("07_modulation", fig)

    # WAV: filter-swept drone
    d2 = drone(136.1, WAV_DUR, harmonics=_harm(8))
    sweep2 = evolving_lfo(WAV_DUR, rate_hz=0.1, depth=2500, offset=2500, seed=12)
    d2 = apply_filter_sweep(d2, sweep2)
    d2 = mono_to_stereo(d2)
    d2 = reverb_hall(d2)
    d2 = fade_in(d2, 0.4)
    d2 = fade_out(d2, 0.5)
    save_wav_stereo(d2, "07_evolving_filter_sweep")


def section_textures():
    print("[08] Texture Bank (9 presets)")
    dur = 5.0
    names = list(REGISTRY.keys())

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    for ax, name in zip(axes.flat, names):
        sig = gen_texture(name, dur, seed=42)
        specgram(sig, ax, name.replace("_", " ").title())
    fig.suptitle("08 — TEXTURE BANK — 9 Evolving Presets", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("08_texture_bank", fig)

    # WAV: ethereal wash + crystal shimmer layered
    tex_a = gen_texture("ethereal_wash", WAV_DUR, seed=1)
    tex_b = gen_texture("crystal_shimmer", WAV_DUR, seed=2)
    tex_c = gen_texture("ocean_bed", WAV_DUR, seed=3)
    layered = mix([tex_a, tex_b, tex_c], volumes_db=[0, -4, -6])
    layered = normalize_lufs(layered, target_lufs=-14)
    layered = fade_in(layered, 0.5)
    layered = fade_out(layered, 0.5)
    save_wav_stereo(layered, "08_texture_layers")


def section_spectral():
    print("[09] Spectral Processing")
    dur = WAV_DUR
    src = drone(220, dur, harmonics=_harm(8))

    frozen = freeze(src, freeze_time=1.5, duration_sec=dur)
    blurred = blur(src, amount=0.7)
    shifted_up = pitch_shift(src, semitones=12)   # octave up
    shifted_dn = pitch_shift(src, semitones=-7)   # 5th down
    noise = pink_noise(dur)
    morphed_30 = morph(src, noise, mix=0.3)
    morphed_70 = morph(src, noise, mix=0.7)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    sigs = [("Original 220 Hz drone",   src),
            ("Spectral Freeze @1.5s",    frozen),
            ("Spectral Blur 0.7",        blurred),
            ("Pitch Shift +12 (octave)", shifted_up),
            ("Pitch Shift -7 (5th dn)",  shifted_dn),
            ("Morph Drone→Pink 30%/70%", mix([morphed_30, morphed_70], volumes_db=[0, -3]))]
    for ax, (name, sig) in zip(axes.flat, sigs):
        specgram(sig, ax, name)
    fig.suptitle("09 — SPECTRAL PROCESSING: Freeze / Blur / Pitch / Morph", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("09_spectral", fig)

    # WAV: frozen + reverb = infinite pad
    inf_pad = freeze(src, freeze_time=2.0, duration_sec=WAV_DUR)
    inf_pad = mono_to_stereo(inf_pad)
    inf_pad = reverb_cathedral(inf_pad)
    inf_pad = fade_in(inf_pad, 0.8)
    inf_pad = fade_out(inf_pad, 0.6)
    save_wav_stereo(inf_pad, "09_frozen_infinite_pad")


def section_spatial():
    print("[10] Spatial Audio")
    dur = 3.0
    src = drone(220, dur, harmonics=_harm(6))

    # Width progression: mono → normal → ultra wide
    stereo_src = np.column_stack([drone(220, dur, harmonics=_harm(6)),
                                   drone(220, dur, harmonics=_harm(6))])
    narrow = stereo_width(stereo_src, width=0.2)
    normal = stereo_width(stereo_src, width=1.0)
    wide = stereo_width(stereo_src, width=2.0)

    ap_slow = auto_pan(src, rate_hz=0.05, depth=0.8)  # 20s cycle
    ap_fast = auto_pan(src, rate_hz=0.3, depth=1.0)   # 3s cycle
    haas = haas_width(src, delay_ms=20)
    rot = rotate(src, revolutions=1.0)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.3)

    # Stereo width comparison (L-R difference)
    for i, (name, sig) in enumerate([("Narrow 0.2x", narrow), ("Normal 1.0x", normal), ("Wide 2.0x", wide)]):
        ax = fig.add_subplot(gs[0, i])
        t = np.linspace(0, dur, len(sig))
        ax.plot(t, sig[:, 0], color="#9a4aba", lw=0.4, alpha=0.8, label="L")
        ax.plot(t, sig[:, 1], color="#4a9aba", lw=0.4, alpha=0.5, label="R")
        ax.set_title(f"Stereo Width {name}", fontsize=7.5, pad=3, color="#e0e0f0")
        ax.set_xlim(0, 0.02)
        ax.legend(fontsize=6)

    # Auto-pan waveforms
    for i, (name, sig) in enumerate([("Auto-Pan slow 0.05Hz", ap_slow),
                                       ("Auto-Pan fast 0.3Hz", ap_fast),
                                       ("Haas Width 20ms", haas)]):
        ax = fig.add_subplot(gs[1, i])
        t = np.linspace(0, dur, len(sig))
        ax.plot(t, sig[:, 0], color="#9a4aba", lw=0.3, alpha=0.8)
        ax.plot(t, sig[:, 1], color="#4a9aba", lw=0.3, alpha=0.5)
        ax.set_title(name, fontsize=7.5, pad=3, color="#e0e0f0")
        ax.set_xlim(0, dur if "slow" in name else 1.0)

    # Spectrogram of rotation
    ax_rot = fig.add_subplot(gs[2, :])
    specgram(rot, ax_rot, "Rotation — 1 revolution over 3s")

    fig.suptitle("10 — SPATIAL AUDIO: Width / Auto-Pan / Haas / Rotation", fontsize=11, color="#fff")
    save_fig("10_spatial", fig)

    # WAV: rotating drone
    d = q.drone(136.1, WAV_DUR, harmonics=_harm(8), seed=5)
    d = rotate(d, revolutions=0.5)
    d = reverb_hall(d)
    d = fade_in(d, 0.3)
    d = fade_out(d, 0.5)
    save_wav_stereo(d, "10_rotating_drone")


def section_harmony():
    print("[11] Harmony, Sacred Freqs, Tuning")
    dur = 2.0

    # --- Scales arpeggiato ---
    scale_names = ["major", "minor", "dorian", "pelog",
                   "raga_bhairav", "hirajoshi", "prometheus", "blues"]
    root = note_to_hz("C3")
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for ax, sname in zip(axes.flat, scale_names):
        freqs = scale(root, sname, octaves=2)
        ndur = 0.25
        arp = np.concatenate([
            fade_in(fade_out(sine(f, ndur) * 0.4, 0.02), 0.01) for f in freqs
        ])
        specgram(arp, ax, sname.replace("_", " ").title(), fmax=2500)
    fig.suptitle("11a — SCALES (C3 root, 2 octaves, arpeggiated)", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("11a_scales", fig)

    # --- Sacred / Planetary frequencies ---
    sacred_items = [
        ("Earth Year (Om) 136.1 Hz",  PLANETARY["earth_year"]),
        ("Earth Day 194.18 Hz",        PLANETARY["earth_day"]),
        ("Moon 210.42 Hz",             PLANETARY["moon"]),
        ("Sun 126.22 Hz",              PLANETARY["sun"]),
        ("Solfeggio 528 Hz (Mi)",      SOLFEGGIO["mi"]),
        ("Solfeggio 432 Hz (ref)",     432.0),
        ("Solfeggio 396 Hz (Ut)",      SOLFEGGIO["ut"]),
        ("Solfeggio 639 Hz (Fa)",      SOLFEGGIO["fa"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for ax, (label, freq) in zip(axes.flat, sacred_items):
        d = drone(freq, dur, harmonics=_harm(6))
        d = mono_to_stereo(d)
        specgram(d, ax, f"{label}", fmax=min(freq * 8 + 200, 8000))
    fig.suptitle("11b — SACRED & PLANETARY FREQUENCIES", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("11b_sacred_frequencies", fig)

    # --- Tuning comparison ---
    root_hz = note_to_hz("C3")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (label, freqs) in zip(axes, [
        ("Equal Temp C-E-G",      [note_to_hz("C3"), note_to_hz("E3"), note_to_hz("G3")]),
        ("Just Intonation C-E-G", just_chord(root_hz, "major")),
        ("Pythagorean C-E-G",     [root_hz, pythagorean(root_hz, 4), pythagorean(root_hz, 7)]),
    ]):
        p = chord_pad(freqs, dur, voices=1)
        specgram(p, ax, label, fmax=2000)
    fig.suptitle("11c — TUNING SYSTEMS COMPARISON", fontsize=11, color="#fff", y=1.05)
    plt.tight_layout()
    save_fig("11c_tuning_systems", fig)

    # WAV: sacred planetary chord
    pl_chord = [PLANETARY["earth_year"], PLANETARY["moon"], PLANETARY["earth_day"]]
    p = chord_pad(pl_chord, WAV_DUR, voices=5, detune_cents=8)
    p = mono_to_stereo(p)
    p = reverb_cathedral(p)
    p = fade_in(p, 0.6)
    p = fade_out(p, 0.5)
    save_wav_stereo(p, "11_planetary_chord")


def section_envelopes():
    print("[12] Envelopes")
    dur = 3.0
    envs = [
        ("ADSR linear",         adsr(dur, 0.2, 0.3, 0.6, 0.8)),
        ("ADSR exponential",    adsr_exp(dur, 0.2, 0.3, 0.6, 0.8, curve=4.0)),
        ("AR (attack-release)", ar(dur, attack=0.4, curve=2.5)),
        ("Swell peak@60%",      swell(dur, peak_time=0.6, hold=0.1, curve=2.5)),
        ("Breathing 0.15 Hz",   breathing(dur, 0.15, depth=0.35, floor=0.65)),
        ("Gate [1,0,0.5,1,0]",  gate_pattern(dur, [1, 0, 0.5, 1, 0, 0.8], step_sec=0.25)),
    ]
    colors = ["#9a4aba", "#da7afa", "#4a9aba", "#7adaba", "#ff9a9a", "#ffc0ff"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax, (name, env), c in zip(axes.flat, envs, colors):
        t = np.linspace(0, dur, len(env))
        ax.fill_between(t, 0, env, color=c, alpha=0.25)
        ax.plot(t, env, color=c, lw=1.5)
        ax.set_title(name, fontsize=8, pad=3, color="#e0e0f0")
        ax.set_xlim(0, dur)
        ax.set_ylim(-0.05, 1.1)
    fig.suptitle("12 — ENVELOPES: ADSR / AR / Breathing / Swell / Gate", fontsize=11, color="#fff", y=1.01)
    plt.tight_layout()
    save_fig("12_envelopes", fig)

    # WAV: gated pad with reverb
    freqs = just_chord(note_to_hz("A2"), "minor")
    p = chord_pad(freqs, WAV_DUR, voices=5, detune_cents=15)
    gate = gate_pattern(WAV_DUR, [1, 0.5, 0, 1, 1, 0, 0.8], step_sec=0.3, smoothing_ms=15)
    p = apply_amplitude_mod(p, gate)
    p = mono_to_stereo(p)
    p = reverb_hall(p)
    save_wav_stereo(p, "12_gated_pad")


def section_mastering():
    print("[13] Mastering Chain")
    dur = WAV_DUR

    # Source: a mix similar to a real stem
    d = q.drone(136.1, dur, harmonics=_harm(8), seed=1)
    p = q.pad([136.1, 163.3, 204.2], dur, voices=4, detune_cents=10, dark=True)
    noise = gen_texture("noise_wash", dur, seed=3)
    raw = mix([mono_to_stereo(d), p, noise], volumes_db=[0, -4, -10])
    raw = normalize_lufs(raw, target_lufs=-14.0)

    # Process step by step
    after_hp = highpass(raw, cutoff_hz=30)
    after_mono_bass = mono_bass(after_hp, crossover_hz=100)
    after_clip = soft_clip(after_mono_bass, threshold_db=-3.0)
    after_limit = limit(after_clip, ceiling_dbtp=-1.0)

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)

    # Waveforms before/after
    ax_bef = fig.add_subplot(gs[0, 0])
    ax_aft = fig.add_subplot(gs[0, 1])
    waveplot(raw, ax_bef, "Before mastering (-14 LUFS raw)", zoom_sec=0.1)
    waveplot(after_limit, ax_aft, "After mastering (-1 dBTP limited)", color="#7adaba", zoom_sec=0.1)

    # Spectrograms before/after
    ax_spec_b = fig.add_subplot(gs[1, 0])
    ax_spec_a = fig.add_subplot(gs[1, 1])
    specgram(raw, ax_spec_b, "Spectrum — before", fmax=8000)
    specgram(after_limit, ax_spec_a, "Spectrum — after", fmax=8000)

    # Step-by-step peak levels
    ax_levels = fig.add_subplot(gs[2, :])
    steps = ["raw", "highpass 30Hz", "mono bass 100Hz", "soft clip -3dB", "limit -1dBTP"]
    sigs = [raw, after_hp, after_mono_bass, after_clip, after_limit]
    peaks = [20 * np.log10(np.max(np.abs(s)) + 1e-10) for s in sigs]
    rms = [20 * np.log10(np.sqrt(np.mean(s**2)) + 1e-10) for s in sigs]
    x = range(len(steps))
    ax_levels.bar(x, peaks, color="#9a4aba", alpha=0.6, label="Peak dBFS", width=0.35)
    ax_levels.bar([xi + 0.38 for xi in x], rms, color="#4a9aba", alpha=0.6, label="RMS dBFS", width=0.35)
    ax_levels.axhline(-1.0, color="#ff4a4a", lw=1, linestyle="--", label="-1 dBTP ceiling")
    ax_levels.set_xticks([xi + 0.19 for xi in x])
    ax_levels.set_xticklabels(steps, fontsize=7)
    ax_levels.set_ylabel("dBFS", fontsize=7)
    ax_levels.set_ylim(-30, 3)
    ax_levels.legend(fontsize=7)
    ax_levels.set_title("Peak & RMS per mastering step", fontsize=8, pad=3, color="#e0e0f0")

    fig.suptitle("13 — MASTERING CHAIN: Highpass / Mono Bass / Soft Clip / Limiter", fontsize=11, color="#fff")
    save_fig("13_mastering_chain", fig)

    # WAV: before/after
    save_wav_stereo(raw, "13a_before_mastering")
    save_wav_stereo(after_limit, "13b_after_mastering")


def section_composition():
    print("[14] Progressive Composition (arc)")
    # Miniature progressive arc — 30s to keep gallery generation fast
    dur = 30.0

    d = q.drone(136.1, dur, harmonics=q.HARMONICS_WARM, cutoff_hz=3000, seed=10)
    p = q.pad([136.1, 163.3, 204.2], dur, voices=4, detune_cents=10, dark=True)
    tex = q.texture("noise_wash", dur, seed=11)
    bb = q.binaural("om_theta", dur, volume_db=-14.0)

    # Volume arcs (Awakening → Deepening → Fullness → Return) — scaled to 30s
    d = apply_amplitude_mod(d, fade_envelope(
        [(0, 0.15), (5, 1.0), (24, 1.0), (28, 0.15), (30, 0.15)], dur))
    p = apply_amplitude_mod(p, fade_envelope(
        [(0, 0.0), (7, 0.0), (10, 0.85), (24, 0.85), (28, 0.0), (30, 0.0)], dur))
    tex = apply_amplitude_mod(tex, fade_envelope(
        [(0, 0.0), (16, 0.0), (19, 0.65), (24, 0.65), (27, 0.0), (30, 0.0)], dur))

    # Breathing on pad
    p = apply_amplitude_mod(p, breathing(dur, breath_rate=0.09, depth=0.2, floor=0.8))

    # Tremolo on drone
    d = tremolo(d, rate_hz=0.12, depth=0.06, seed=12)

    # Filter sweep
    filt = fade_envelope([(0, 600), (8, 2200), (15, 3000), (22, 2500), (27, 900), (30, 600)], dur)
    d = apply_filter_sweep(d, filt)

    # Spatial: auto-pan on texture
    from audiomancer.spatial import auto_pan
    tex = auto_pan(tex, rate_hz=0.04, depth=0.5)

    # Mix + master
    stem = mix([mono_to_stereo(d), p, tex, bb], volumes_db=[0, -4, -8, 0])
    stem = normalize_lufs(stem, target_lufs=-14)
    stem = master_chain(stem)
    stem = make_loopable(stem, crossfade_sec=3.0)

    # Loop quality
    score, report = verify_loop(stem, crossfade_sec=3.0)

    # Visualization
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    # Volume envelopes overlay
    ax_env = fig.add_subplot(gs[0])
    t = np.linspace(0, dur, int(SR * dur))
    d_env = fade_envelope([(0, 0.15), (5, 1.0), (24, 1.0), (28, 0.15), (30, 0.15)], dur)
    p_env = fade_envelope([(0, 0.0), (7, 0.0), (10, 0.85), (24, 0.85), (28, 0.0), (30, 0.0)], dur)
    tex_env = fade_envelope([(0, 0.0), (16, 0.0), (19, 0.65), (24, 0.65), (27, 0.0), (30, 0.0)], dur)
    ax_env.fill_between(t, 0, d_env, alpha=0.3, color="#9a4aba", label="Drone")
    ax_env.fill_between(t, 0, p_env, alpha=0.3, color="#4a9aba", label="Pad")
    ax_env.fill_between(t, 0, tex_env, alpha=0.3, color="#7adaba", label="Texture")
    ax_env.plot(t, d_env, color="#9a4aba", lw=1.0)
    ax_env.plot(t, p_env, color="#4a9aba", lw=1.0)
    ax_env.plot(t, tex_env, color="#7adaba", lw=1.0)
    ax_env.set_title("Volume Envelopes — Awakening → Deepening → Fullness → Return", fontsize=8, pad=3, color="#e0e0f0")
    ax_env.legend(fontsize=7, loc="upper right")
    ax_env.set_xlim(0, dur)
    ax_env.axvline(5, color="#333", lw=0.5, linestyle=":")
    ax_env.axvline(16, color="#333", lw=0.5, linestyle=":")
    ax_env.axvline(24, color="#333", lw=0.5, linestyle=":")
    for label, x in [("AWAKEN", 2), ("DEEPENING", 10), ("FULLNESS", 20), ("RETURN", 27)]:
        ax_env.text(x, 0.92, label, fontsize=6, color="#888", ha="center")

    # Spectrogram full mix
    ax_spec = fig.add_subplot(gs[1])
    specgram(stem, ax_spec, "Full Mix Spectrogram (mastered)")

    # Waveform
    ax_wave = fig.add_subplot(gs[2])
    waveplot(stem, ax_wave, f"Waveform — Loop score: {score:.3f} (corr={report['correlation']:.3f})")

    fig.suptitle(
        "14 — COMPOSITION: 60s Progressive Arc — Om 136 Hz\n"
        "Drone + Pad + Texture + Binaural | Breathing | Filter Sweep | Mastering | Loop Verified",
        fontsize=10, color="#fff"
    )
    save_fig("14_composition_arc", fig)

    # WAV: first 30s of the composition
    save_wav_stereo(stem, "14_progressive_composition", dur=WAV_CLIP)


def section_hero():
    print("[HERO] Full Production Mix")
    dur = WAV_DUR * 2  # 8s

    # Layer 1: Om drone — filter drifts, tremolo
    d = q.drone(136.1, dur, harmonics=q.HARMONICS_WARM, cutoff_hz=3500, seed=20)
    filt = drift(dur, speed=0.08, depth=1800, offset=2200, seed=20)
    d = apply_filter_sweep(d, filt)
    d = tremolo(d, rate_hz=0.1, depth=0.07, seed=21)
    d = mono_to_stereo(d)
    d = reverb_cathedral(d)

    # Layer 2: Just minor pad with breathing + auto-pan
    ji = [136.1, 136.1 * 6/5, 136.1 * 3/2]
    p = chord_pad(ji, dur, voices=5, detune_cents=12, amplitude=0.3)
    breath = breathing(dur, breath_rate=0.09, depth=0.25, floor=0.75)
    p = apply_amplitude_mod(p, breath)
    p = mono_to_stereo(p)
    p = auto_pan(p, rate_hz=0.04, depth=0.35)
    p = chorus_subtle(p)

    # Layer 3: Theta binaural
    bb = q.binaural("om_theta", dur, volume_db=-14.0)

    # Layer 4: Textural beds
    ocean = gen_texture("ocean_bed", dur, seed=22)
    shimmer = gen_texture("crystal_shimmer", dur, seed=23)
    shimmer_vol = fade_envelope([(0, 0.0), (3, 0.0), (5, 0.6), (7, 0.4), (8, 0.0)], dur)
    shimmer = apply_amplitude_mod(shimmer, shimmer_vol)

    # Master
    master = mix([d, p, bb, ocean, shimmer], volumes_db=[0, -3, -6, -8, -10])
    master = normalize_lufs(master, target_lufs=-14.0)
    master = master_chain(master)
    master = fade_in(master, 1.0)
    master = fade_out(master, 1.5)

    # Full visualization
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)

    ax_spec = fig.add_subplot(gs[0, :])
    specgram(master, ax_spec, "Full Mix Spectrogram — 5 layers", fmax=8000)

    ax_wave = fig.add_subplot(gs[1, :])
    t = np.linspace(0, dur, len(master))
    ax_wave.plot(t, master[:, 0], color="#9a4aba", lw=0.3, alpha=0.7, label="L")
    ax_wave.plot(t, master[:, 1], color="#4a9aba", lw=0.3, alpha=0.6, label="R")
    ax_wave.set_title("Waveform — L/R", fontsize=8, pad=3, color="#e0e0f0")
    ax_wave.set_xlim(0, dur)
    ax_wave.legend(fontsize=7)

    # Layer breakdown
    for i, (name, sig) in enumerate([("Drone", d), ("Pad", p), ("Ocean Bed", ocean)]):
        ax = fig.add_subplot(gs[2, min(i, 1)])
        if i == 2:
            specgram(sig, ax, name)
        else:
            specgram(sig, ax, name)

    peak_db = 20 * np.log10(np.max(np.abs(master)) + 1e-10)
    fig.suptitle(
        f"00 — HERO MIX — Om 136 Hz\n"
        f"Drone + Just Minor Pad + Binaural + Ocean Bed + Crystal Shimmer\n"
        f"Breathing | Cathedral Reverb | Auto-Pan | Mastered | Peak: {peak_db:.1f} dBFS",
        fontsize=10, color="#fff"
    )
    save_fig("00_hero_mix", fig)
    save_wav_stereo(master, "00_hero_mix", dur=WAV_CLIP)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("AUDIOMANCER GALLERY — 15 modules, 15 sections")
    print("=" * 65)
    print()

    section_waveforms()
    section_noise()
    section_drones()
    section_pads()
    section_binaural()
    section_effects()
    section_modulation()
    section_textures()
    section_spectral()
    section_spatial()
    section_harmony()
    section_envelopes()
    section_mastering()
    section_composition()
    section_hero()

    n_png = sum(1 for p in generated if p.suffix == ".png")
    n_wav = sum(1 for p in generated if p.suffix == ".wav")
    total_bytes = sum(p.stat().st_size for p in generated if p.exists())
    print()
    print("=" * 65)
    print(f"GALLERY COMPLETE")
    print(f"  {n_png} PNG  +  {n_wav} WAV clips")
    print(f"  Total: {total_bytes / 1024 / 1024:.1f} MB  —  {OUT}/")
    print("=" * 65)
