#!/usr/bin/env python3
"""Gallery — visual + audio showcase of all audiomancer capabilities.

Generates:
- PNG spectrograms, waveforms, and visualizations
- Short WAV clips (2-5s each, 8-bit mono to save space)
- Total output < 5 MB

Usage:
    python scripts/11_gallery.py
    => output/gallery/ (PNG + WAV files)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

# Project imports
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
from audiomancer.compose import fade_envelope, tremolo, make_loopable
from audiomancer.layers import mix, layer, crossfade, normalize_lufs
from audiomancer.field import clean, noise_gate
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR = SAMPLE_RATE
OUT = Path("output/gallery")
OUT.mkdir(parents=True, exist_ok=True)

# Custom dark colormap for spectrograms
CMAP_DARK = LinearSegmentedColormap.from_list(
    "audiomancer",
    ["#0a0a1a", "#1a0a3a", "#3a1a6a", "#6a2a9a", "#9a4aba", "#da7afa", "#ffc0ff"],
)

# Compact figure style
plt.rcParams.update({
    "figure.facecolor": "#0a0a1a",
    "axes.facecolor": "#0a0a1a",
    "axes.edgecolor": "#444",
    "text.color": "#ddd",
    "axes.labelcolor": "#ddd",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

generated = []  # track files for final report


def _harm(n: int) -> list[tuple[float, float]]:
    """Build harmonic list: [(1, 1.0), (2, 0.5), ...] with 1/n rolloff."""
    return [(i, 1.0 / i) for i in range(1, n + 1)]


def save_fig(name: str, fig=None):
    """Save and close figure."""
    path = OUT / f"{name}.png"
    if fig is None:
        fig = plt.gcf()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    generated.append(path)
    print(f"  [PNG] {path.name}")


def save_wav(signal: np.ndarray, name: str, sr: int = SR):
    """Save a short WAV clip (16-bit mono, max 3s to save space)."""
    max_samples = int(sr * 1.5)
    sig = signal[:max_samples]
    # Convert to mono if stereo
    if sig.ndim == 2:
        sig = np.mean(sig, axis=1)
    sig = normalize(sig, target_db=-1.0)
    path = OUT / f"{name}.wav"
    export_wav(sig, path, sample_rate=sr, bit_depth=16)
    generated.append(path)
    print(f"  [WAV] {path.name}")


def plot_waveform(signal, title, ax, color="#9a4aba", alpha=0.8):
    """Plot waveform on given axes."""
    t = np.linspace(0, len(signal) / SR, len(signal))
    if signal.ndim == 2:
        ax.plot(t, signal[:, 0], color=color, alpha=alpha, linewidth=0.3, label="L")
        ax.plot(t, signal[:, 1], color="#4a9aba", alpha=alpha, linewidth=0.3, label="R")
    else:
        ax.plot(t, signal, color=color, alpha=alpha, linewidth=0.3)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.1, 1.1)


def plot_spectrogram(signal, title, ax, sr=SR):
    """Plot spectrogram on given axes."""
    mono = stereo_to_mono(signal) if signal.ndim == 2 else signal
    f, t, Sxx = spectrogram(mono, fs=sr, nperseg=2048, noverlap=1536)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), cmap=CMAP_DARK,
                  shading="gouraud", vmin=-80, vmax=0)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_ylim(0, 8000)
    ax.set_ylabel("Hz")


# ===========================================================================
# GALLERY SECTIONS
# ===========================================================================

def section_waveforms():
    """01 — Basic waveform gallery."""
    print("\n[1/12] Waveforms")
    dur = 0.02  # 20ms for visual clarity
    waves = {
        "Sine 440 Hz": sine(440, dur),
        "Square 440 Hz": square(440, dur),
        "Sawtooth 440 Hz": sawtooth(440, dur),
        "Triangle 440 Hz": triangle(440, dur),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8, 4))
    colors = ["#9a4aba", "#da7afa", "#4a9aba", "#7adaba"]
    for ax, (name, sig), c in zip(axes.flat, waves.items(), colors):
        plot_waveform(sig, name, ax, color=c)
    fig.suptitle("BASIC WAVEFORMS", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("01_waveforms", fig)


def section_noise():
    """02 — Noise colors comparison."""
    print("[2/12] Noise Colors")
    dur = 2.0
    noises = {
        "White Noise": white_noise(dur),
        "Pink Noise (1/f)": pink_noise(dur),
        "Brown Noise (1/f²)": brown_noise(dur),
    }
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))
    colors = ["#da7afa", "#ff9a9a", "#9a6a4a"]
    for i, (name, sig) in enumerate(noises.items()):
        plot_waveform(sig[:SR], f"{name} — Waveform", axes[i, 0], color=colors[i])
        plot_spectrogram(sig, f"{name} — Spectrum", axes[i, 1])
    fig.suptitle("NOISE COLORS", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("02_noise_colors", fig)


def section_drones():
    """03 — Drone harmonics visualization."""
    print("[3/12] Drones & Harmonics")
    dur = 3.0
    h4 = [(i, 1.0 / i) for i in range(1, 5)]
    h8 = [(i, 1.0 / i) for i in range(1, 9)]
    h16 = [(i, 1.0 / i) for i in range(1, 17)]
    d1 = drone(136.1, dur, harmonics=h4)  # Om frequency
    d2 = drone(136.1, dur, harmonics=h8)
    d3 = drone(136.1, dur, harmonics=h16)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, sig, n in zip(axes, [d1, d2, d3], [4, 8, 16]):
        plot_spectrogram(sig, f"Om Drone — {n} harmonics", ax)
    fig.suptitle("HARMONIC DRONES (136.1 Hz Om)", fontsize=12, color="#fff", y=1.05)
    plt.tight_layout()
    save_fig("03_drones_harmonics", fig)

    # Save audio: rich 8-harmonic Om drone with cathedral reverb
    d = drone(136.1, 5.0, harmonics=_harm(8))
    d = mono_to_stereo(d)
    d = reverb_cathedral(d)
    save_wav(d, "03_om_drone_cathedral")


def section_pads():
    """04 — Chord pads and detuning."""
    print("[4/12] Chord Pads")
    dur = 3.0
    # C major triad with different voice counts
    freqs = [note_to_hz("C3"), note_to_hz("E3"), note_to_hz("G3")]
    p1 = chord_pad(freqs, dur, voices=1, detune_cents=0)
    p3 = chord_pad(freqs, dur, voices=3, detune_cents=10)
    p7 = chord_pad(freqs, dur, voices=7, detune_cents=25)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    titles = ["1 voice (clean)", "3 voices ±10¢", "7 voices ±25¢ (supersaw)"]
    for ax, sig, t in zip(axes, [p1, p3, p7], titles):
        plot_spectrogram(sig, t, ax)
    fig.suptitle("CHORD PADS — C Major (C3-E3-G3)", fontsize=12, color="#fff", y=1.05)
    plt.tight_layout()
    save_fig("04_chord_pads", fig)

    # Audio: lush pad
    p = chord_pad(freqs, 5.0, voices=7, detune_cents=20)
    p = mono_to_stereo(p)
    p = chorus(p, rate_hz=0.3, depth=0.2, mix=0.4)
    p = reverb_hall(p)
    save_wav(p, "04_lush_pad")


def section_binaural():
    """05 — Binaural beats visualization."""
    print("[5/12] Binaural Beats")
    dur = 3.0
    b_theta = from_preset("theta_deep", dur)
    b_alpha = from_preset("alpha_relax", dur)
    b_delta = from_preset("delta_sleep", dur)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    presets = [("Theta Deep (4 Hz)", b_theta, "#4a9aba"),
               ("Alpha Relax (10 Hz)", b_alpha, "#9a4aba"),
               ("Delta Sleep (2 Hz)", b_delta, "#2a6a4a")]
    for ax, (name, sig, c) in zip(axes, presets):
        t = np.linspace(0, dur, len(sig))
        ax.plot(t, sig[:, 0], color=c, alpha=0.7, linewidth=0.5, label="Left")
        ax.plot(t, sig[:, 1], color="#da7afa", alpha=0.5, linewidth=0.5, label="Right")
        ax.set_title(name, fontsize=9, pad=3)
        ax.set_xlim(0, 0.5)  # Zoom to show beat
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("BINAURAL BEATS — Zoomed to 0.5s", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("05_binaural_beats", fig)


def section_effects():
    """06 — Effects chain comparison."""
    print("[6/12] Effects Chain")
    dur = 3.0
    src = drone(220, dur, harmonics=_harm(6))
    src_s = mono_to_stereo(src)

    dry = src_s
    lp = lowpass(src, 800)
    lp_s = mono_to_stereo(lp)
    rev = reverb_cathedral(src_s)
    dly = delay(src_s, delay_seconds=0.3, feedback=0.5, mix=0.4)
    cho = chorus(src_s, rate_hz=0.5, depth=0.3, mix=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    sigs = [("Dry Signal", dry), ("Lowpass 800 Hz", lp_s),
            ("Cathedral Reverb", rev), ("Delay 300ms", dly),
            ("Chorus", cho), ("Compress -10 dB", compress(src_s, threshold_db=-10))]
    for ax, (name, sig) in zip(axes.flat, sigs):
        plot_spectrogram(sig, name, ax)
    fig.suptitle("EFFECTS CHAIN — 220 Hz Drone", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("06_effects_chain", fig)

    # Audio: effected drone
    save_wav(rev, "06_cathedral_drone")


def section_modulation():
    """07 — LFOs, drift, and modulation."""
    print("[7/12] Modulation")
    dur = 10.0
    lfo_s = lfo_sine(dur, rate_hz=0.1, depth=0.3, offset=1.0)
    lfo_t = lfo_triangle(dur, rate_hz=0.1, depth=0.3, offset=1.0)
    dr = drift(dur, speed=0.2, depth=0.3, offset=1.0, seed=42)
    ev = evolving_lfo(dur, rate_hz=0.1, depth=0.3, offset=1.0, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    mods = [("LFO Sine (0.1 Hz)", lfo_s, "#9a4aba"),
            ("LFO Triangle (0.1 Hz)", lfo_t, "#4a9aba"),
            ("Brownian Drift", dr, "#da7afa"),
            ("Evolving LFO", ev, "#7adaba")]
    for ax, (name, mod, c) in zip(axes.flat, mods):
        t = np.linspace(0, dur, len(mod))
        ax.plot(t, mod, color=c, linewidth=1.0)
        ax.set_title(name, fontsize=9, pad=3)
        ax.axhline(1.0, color="#444", linewidth=0.5, linestyle="--")
        ax.set_xlim(0, dur)
    fig.suptitle("MODULATION SOURCES", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("07_modulation", fig)

    # Audio: filter-swept drone
    d = drone(136.1, 5.0, harmonics=_harm(8))
    sweep = lfo_sine(5.0, rate_hz=0.15, depth=2000, offset=3000)
    swept = apply_filter_sweep(d, sweep)
    swept = mono_to_stereo(swept)
    swept = reverb_hall(swept)
    save_wav(swept, "07_filter_sweep")


def section_textures():
    """08 — All 9 texture presets."""
    print("[8/12] Texture Bank (9 presets)")
    dur = 5.0
    names = list(REGISTRY.keys())

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for ax, name in zip(axes.flat, names):
        sig = gen_texture(name, dur, seed=42)
        plot_spectrogram(sig, name.replace("_", " ").title(), ax)
    fig.suptitle("TEXTURE BANK — 9 Evolving Presets (5s each)",
                 fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("08_texture_bank", fig)

    # Audio: deep space texture
    ds = gen_texture("deep_space", 5.0, seed=42)
    save_wav(ds, "08_deep_space")


def section_spectral():
    """09 — Spectral processing gallery."""
    print("[9/12] Spectral Processing")
    dur = 3.0
    src = drone(220, dur, harmonics=_harm(6))

    frozen = freeze(src, freeze_time=1.0, duration_sec=dur)
    blurred = blur(src, amount=0.8)
    shifted_up = pitch_shift(src, semitones=7)
    shifted_dn = pitch_shift(src, semitones=-5)
    gated = spectral_gate(src, threshold_db=-30)

    # Morph: drone → noise
    noise = pink_noise(dur)
    morphed = morph(src, noise, mix=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    sigs = [("Original Drone", src), ("Spectral Freeze @1s", frozen),
            ("Spectral Blur 0.8", blurred), ("Pitch +7 (5th up)", shifted_up),
            ("Pitch -5 (4th down)", shifted_dn), ("Morph: Drone↔Noise 50%", morphed)]
    for ax, (name, sig) in zip(axes.flat, sigs):
        plot_spectrogram(sig, name, ax)
    fig.suptitle("SPECTRAL PROCESSING", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("09_spectral", fig)

    # Audio: frozen drone
    f = freeze(src, freeze_time=1.0, duration_sec=5.0)
    f = mono_to_stereo(f)
    f = reverb_cathedral(f)
    save_wav(f, "09_frozen_drone")


def section_spatial():
    """10 — Spatial audio visualization."""
    print("[10/12] Spatial Audio")
    dur = 3.0
    src = sine(440, dur)

    panned_l = pan(src, position=-0.8)
    panned_r = pan(src, position=0.8)
    ap = auto_pan(src, rate_hz=0.5, depth=1.0)
    rot = rotate(src, revolutions=2.0)
    haas = haas_width(src, delay_ms=20)

    # Stereo width demo
    stereo_src = np.column_stack([sine(440, dur), sine(550, dur)])
    wide = stereo_width(stereo_src, width=2.0)
    narrow = stereo_width(stereo_src, width=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    sigs = [("Pan Left -0.8", panned_l), ("Pan Right +0.8", panned_r),
            ("Auto-Pan 0.5 Hz", ap), ("Rotate 2 rev", rot),
            ("Haas Width 20ms", haas), ("Stereo Width 2.0x", wide)]
    for ax, (name, sig) in zip(axes.flat, sigs):
        t = np.linspace(0, len(sig) / SR, len(sig))
        ax.plot(t, sig[:, 0], color="#9a4aba", alpha=0.7, linewidth=0.4, label="L")
        ax.plot(t, sig[:, 1], color="#4a9aba", alpha=0.7, linewidth=0.4, label="R")
        ax.set_title(name, fontsize=9, pad=3)
        ax.set_xlim(0, min(0.05, t[-1]))  # Zoom to see stereo difference
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("SPATIAL AUDIO", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("10_spatial", fig)

    # Audio: auto-panned rotating drone
    d = drone(220, 5.0, harmonics=_harm(6))
    d = rotate(d, revolutions=1.0)
    d = reverb_hall(d)
    save_wav(d, "10_rotating_drone")


def section_harmony():
    """11 — Harmony and tuning systems."""
    print("[11/12] Harmony & Tuning")

    # Scales visualization
    scale_names = ["major", "pentatonic_minor", "hirajoshi", "pelog",
                   "raga_bhairav", "prometheus", "blues", "whole_tone"]
    root_hz = note_to_hz("C3")

    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for ax, sname in zip(axes.flat, scale_names):
        freqs = scale(root_hz, sname, octaves=2)
        # Generate a quick arpeggio
        note_dur = 0.3
        arp = np.concatenate([sine(f, note_dur) * 0.5 for f in freqs])
        arp = fade_in(arp, 0.01)
        arp = fade_out(arp, 0.01)
        plot_spectrogram(arp, sname.replace("_", " ").title(), ax, sr=SR)
        ax.set_ylim(0, 2000)
    fig.suptitle("SCALES — Arpeggiated (C3 root, 2 octaves)",
                 fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("11_scales", fig)

    # Tuning comparison: equal temperament vs just intonation
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # Equal temperament chord
    et_freqs = [note_to_hz("C3"), note_to_hz("E3"), note_to_hz("G3")]
    et_chord = chord_pad(et_freqs, 2.0, voices=1)
    plot_spectrogram(et_chord, "Equal Temperament C-E-G", axes[0])

    # Just intonation chord
    ji_freqs = just_chord(note_to_hz("C3"), "major")
    ji_chord = chord_pad(ji_freqs, 2.0, voices=1)
    plot_spectrogram(ji_chord, "Just Intonation C-E-G", axes[1])

    # Fibonacci frequencies
    fib_freqs = fibonacci_freqs(note_to_hz("C3"), n=6)
    fib_chord = chord_pad(fib_freqs, 2.0, voices=1)
    plot_spectrogram(fib_chord, "Fibonacci Ratios", axes[2])

    for ax in axes:
        ax.set_ylim(0, 2000)
    fig.suptitle("TUNING SYSTEMS", fontsize=12, color="#fff", y=1.05)
    plt.tight_layout()
    save_fig("11_tuning_systems", fig)

    # Audio: just intonation pad
    ji = just_chord(note_to_hz("C3"), "maj7")
    p = chord_pad(ji, 5.0, voices=5, detune_cents=8)
    p = mono_to_stereo(p)
    p = reverb_cathedral(p)
    save_wav(p, "11_just_intonation_maj7")


def section_envelopes():
    """12 — Envelope shapes."""
    print("[12/12] Envelopes")
    dur = 2.0

    envs = {
        "ADSR (linear)": adsr(dur, attack=0.1, decay=0.2, sustain=0.6, release=0.5),
        "ADSR (exponential)": adsr_exp(dur, attack=0.1, decay=0.2, sustain=0.6,
                                        release=0.5, curve=4.0),
        "AR (symmetric)": ar(dur, attack=0.5, curve=2.0),
        "Swell (peak @70%)": swell(dur, peak_time=0.7, curve=3.0),
        "Breathing (0.5 Hz)": breathing(dur, breath_rate=0.5, depth=0.4, floor=0.6),
        "Gate [1,0,1,1,0,1]": gate_pattern(dur, [1, 0, 1, 1, 0, 1], step_sec=0.15),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    colors = ["#9a4aba", "#da7afa", "#4a9aba", "#7adaba", "#ff9a9a", "#ffc0ff"]
    for ax, (name, env), c in zip(axes.flat, envs.items(), colors):
        t = np.linspace(0, dur, len(env))
        ax.fill_between(t, 0, env, color=c, alpha=0.3)
        ax.plot(t, env, color=c, linewidth=1.5)
        ax.set_title(name, fontsize=9, pad=3)
        ax.set_xlim(0, dur)
        ax.set_ylim(-0.05, 1.1)
    fig.suptitle("ENVELOPE SHAPES", fontsize=12, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("12_envelopes", fig)

    # Audio: gated pad
    freqs = [note_to_hz("D3"), note_to_hz("F#3"), note_to_hz("A3")]
    p = chord_pad(freqs, 5.0, voices=5, detune_cents=15)
    gate = gate_pattern(5.0, [1, 0, 0.5, 1, 0, 0.8, 1, 0], step_sec=0.25,
                        smoothing_ms=10)
    p = apply_amplitude_mod(p, gate)
    p = mono_to_stereo(p)
    p = reverb_hall(p)
    save_wav(p, "12_gated_pad")


# ---------------------------------------------------------------------------
# HERO PIECE — Everything combined
# ---------------------------------------------------------------------------

def section_hero():
    """HERO — Full production piece combining all modules."""
    print("\n[HERO] Full Production Mix")
    dur = 5.0

    # Layer 1: Om drone with evolving filter (synth + modulation + effects)
    d = drone(136.1, dur, harmonics=_harm(8), amplitude=0.4)
    sweep = drift(dur, speed=0.1, depth=1500, offset=2500, seed=1)
    d = apply_filter_sweep(d, sweep)
    d = mono_to_stereo(d)
    d = reverb_cathedral(d)

    # Layer 2: Just intonation pad with breathing (harmony + envelope)
    ji_freqs = just_chord(136.1, "minor")
    p = chord_pad(ji_freqs, dur, voices=5, detune_cents=12, amplitude=0.3)
    breath = breathing(dur, breath_rate=0.12, depth=0.3, floor=0.7)
    p = apply_amplitude_mod(p, breath)
    p = mono_to_stereo(p)
    p = chorus_subtle(p)
    p = auto_pan(p, rate_hz=0.04, depth=0.3)

    # Layer 3: Theta binaural (binaural)
    b = binaural(136.1, 6.0, dur, amplitude=0.15)

    # Layer 4: Deep space texture (textures)
    tex = gen_texture("ocean_bed", dur, seed=42)
    tex = normalize(tex, target_db=-12)

    # Mix everything
    master = mix([d, p, b, tex], volumes_db=[0, -3, -8, -6])
    master = normalize_lufs(master, target_lufs=-14)
    master = fade_in(master, 0.5)
    master = fade_out(master, 1.0)

    # Visualize the hero piece
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_spectrogram(master, "", axes[0])
    axes[0].set_title("SPECTROGRAM", fontsize=10, pad=3)

    t = np.linspace(0, dur, len(master))
    axes[1].plot(t, master[:, 0], color="#9a4aba", alpha=0.6, linewidth=0.3, label="L")
    axes[1].plot(t, master[:, 1], color="#4a9aba", alpha=0.6, linewidth=0.3, label="R")
    axes[1].set_title("WAVEFORM", fontsize=10, pad=3)
    axes[1].set_xlim(0, dur)
    axes[1].legend(fontsize=8)

    fig.suptitle("HERO MIX — Om Drone + Just Minor Pad + Theta Binaural + Ocean Bed\n"
                 "136.1 Hz | Breathing Envelope | Cathedral Reverb | Auto-Pan | -14 LUFS",
                 fontsize=11, color="#fff", y=1.02)
    plt.tight_layout()
    save_fig("00_hero_mix", fig)
    save_wav(master, "00_hero_mix")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIOMANCER GALLERY — Showcasing 14 modules")
    print("=" * 60)

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
    section_hero()

    # Final report
    total_bytes = sum(p.stat().st_size for p in generated)
    n_png = sum(1 for p in generated if p.suffix == ".png")
    n_wav = sum(1 for p in generated if p.suffix == ".wav")
    print(f"\n{'=' * 60}")
    print(f"GALLERY COMPLETE")
    print(f"  {n_png} PNG images + {n_wav} WAV clips")
    print(f"  Total size: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"  Output: {OUT}/")
    print(f"{'=' * 60}")
