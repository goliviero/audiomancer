"""Stem builders — parametric generators consumed by the config-driven renderer.

Each builder takes `duration`, `seed`, `sample_rate` + builder-specific kwargs
and returns a stereo numpy array WITHOUT LUFS/master/loop (those are handled
by `scripts/render_stem.py` and `scripts/render_mix.py`).

Registry pattern: a config (e.g. configs/V005.py) names a builder via its
string key, and the renderer dispatches through REGISTRY.

Introduced with V006 (post-V005 refactor). V004 and V005 archive scripts in
scripts/ stay as-is — historical reference, reproducible per-video.
"""

import hashlib

import numpy as np

import audiomancer.quick as _q
from audiomancer.effects import (
    chorus_subtle,
    highpass,
    lowpass,
    reverb,
)
from audiomancer.modulation import (
    apply_amplitude_mod,
    apply_filter_sweep,
    evolving_lfo,
    multi_lfo,
    random_walk,
)
from audiomancer.spatial import auto_pan, haas_width
from audiomancer.synth import sine, triangle
from audiomancer.utils import mono_to_stereo


def derived_seed(root: int, role: str) -> int:
    """Coordinated-but-not-sync seed derivation per role."""
    h = int(hashlib.md5(role.encode()).hexdigest()[:8], 16)
    return (root + h) % (2**31)


# ---------------------------------------------------------------------------
# Pad alive — multi-voice C major open with spectral movement (anti-tell)
# ---------------------------------------------------------------------------

def _voice(freq: float, duration: float, sample_rate: int,
           seed: int) -> np.ndarray:
    """Single voice with 3 detuned sines (seeded micro-jitter)."""
    rng = np.random.default_rng(seed)
    offsets = np.array([-2.0, 0.0, 2.0]) + rng.uniform(-1.0, 1.0, size=3)
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    sig = np.zeros_like(t)
    for cents in offsets:
        detuned = freq * 2 ** (cents / 1200)
        sig += np.sin(2 * np.pi * detuned * t)
    return sig / 3.0


def pad_alive(duration: float, seed: int, sample_rate: int,
              chord: list[float],
              intensity: str = "moderate") -> np.ndarray:
    """Pad with filter drift + per-voice rotation + auto-pan.

    Args:
        duration, seed, sample_rate: common context.
        chord: list of fundamental frequencies (Hz), 1-5 voices typical.
        intensity: 'gentle' | 'moderate' | 'strong' — drives filter & voice depth.

    Returns stereo ndarray (n, 2) at peak ~0.85, un-mastered.
    """
    if intensity == "gentle":
        filter_center, filter_depth = 2800, 800
        voice_mod_depth = 0.15
        use_pan = False
    elif intensity == "strong":
        filter_center, filter_depth = 2200, 1800
        voice_mod_depth = 0.55
        use_pan = True
    else:  # moderate
        filter_center, filter_depth = 2500, 1500
        voice_mod_depth = 0.35
        use_pan = True

    # Per-voice modulation rates (prime-ish to avoid beat alignment)
    voice_rates = [1 / 19.0, 1 / 29.0, 1 / 13.0, 1 / 23.0, 1 / 17.0]
    voice_amps = [0.9, 0.8, 0.5, 0.35, 0.25]
    n_voices = len(chord)
    voice_rates = voice_rates[:n_voices]
    voice_amps = voice_amps[:n_voices]

    signal = np.zeros(int(duration * sample_rate))
    for i, (freq, rate, base_amp) in enumerate(zip(chord, voice_rates, voice_amps)):
        raw = _voice(freq, duration, sample_rate, seed=seed + i * 17)
        mod = evolving_lfo(
            duration, rate_hz=rate, depth=voice_mod_depth, offset=1.0,
            drift_speed=0.05,
            seed=seed + i * 31, sample_rate=sample_rate,
        )
        signal += raw * base_amp * mod

    # Normalize before filter
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * 0.85 / peak

    # Dynamic LP filter sweep
    walk = random_walk(duration, sigma=0.3, tau=4.0,
                       seed=seed + 7, sample_rate=sample_rate)
    filter_curve = filter_center + filter_depth * (walk - 1.0)
    filter_curve = np.clip(filter_curve, 400, 6000)
    signal = apply_filter_sweep(signal, filter_curve, sample_rate=sample_rate)

    stereo = mono_to_stereo(signal)

    if use_pan:
        stereo = auto_pan(stereo, rate_hz=0.04, depth=0.35, center=0.0,
                          sample_rate=sample_rate)

    stereo = chorus_subtle(stereo, sample_rate=sample_rate)
    stereo = reverb(stereo, room_size=0.80, damping=0.6, wet_level=0.50,
                    sample_rate=sample_rate)

    # Gentle overall volume breath
    breath = multi_lfo(
        duration, layers=[(1 / 8.0, 0.02), (1 / 25.0, 0.06)],
        seed=seed, sample_rate=sample_rate,
    )
    stereo = apply_amplitude_mod(stereo, breath)
    return stereo


# ---------------------------------------------------------------------------
# Pendulum bass — 2-note pendulum (e.g. C2 <-> G2)
# ---------------------------------------------------------------------------

def _bass_note(freq: float, dur: float, sample_rate: int,
               triangle_db: float = -18.0) -> np.ndarray:
    """Pure sine + subtle triangle."""
    gain = 10 ** (triangle_db / 20)
    main = sine(freq, dur, amplitude=0.7, sample_rate=sample_rate)
    tri = triangle(freq, dur, amplitude=0.7 * gain, sample_rate=sample_rate)
    return main + tri


def pendulum_bass(duration: float, seed: int, sample_rate: int,
                  pendulum: list[float],
                  note_dur: float = 60.0,
                  xfade: float = 15.0,
                  lp_hz: float = 300.0,
                  reverb_room: float = 0.4,
                  reverb_wet: float = 0.2,
                  random_walk_sigma: float = 0.05,
                  random_walk_tau: float = 30.0) -> np.ndarray:
    """Two-note pendulum bass (e.g. C2 <-> G2 for V005 grounding).

    Note duration jitter (+-2s) + random walk amplitude for anti-tell.
    """
    rng = np.random.default_rng(seed)
    # Random start direction
    pend = list(pendulum)
    if rng.random() > 0.5:
        pend = list(reversed(pend))

    n_samples = int(duration * sample_rate)
    xfade_samples = int(xfade * sample_rate)

    output = np.zeros(n_samples)
    pos = 0
    note_idx = 0

    while pos < n_samples:
        detune_cents = rng.uniform(-1.0, 1.0)
        freq = pend[note_idx % len(pend)] * 2 ** (detune_cents / 1200)
        note_idx += 1

        this_note_dur = note_dur + rng.uniform(-2.0, 2.0)
        note_samples_var = int(this_note_dur * sample_rate)
        note_len_sec = this_note_dur + xfade
        note = _bass_note(freq, note_len_sec, sample_rate)

        hold_samples = max(0, note_samples_var - xfade_samples)
        fade_in_env = np.linspace(0, 1, xfade_samples)
        fade_out_env = np.linspace(1, 0, xfade_samples)
        env = np.concatenate([fade_in_env, np.ones(hold_samples), fade_out_env])
        env = env[:len(note)]
        if len(env) < len(note):
            note = note[:len(env)]
        note = note * env

        end = min(pos + len(note), n_samples)
        chunk = end - pos
        output[pos:end] += note[:chunk]
        pos += note_samples_var

    output = lowpass(output, cutoff_hz=lp_hz, sample_rate=sample_rate)
    stereo = mono_to_stereo(output)
    stereo = reverb(stereo, room_size=reverb_room, damping=0.5,
                    wet_level=reverb_wet, sample_rate=sample_rate)

    walk = random_walk(duration, sigma=random_walk_sigma, tau=random_walk_tau,
                       seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, walk)
    return stereo


# ---------------------------------------------------------------------------
# Arpege bass — N-note palindrome across octaves
# ---------------------------------------------------------------------------

def arpege_bass(duration: float, seed: int, sample_rate: int,
                arpege: list[float],
                note_dur: float = 15.0,
                xfade: float = 3.0,
                hp_hz: float = 50.0,
                lp_hz: float = 500.0,
                haas_ms: float = 3.0,
                reverb_room: float = 0.5,
                reverb_wet: float = 0.35) -> np.ndarray:
    """Arpege bass across N notes (palindrome or monotonic)."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sample_rate)
    xfade_samples = int(xfade * sample_rate)

    output = np.zeros(n_samples)
    pos = 0
    note_idx = 0

    while pos < n_samples:
        detune_cents = rng.uniform(-1.0, 1.0)
        freq = arpege[note_idx % len(arpege)] * 2 ** (detune_cents / 1200)
        note_idx += 1

        this_note_dur = note_dur + rng.uniform(-1.0, 1.0)
        note_samples_var = int(this_note_dur * sample_rate)
        note_len_sec = this_note_dur + xfade
        note = _bass_note(freq, note_len_sec, sample_rate)

        hold_samples = max(0, note_samples_var - xfade_samples)
        fade_in_env = np.linspace(0, 1, xfade_samples)
        fade_out_env = np.linspace(1, 0, xfade_samples)
        env = np.concatenate([fade_in_env, np.ones(hold_samples), fade_out_env])
        env = env[:len(note)]
        if len(env) < len(note):
            note = note[:len(env)]
        note = note * env

        end = min(pos + len(note), n_samples)
        chunk = end - pos
        output[pos:end] += note[:chunk]
        pos += note_samples_var

    output = highpass(output, cutoff_hz=hp_hz, sample_rate=sample_rate)
    output = lowpass(output, cutoff_hz=lp_hz, sample_rate=sample_rate)
    stereo = mono_to_stereo(output)
    stereo = haas_width(stereo, delay_ms=haas_ms, sample_rate=sample_rate)
    stereo = chorus_subtle(stereo, sample_rate=sample_rate)
    stereo = reverb(stereo, room_size=reverb_room, damping=0.5,
                    wet_level=reverb_wet, sample_rate=sample_rate)
    walk = random_walk(duration, sigma=0.05, tau=30.0,
                       seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, walk)
    return stereo


# ---------------------------------------------------------------------------
# Binaural beat — carrier + beat frequency, seeded carrier jitter
# ---------------------------------------------------------------------------

def binaural_beat(duration: float, seed: int, sample_rate: int,
                  carrier_hz: float,
                  beat_hz: float,
                  volume_db: float = -6.0,
                  carrier_jitter_cents: float = 2.0) -> np.ndarray:
    """Binaural beat. Beat frequency stays exact (critical), carrier jitters.

    Args:
        carrier_hz: Base carrier frequency.
        beat_hz: Left/right difference in Hz (binaural beat).
        volume_db: Output level.
        carrier_jitter_cents: +-cents random applied to carrier (beat unchanged).
    """
    rng = np.random.default_rng(seed)
    c = carrier_hz * 2 ** (rng.uniform(-carrier_jitter_cents,
                                       carrier_jitter_cents) / 1200)
    return _q.binaural_custom(carrier_hz=c, beat_hz=beat_hz,
                              duration_sec=duration, volume_db=volume_db,
                              sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Registry — string key -> builder function
# ---------------------------------------------------------------------------

REGISTRY = {
    "pad_alive": pad_alive,
    "pendulum_bass": pendulum_bass,
    "arpege_bass": arpege_bass,
    "binaural_beat": binaural_beat,
}
