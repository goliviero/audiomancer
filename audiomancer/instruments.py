"""Synthetic ethnic instrument generators.

Produces recognizable approximations of 5 acoustic instruments via pure
synthesis — no samples required. Quality is "ambient-usable" but NOT
substitute for real recordings for ethnomusicological purposes.

For higher realism with free pitch control, see audiomancer.sampler (loads
CC0 one-shot samples + pitch-shifts to any target note).

Functions:
    didgeridoo    — low drone + formant vocal modulation + breath rhythm
    handpan       — metallic inharmonic bell with fast attack / long decay
    oud           — plucked fretless lute via Karplus + body resonance
    sitar         — plucked string with jawari buzz + sympathetic strings
    derbouka_hit  — goblet drum single hit (dum / tek)
    derbouka_pattern — simple pattern sequencer on top of derbouka_hit
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audiomancer import SAMPLE_RATE
from audiomancer.synth import (
    _normalize_peak,
    karplus_strong,
    sine,
)

DEFAULT_AMPLITUDE = 0.5


# ---------------------------------------------------------------------------
# Didgeridoo — drone + vocal formant
# ---------------------------------------------------------------------------

def didgeridoo(frequency: float = 73.0, duration_sec: float = 5.0,
               breath_rate: float = 0.35,
               formant_shift: float = 0.0,
               amplitude: float = DEFAULT_AMPLITUDE,
               seed: int | None = None,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic didgeridoo — drone + vocal formants + breath rhythm.

    Args:
        frequency: Fundamental (typical range 60-80 Hz for real instruments).
        duration_sec: Total duration.
        breath_rate: Hz of the breath amplitude modulation (~0.3 Hz typical).
        formant_shift: +-1.0 semitone shift of formant peaks (vocal "aah" vs "ooh").
        seed: Random seed for breath noise jitter.

    Returns mono signal.
    """
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False)

    # Base drone: fundamental + harmonic series (overtone-rich)
    signal = np.zeros(n)
    harmonics = [(1, 1.0), (2, 0.7), (3, 0.5), (4, 0.35), (5, 0.25),
                 (6, 0.18), (7, 0.12), (8, 0.08)]
    nyquist = sample_rate / 2
    for h, amp in harmonics:
        freq = frequency * h
        if freq >= nyquist:
            continue
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Vocal formant emphasis (parallel bandpass at ~500 Hz and ~1200 Hz)
    # shift with formant_shift (semitones)
    f1_base = 500.0 * 2 ** (formant_shift / 12)
    f2_base = 1200.0 * 2 ** (formant_shift / 12)

    sos_f1 = butter(2, [f1_base * 0.8 / nyquist, f1_base * 1.25 / nyquist],
                    btype="band", output="sos")
    sos_f2 = butter(2, [f2_base * 0.8 / nyquist, f2_base * 1.25 / nyquist],
                    btype="band", output="sos")
    f1 = sosfiltfilt(sos_f1, signal)
    f2 = sosfiltfilt(sos_f2, signal)
    signal = signal + 0.3 * f1 + 0.2 * f2

    # Breath rhythm: pulsating amplitude mod (not pure sine — slightly asymmetric)
    rng = np.random.default_rng(seed)
    breath_phase = rng.uniform(0, 2 * np.pi)
    breath = 0.85 + 0.15 * (0.6 * np.sin(2 * np.pi * breath_rate * t + breath_phase)
                           + 0.4 * np.sin(2 * np.pi * breath_rate * 2 * t + breath_phase))
    signal = signal * breath

    # Subtle breath noise (air column) — seeded white noise shaped pink-ish
    # via a steep lowpass. Deterministic with `seed`.
    white = rng.standard_normal(n) * 0.04
    sos_n = butter(3, 1500 / nyquist, btype="low", output="sos")
    noise_bed = sosfiltfilt(sos_n, white)
    signal = signal + noise_bed * breath

    return _normalize_peak(signal, amplitude)


# ---------------------------------------------------------------------------
# Handpan — inharmonic metallic bell
# ---------------------------------------------------------------------------

def handpan(frequency: float = 146.83, duration_sec: float = 4.0,
            inharmonicity: float = 0.08,
            decay: float = 0.998,
            amplitude: float = DEFAULT_AMPLITUDE,
            seed: int | None = None,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic handpan / Hang drum — inharmonic metallic bell.

    Additive synthesis with slightly stretched partials (inharmonic) + fast
    attack + exponential decay 3-5s. Sounds like a bright singing bowl /
    steel drum hybrid. Not a perfect handpan but recognizable in mix.

    Args:
        frequency: Fundamental (D3 = 146.83 is a common handpan note).
        duration_sec: Total duration (3-5s realistic).
        inharmonicity: Stretched partial factor. 0 = harmonic, 0.1 = metallic.
        decay: Envelope decay rate.
        seed: Small random phase per partial.

    Returns mono signal.
    """
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    rng = np.random.default_rng(seed)

    # Partial ratios inspired by real handpan: strong fundamental + octave + 5th
    # Each partial stretched slightly (inharmonic)
    partial_ratios = [1.0, 2.01, 3.02, 4.05, 6.1]
    partial_amps = [1.0, 0.55, 0.4, 0.25, 0.15]

    signal = np.zeros(n)
    nyquist = sample_rate / 2
    for base_ratio, amp in zip(partial_ratios, partial_amps):
        # Add small inharmonic stretch
        ratio = base_ratio * (1 + inharmonicity * (base_ratio - 1) / 5)
        freq = frequency * ratio
        if freq >= nyquist:
            continue
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    # Fast attack (5ms) + exponential decay
    attack_n = int(0.005 * sample_rate)
    env = np.zeros(n)
    env[:attack_n] = np.linspace(0, 1, attack_n)
    # Decay: exp(-t * (1-decay) * sample_rate)
    decay_factor = (1 - decay) * sample_rate
    env[attack_n:] = np.exp(-(t[attack_n:] - t[attack_n]) * decay_factor)

    signal = signal * env

    # Subtle metallic shimmer: high partial + noise burst at attack
    shimmer = sine(frequency * 8.1, duration_sec, amplitude=0.06,
                   sample_rate=sample_rate)
    shimmer_env = np.exp(-t * decay_factor * 2)
    signal += shimmer * shimmer_env

    return _normalize_peak(signal, amplitude)


# ---------------------------------------------------------------------------
# Oud — plucked fretless lute via Karplus + body resonance
# ---------------------------------------------------------------------------

def oud(frequency: float = 146.83, duration_sec: float = 3.0,
        body_resonance_hz: float = 400.0,
        amplitude: float = DEFAULT_AMPLITUDE,
        seed: int | None = None,
        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic oud — Karplus-Strong + body resonance bandpass.

    Args:
        frequency: Fundamental (D3 classic oud tuning is 146.83).
        duration_sec: Total duration (3s realistic for a single note).
        body_resonance_hz: Center of body resonance bandpass (200-600 typical).
        seed: Random seed for Karplus initial noise.

    Returns mono signal.
    """
    # Base: Karplus plucked string, moderate decay
    kp = karplus_strong(frequency, duration_sec, decay=0.997, brightness=0.6,
                        amplitude=0.8, seed=seed, sample_rate=sample_rate)

    # Body resonance: gentle bandpass around body_resonance_hz
    nyquist = sample_rate / 2
    lo = max(50, body_resonance_hz * 0.5) / nyquist
    hi = min(nyquist - 100, body_resonance_hz * 2.0) / nyquist
    sos = butter(2, [lo, hi], btype="band", output="sos")
    body = sosfiltfilt(sos, kp)
    # Mix original + body resonance (parallel bandpass emphasis)
    signal = kp + 0.4 * body

    return _normalize_peak(signal, amplitude)


# ---------------------------------------------------------------------------
# Sitar — plucked + jawari buzz + sympathetic strings
# ---------------------------------------------------------------------------

def sitar(frequency: float = 196.0, duration_sec: float = 4.0,
          buzz_amount: float = 0.35,
          sympathetic_strings: bool = True,
          amplitude: float = DEFAULT_AMPLITUDE,
          seed: int | None = None,
          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic sitar — Karplus + jawari buzz + sympathetic strings.

    The jawari bridge is approximated by a mild waveshaping + comb filter.
    Sympathetic strings are detuned quiet Karplus layers.

    Args:
        frequency: Main string fundamental (G3 = 196 common for sitar).
        buzz_amount: Intensity of jawari buzz (0 = clean, 1 = harsh).
        sympathetic_strings: If True, adds 4 detuned quiet strings.
        seed: Random seed.

    Returns mono signal.
    """
    # Main plucked string
    main = karplus_strong(frequency, duration_sec, decay=0.9985, brightness=0.7,
                          amplitude=0.8, seed=seed, sample_rate=sample_rate)

    # Jawari buzz: shape with asymmetric soft-clip + a short comb filter
    if buzz_amount > 0:
        # Asymmetric clip emphasizes odd harmonics -> "buzzy"
        driven = np.tanh(main * (1 + 3 * buzz_amount))
        # Short comb (2ms delay, 0.5 feedback) for metallic edge
        comb_samples = int(0.002 * sample_rate)
        combed = driven.copy()
        for i in range(comb_samples, len(combed)):
            combed[i] += 0.5 * combed[i - comb_samples]
        main = main * (1 - buzz_amount) + combed * buzz_amount * 0.6
        # Renormalize after buzz
        peak = np.max(np.abs(main))
        if peak > 0:
            main = main * 0.8 / peak

    # Sympathetic strings: 4 quieter Karplus at just-intonation intervals
    if sympathetic_strings:
        sympa_freqs = [frequency * r for r in (9 / 8, 5 / 4, 4 / 3, 3 / 2)]
        rng = np.random.default_rng((seed or 0) + 7919)
        for i, sf in enumerate(sympa_freqs):
            sp = karplus_strong(sf, duration_sec, decay=0.999,
                                brightness=0.5, amplitude=0.15,
                                seed=int(rng.integers(0, 100000)),
                                sample_rate=sample_rate)
            # Slight delay so sympathetics come in after main pluck
            delay_samples = int((0.02 + i * 0.01) * sample_rate)
            padded = np.concatenate([np.zeros(delay_samples), sp])[:len(main)]
            main = main + padded

    return _normalize_peak(main, amplitude)


# ---------------------------------------------------------------------------
# Derbouka — goblet drum hits (dum / tek)
# ---------------------------------------------------------------------------

def derbouka_hit(hit_type: str = "dum",
                 duration_sec: float = 0.5,
                 amplitude: float = DEFAULT_AMPLITUDE,
                 seed: int | None = None,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Single derbouka (goblet drum) hit.

    Args:
        hit_type: 'dum' (low center hit) or 'tek' (high edge slap).
        duration_sec: Sample duration (0.3-0.8s typical).
        seed: Random seed for the noise burst component.

    Returns mono signal.
    """
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    nyquist = sample_rate / 2

    if hit_type == "dum":
        # Low bass hit: 80 Hz fundamental + sine sweep down + noise burst
        tone_freq = 80.0
        decay_rate = 15.0
        noise_amp = 0.25
        noise_cutoff = 400
    elif hit_type == "tek":
        # High slap: 900 Hz + sharper attack + brighter noise
        tone_freq = 900.0
        decay_rate = 40.0
        noise_amp = 0.5
        noise_cutoff = 4000
    else:
        raise ValueError(f"hit_type must be 'dum' or 'tek', got {hit_type!r}")

    # Main tone with pitch drop (skin stretch modulation)
    pitch_env = np.exp(-t * decay_rate * 0.3)
    instantaneous_freq = tone_freq * (1.0 + 0.3 * pitch_env)
    phase = np.cumsum(2 * np.pi * instantaneous_freq / sample_rate)
    tone = np.sin(phase)

    # Noise burst (skin friction)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # Filter the noise to body range
    sos = butter(2, noise_cutoff / nyquist, btype="low", output="sos")
    noise = sosfiltfilt(sos, noise)

    # Short attack (1-2 ms) then exponential decay
    attack_n = max(1, int(0.001 * sample_rate))
    env = np.zeros(n)
    env[:attack_n] = np.linspace(0, 1, attack_n)
    env[attack_n:] = np.exp(-(t[attack_n:] - t[attack_n]) * decay_rate)

    signal = (tone + noise_amp * noise) * env
    return _normalize_peak(signal, amplitude)


def derbouka_pattern(pattern: str = "D t t D t t D t",
                     bpm: float = 120.0,
                     duration_sec: float | None = None,
                     amplitude: float = DEFAULT_AMPLITUDE,
                     seed: int | None = None,
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a derbouka rhythm from a simple text pattern.

    Pattern alphabet:
        D = dum (low)
        t = tek (high)
        . = rest (silence)
        Spaces are ignored.

    Args:
        pattern: Text pattern, e.g. "D t t D t t D t".
        bpm: Beats per minute (each symbol = 1/8 note by default).
        duration_sec: If None, duration = one pattern iteration. If set,
            the pattern loops to fill the duration.
        seed: Base seed (each hit gets a derived seed).

    Returns mono signal.
    """
    symbols = [c for c in pattern if c in "Dt."]
    if not symbols:
        raise ValueError(f"Pattern {pattern!r} has no valid symbols.")

    # Each symbol = 1/8 note at `bpm`
    eighth_sec = 60.0 / bpm / 2
    pattern_sec = len(symbols) * eighth_sec

    if duration_sec is None:
        total_sec = pattern_sec
    else:
        total_sec = duration_sec

    n_total = int(total_sec * sample_rate)
    output = np.zeros(n_total)
    hit_dur = eighth_sec * 1.2  # let hits overlap slightly into next slot

    pos = 0.0
    idx = 0
    while pos < total_sec:
        sym = symbols[idx % len(symbols)]
        idx += 1
        start = int(pos * sample_rate)
        pos += eighth_sec
        if sym == ".":
            continue
        hit_type = "dum" if sym == "D" else "tek"
        hit = derbouka_hit(hit_type, duration_sec=hit_dur,
                           amplitude=0.8,
                           seed=(seed + idx * 101) if seed is not None else None,
                           sample_rate=sample_rate)
        end = min(start + len(hit), n_total)
        output[start:end] += hit[:end - start]

    return _normalize_peak(output, amplitude)
