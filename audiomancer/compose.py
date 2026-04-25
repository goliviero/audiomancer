"""compose — outils de composition temporelle pour stems progressifs.

Ajoute la dimension manquante : progression logique sur la durée.
Permet de créer des stems qui évoluent (swells, sections, filtres) et
qui se bouclent sans click.

Usage:
    from audiomancer.compose import fade_envelope, tremolo, stitch, make_loopable

    # Enveloppe de volume : drone qui entre progressivement
    env = fade_envelope([(0, 0.1), (60, 1.0), (240, 1.0), (300, 0.1)], 300)
    drone_sig = apply_amplitude_mod(drone_sig, env)

    # Tremolo lent sur le drone
    drone_sig = tremolo(drone_sig, rate_hz=0.15, depth=0.05)

    # Coller 4 sections avec crossfade
    full = stitch([section_a, section_b, section_c, section_d], crossfade_sec=5.0)

    # Rendre le résultat seamless pour ffmpeg
    loopable = make_loopable(full, crossfade_sec=5.0)

    # Verify loop quality
    score, report = verify_loop(loopable, crossfade_sec=5.0)
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.layers import crossfade as _crossfade
from audiomancer.modulation import apply_amplitude_mod, evolving_lfo

# ---------------------------------------------------------------------------
# Breakpoint envelopes
# ---------------------------------------------------------------------------

def fade_envelope(waypoints: list[tuple[float, float]],
                  duration_sec: float,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a smooth amplitude envelope from (time, value) breakpoints.

    Linear interpolation between waypoints. Useful for:
    - Volume automation per layer (drone swell, pad fade in/out)
    - Filter cutoff curves (pass to apply_filter_sweep)

    Args:
        waypoints: List of (time_sec, value) pairs in chronological order.
                   First point should be at t=0, last at t=duration_sec.
        duration_sec: Total duration to generate.
        sample_rate: Sample rate.

    Returns:
        Envelope signal (n_samples,). Monotonically interpolated between waypoints.

    Examples:
        # Drone swell: quiet → full → quiet (loop-friendly)
        env = fade_envelope([(0, 0.1), (60, 1.0), (240, 1.0), (300, 0.1)], 300)

        # Filter sweep: closed → open → closed
        cutoffs = fade_envelope([(0, 800), (90, 2500), (210, 2500), (300, 800)], 300)
        sig = apply_filter_sweep(sig, cutoffs)
    """
    n = int(sample_rate * duration_sec)
    times = np.array([t for t, _ in waypoints])
    values = np.array([v for _, v in waypoints])
    t_samples = np.linspace(0, duration_sec, n, endpoint=False)
    return np.interp(t_samples, times, values)


# ---------------------------------------------------------------------------
# Tremolo (slow rhythmic pulsation)
# ---------------------------------------------------------------------------

def tremolo(signal: np.ndarray,
            rate_hz: float = 0.15,
            depth: float = 0.05,
            seed: int | None = None,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply slow evolving tremolo to a signal.

    Uses evolving_lfo internally so the rate drifts slightly — never exactly
    repeating. At depth=0.05 (5%), the pulsation is felt rather than heard.

    For ambient use:
        rate_hz=0.10  → one swell every ~10s (very slow, glacial)
        rate_hz=0.15  → one swell every ~7s  (gentle breathing)
        rate_hz=0.25  → one swell every ~4s  (more active, still subtle)

    Args:
        signal: Input signal (mono or stereo).
        rate_hz: Base LFO frequency in Hz.
        depth: Amplitude variation depth (0.05 = ±5%). Keep ≤ 0.10 for ambient.
        seed: Random seed for reproducibility.
        sample_rate: Sample rate.

    Returns:
        Signal with tremolo applied.
    """
    duration_sec = signal.shape[0] / sample_rate
    mod = evolving_lfo(duration_sec, rate_hz=rate_hz,
                       depth=depth, offset=1.0,
                       drift_speed=0.03, seed=seed,
                       sample_rate=sample_rate)
    return apply_amplitude_mod(signal, mod)


# ---------------------------------------------------------------------------
# Section stitching
# ---------------------------------------------------------------------------

def stitch(sections: list[np.ndarray],
           crossfade_sec: float = 5.0,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Concatenate sections with smooth crossfades between each pair.

    Args:
        sections: List of audio signals (must all be same type: mono or stereo).
                  Each section should be longer than crossfade_sec.
        crossfade_sec: Crossfade duration between sections.
        sample_rate: Sample rate.

    Returns:
        Single continuous signal.

    Raises:
        ValueError: If sections list is empty.
    """
    if not sections:
        raise ValueError("sections list is empty")
    if len(sections) == 1:
        return sections[0]

    result = sections[0]
    for next_section in sections[1:]:
        result = _crossfade(result, next_section,
                            crossfade_sec=crossfade_sec,
                            sample_rate=sample_rate)
    return result


# ---------------------------------------------------------------------------
# Loop sealing
# ---------------------------------------------------------------------------

def make_loopable(stem: np.ndarray,
                  crossfade_sec: float = 5.0,
                  sample_rate: int = SAMPLE_RATE,
                  boundary_continuity: bool = False) -> np.ndarray:
    """Seal the loop point: crossfade the end back onto the beginning.

    When ffmpeg loops a WAV with -stream_loop -1, it jumps from the last
    sample back to the first. This crossfade ensures the jump is inaudible.

    Requirements: The stem's start and end should already be at comparable
    amplitude levels (both near-silent, or both at the same volume).
    If start and end are at very different amplitudes, the crossfade will
    still reduce the click but can't fully mask a large jump.

    Args:
        stem: Full stem signal (should be at least 2× crossfade_sec long).
        crossfade_sec: Duration of the crossfade overlap at the loop point.
        sample_rate: Sample rate.
        boundary_continuity: If True, apply a global DC tilt so result[-1]
            exactly matches result[0]. Useful for aperiodic content like
            pure noise where the 5s crossfade alone doesn't guarantee a
            sample-clean ffmpeg -stream_loop join. Per-sample correction
            is delta/N (~1e-8 for typical noise), inaudible. Default off
            to preserve exact-equality behavior for musical content.

    Returns:
        Same-length stem with a smooth loop point baked in.
    """
    xf_samples = int(sample_rate * crossfade_sec)
    xf_samples = min(xf_samples, stem.shape[0] // 2)

    fade_out_ramp = np.linspace(1.0, 0.0, xf_samples)
    fade_in_ramp = np.linspace(0.0, 1.0, xf_samples)

    if stem.ndim == 2:
        fade_out_ramp = fade_out_ramp[:, np.newaxis]
        fade_in_ramp = fade_in_ramp[:, np.newaxis]

    # Blend the tail with the head
    tail = stem[-xf_samples:] * fade_out_ramp
    head = stem[:xf_samples] * fade_in_ramp
    blended = tail + head

    # Replace the tail with the blend
    result = stem.copy()
    result[-xf_samples:] = blended

    if boundary_continuity:
        boundary_delta = result[-1] - result[0]
        if np.any(np.abs(boundary_delta) > 1e-6):
            n = result.shape[0]
            ramp = np.linspace(0.0, 1.0, n)
            if result.ndim == 2:
                result = result - ramp[:, None] * boundary_delta
            else:
                result = result - ramp * boundary_delta

    return result


# ---------------------------------------------------------------------------
# Edge pre-fade (safety fade at file boundaries)
# ---------------------------------------------------------------------------

def apply_pre_fade(signal: np.ndarray, fade_sec: float = 1.5,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Linear fade-in and fade-out at the signal edges.

    Applied after ``make_loopable`` as a safety pass: silences the very first
    and last samples so a non-looped playback (e.g. file preview) doesn't
    click. With typical fade_sec ≤ 2 and make_loopable crossfade ≥ 5s,
    the loop continuity inside the crossfade region is preserved.

    Args:
        signal: Input signal (mono or stereo).
        fade_sec: Duration of fade-in / fade-out at each edge.
        sample_rate: Sample rate.

    Returns:
        Same-length signal with linear fades at both edges.
    """
    n_fade = int(fade_sec * sample_rate)
    if n_fade <= 0 or signal.shape[0] < 2 * n_fade:
        return signal

    ramp_in = np.linspace(0.0, 1.0, n_fade)
    ramp_out = np.linspace(1.0, 0.0, n_fade)
    out = signal.copy()
    if out.ndim == 2:
        out[:n_fade] *= ramp_in[:, np.newaxis]
        out[-n_fade:] *= ramp_out[:, np.newaxis]
    else:
        out[:n_fade] *= ramp_in
        out[-n_fade:] *= ramp_out
    return out


# ---------------------------------------------------------------------------
# Loop verification
# ---------------------------------------------------------------------------

def verify_loop(stem: np.ndarray, crossfade_sec: float = 5.0,
                sample_rate: int = SAMPLE_RATE) -> tuple[float, dict]:
    """Verify loop quality at the junction point.

    Checks for discontinuities, level mismatch, and spectral similarity
    between the end and start of the stem.

    Args:
        stem: Loopable stem (output of make_loopable).
        crossfade_sec: Duration of the crossfade zone to analyze.
        sample_rate: Sample rate.

    Returns:
        Tuple of (score, report):
            score: 0.0 (terrible) to 1.0 (perfect loop).
            report: Dict with individual metrics.
    """
    xf_samples = int(sample_rate * crossfade_sec)
    xf_samples = min(xf_samples, stem.shape[0] // 4)

    # Work in mono for analysis
    if stem.ndim == 2:
        mono = np.mean(stem, axis=1)
    else:
        mono = stem

    head = mono[:xf_samples]
    tail = mono[-xf_samples:]

    # 1. Level difference at junction (should be near 0)
    head_rms = np.sqrt(np.mean(head ** 2))
    tail_rms = np.sqrt(np.mean(tail ** 2))
    if head_rms > 0 and tail_rms > 0:
        level_diff_db = abs(20 * np.log10(tail_rms / head_rms))
    else:
        level_diff_db = 0.0

    # 2. Sample discontinuity at exact loop point (last sample → first sample)
    jump = abs(mono[-1] - mono[0])
    jump_score = max(0.0, 1.0 - jump * 100)  # penalize jumps > 0.01

    # 3. Cross-correlation between head and tail (spectral similarity)
    head_norm = head - np.mean(head)
    tail_norm = tail - np.mean(tail)
    h_std = np.std(head_norm)
    t_std = np.std(tail_norm)
    if h_std > 0 and t_std > 0:
        correlation = np.corrcoef(head_norm, tail_norm)[0, 1]
    else:
        correlation = 1.0

    # 4. Envelope continuity (RMS in short windows at boundary)
    window_ms = 50
    window_samples = int(sample_rate * window_ms / 1000)
    tail_end_rms = np.sqrt(np.mean(mono[-window_samples:] ** 2))
    head_start_rms = np.sqrt(np.mean(mono[:window_samples] ** 2))
    if tail_end_rms > 0 and head_start_rms > 0:
        boundary_diff_db = abs(20 * np.log10(tail_end_rms / head_start_rms))
    else:
        boundary_diff_db = 0.0

    # Composite score (weighted)
    level_score = max(0.0, 1.0 - level_diff_db / 6.0)    # 6 dB = 0 score
    corr_score = max(0.0, (correlation + 1) / 2)           # -1..1 → 0..1
    boundary_score = max(0.0, 1.0 - boundary_diff_db / 3.0)

    score = (
        0.25 * level_score
        + 0.25 * jump_score
        + 0.25 * corr_score
        + 0.25 * boundary_score
    )

    report = {
        "level_diff_db": round(level_diff_db, 2),
        "jump_amplitude": round(jump, 6),
        "correlation": round(correlation, 4),
        "boundary_diff_db": round(boundary_diff_db, 2),
        "level_score": round(level_score, 3),
        "jump_score": round(jump_score, 3),
        "corr_score": round(corr_score, 3),
        "boundary_score": round(boundary_score, 3),
        "overall": round(score, 3),
    }

    return score, report


# ---------------------------------------------------------------------------
# Phase D — long-term density profile (5 min story arc)
# ---------------------------------------------------------------------------

def density_profile(duration_sec: float, profile: str = "flat",
                    seed: int | None = None,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Long-term amplitude envelope — defines how "busy" a stem feels over time.

    Applied once at the stem level (before LUFS normalization) to give a
    5-min render a sense of direction without needing per-layer orchestration.

    Profiles:
        - "flat": constant 1.0 (legacy, no envelope)
        - "breathing": one raised-cosine half-cycle (0.3 -> 1.0 -> 0.3)
        - "arc": smooth rise to center, fade out (0.4 -> 1.0 -> 0.4, symmetric)
        - "random_walk": Ornstein-Uhlenbeck drift, non-periodic
        - "sparse": binary-ish random gating (sections on/off), smooth edges

    Args:
        duration_sec: Total duration.
        profile: One of the names above.
        seed: Random seed (only used by stochastic profiles).
        sample_rate: Sample rate.

    Returns:
        Envelope in [0, 1+], shape (n_samples,). Multiply your stem by it.
    """
    n = int(sample_rate * duration_sec)

    if profile == "flat":
        return np.ones(n)

    if profile == "breathing":
        # Raised cosine: 0.3 at edges, 1.0 at center
        t = np.linspace(0, 1, n, endpoint=False)
        env = 0.3 + 0.7 * 0.5 * (1 - np.cos(2 * np.pi * t))
        return env

    if profile == "arc":
        # Asymmetric smooth arc: 0.4 -> 1.0 -> 0.4 via half-sine envelope
        t = np.linspace(0, 1, n, endpoint=False)
        env = 0.4 + 0.6 * np.sin(np.pi * t)
        return env

    if profile == "random_walk":
        # Import here to avoid circular imports
        from audiomancer.modulation import random_walk as _rw
        return _rw(duration_sec, sigma=0.08, tau=duration_sec / 5,
                   seed=seed, sample_rate=sample_rate)

    if profile == "sparse":
        # Alternating sections 20-60s with smooth transitions
        rng = np.random.default_rng(seed)
        env = np.zeros(n)
        pos = 0
        state = 1.0
        while pos < n:
            section_sec = rng.uniform(20.0, 60.0)
            section_len = int(section_sec * sample_rate)
            end = min(pos + section_len, n)
            env[pos:end] = state
            pos = end
            state = 0.3 if state > 0.5 else 1.0
        # Smooth transitions with 2s crossfade
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, 0.5 / (sample_rate / 2), btype="low", output="sos")
        env = sosfiltfilt(sos, env)
        return np.clip(env, 0.0, 1.0)

    raise ValueError(
        f"Unknown density profile: {profile!r}. "
        "Valid: flat, breathing, arc, random_walk, sparse."
    )
