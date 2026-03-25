"""Drum synthesis -- synthesized percussion from pure math.

Kick, snare, hihat, clap, tom, cymbal, and pre-configured drum kits.
All functions return np.ndarray (mono), consistent with generators.py.

Techniques used:
- Kick: sine pitch sweep (exponential decay from pitch_start to pitch_end)
  using instantaneous phase integration, plus noise click transient.
- Snare: sine body + bandpass-filtered noise + optional ring harmonics.
- Hihat: highpass-filtered noise with amplitude envelope controlling openness.
- Clap: multiple noise bursts with bandpass filtering.
- Tom: pitched sine sweep (shorter than kick) with body resonance.
- Cymbal: layered filtered noise with slow decay.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE
from fractal.generators import _time_axis


# ---------------------------------------------------------------------------
# Kick
# ---------------------------------------------------------------------------

def kick(
    pitch_start: float = 150.0,
    pitch_end: float = 40.0,
    duration_sec: float = 0.3,
    click_amount: float = 0.3,
    drive: float = 1.0,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a kick drum.

    Uses an exponentially decaying pitch sweep (sine oscillator) for the body,
    plus a short noise burst for the attack click. Optional soft-clip drive
    for saturation.

    Args:
        pitch_start: Starting frequency of the pitch sweep in Hz.
        pitch_end: Ending frequency in Hz.
        duration_sec: Duration in seconds.
        click_amount: Amount of noise click transient (0.0 to 1.0).
        drive: Saturation amount (1.0 = clean, 3.0+ = heavy).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n = int(sample_rate * duration_sec)
    t = np.arange(n) / sample_rate

    # Exponential pitch sweep: freq(t) = end + (start - end) * exp(-decay * t)
    # Decay constant chosen so pitch reaches ~95% of end by 60% of duration
    decay = 12.0 / duration_sec
    freq_curve = pitch_end + (pitch_start - pitch_end) * np.exp(-decay * t)

    # Instantaneous phase via integration of frequency curve
    phase = 2 * np.pi * np.cumsum(freq_curve) / sample_rate
    body = np.sin(phase)

    # Amplitude envelope: fast attack, exponential decay
    body_env = np.exp(-5.0 * t / duration_sec)
    body = body * body_env

    # Click transient: short noise burst
    click = np.zeros(n)
    if click_amount > 0:
        click_samples = min(int(0.005 * sample_rate), n)  # 5ms click
        click[:click_samples] = np.random.default_rng(42).uniform(
            -1.0, 1.0, click_samples
        )
        click_env = np.zeros(n)
        click_env[:click_samples] = np.exp(-np.linspace(0, 8, click_samples))
        click = click * click_env * click_amount

    signal = body + click

    # Soft-clip drive (tanh saturation)
    if drive > 1.0:
        signal = np.tanh(drive * signal) / np.tanh(drive)

    # Normalize to target amplitude
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Snare
# ---------------------------------------------------------------------------

def snare(
    tone_hz: float = 200.0,
    noise_amount: float = 0.6,
    duration_sec: float = 0.2,
    ring: float = 0.3,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a snare drum.

    Combines a sine body (pitched component) with bandpass-filtered noise
    (the snare wires). Optional ring harmonics add metallic character.

    Args:
        tone_hz: Body tone frequency in Hz.
        noise_amount: Mix of noise vs tone (0.0 = pure tone, 1.0 = pure noise).
        duration_sec: Duration in seconds.
        ring: Amount of harmonic ring (0.0 to 1.0).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n = int(sample_rate * duration_sec)
    t = np.arange(n) / sample_rate

    # Body: sine with fast decay
    body_env = np.exp(-20.0 * t / duration_sec)
    body = np.sin(2 * np.pi * tone_hz * t) * body_env

    # Ring harmonics (add metallic character)
    if ring > 0:
        ring_env = np.exp(-15.0 * t / duration_sec)
        body += ring * np.sin(2 * np.pi * tone_hz * 1.5 * t) * ring_env
        body += ring * 0.5 * np.sin(2 * np.pi * tone_hz * 2.2 * t) * ring_env

    # Noise: bandpass-filtered white noise (snare wires)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, n)

    # Bandpass filter for snare character (1kHz - 8kHz)
    nyquist = sample_rate / 2
    low = min(1000 / nyquist, 0.99)
    high = min(8000 / nyquist, 0.99)
    if low < high:
        sos = butter(2, [low, high], btype="band", output="sos")
        noise = sosfilt(sos, noise)

    noise_env = np.exp(-10.0 * t / duration_sec)
    noise = noise * noise_env

    # Mix body and noise
    tone_amount = 1.0 - noise_amount
    signal = tone_amount * body + noise_amount * noise

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Hi-hat
# ---------------------------------------------------------------------------

def hihat(
    duration_sec: float = 0.05,
    openness: float = 0.0,
    tone_hz: float = 8000.0,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a hi-hat.

    Highpass-filtered noise with amplitude envelope. Openness controls
    the decay time: 0.0 = closed (tight), 1.0 = open (sustained).

    Args:
        duration_sec: Duration in seconds.
        openness: How open the hi-hat is (0.0 = closed, 1.0 = open).
        tone_hz: Center frequency for the highpass filter.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n = int(sample_rate * duration_sec)
    t = np.arange(n) / sample_rate

    # Noise source
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, n)

    # Highpass filter
    nyquist = sample_rate / 2
    cutoff = min(tone_hz / nyquist, 0.99)
    cutoff = max(cutoff, 0.01)
    sos = butter(3, cutoff, btype="high", output="sos")
    signal = sosfilt(sos, noise)

    # Envelope: decay speed depends on openness
    # Closed = fast decay (high constant), open = slow decay (low constant)
    decay_rate = 40.0 - 35.0 * np.clip(openness, 0.0, 1.0)  # 40 (closed) to 5 (open)
    env = np.exp(-decay_rate * t / duration_sec)
    signal = signal * env

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Clap
# ---------------------------------------------------------------------------

def clap(
    duration_sec: float = 0.15,
    spread: float = 0.5,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a handclap.

    Multiple short noise bursts with bandpass filtering, simulating
    the layered character of a real handclap.

    Args:
        duration_sec: Duration in seconds.
        spread: Timing spread of the noise bursts (0.0 = tight, 1.0 = loose).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n = int(sample_rate * duration_sec)

    # Generate noise
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, n)

    # Bandpass filter (800Hz - 3500Hz) for clap character
    nyquist = sample_rate / 2
    low = min(800 / nyquist, 0.99)
    high = min(3500 / nyquist, 0.99)
    if low < high:
        sos = butter(2, [low, high], btype="band", output="sos")
        noise = sosfilt(sos, noise)

    # Multiple bursts envelope: 3-4 short hits then a tail
    env = np.zeros(n)
    burst_duration = int(0.003 * sample_rate)  # 3ms per burst
    gap = int(0.005 * sample_rate * (0.5 + spread))  # gap depends on spread
    n_bursts = 4

    pos = 0
    for i in range(n_bursts):
        end = min(pos + burst_duration, n)
        if pos < n:
            burst_len = end - pos
            env[pos:end] = np.exp(-np.linspace(0, 3, burst_len))
        pos += burst_duration + gap

    # Add decay tail after last burst
    tail_start = min(pos, n)
    if tail_start < n:
        tail_len = n - tail_start
        env[tail_start:] = 0.6 * np.exp(-np.linspace(0, 8, tail_len))

    signal = noise * env

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Tom
# ---------------------------------------------------------------------------

def tom(
    pitch_hz: float = 100.0,
    duration_sec: float = 0.25,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a tom drum.

    Similar to kick but with less pitch sweep and more body resonance.

    Args:
        pitch_hz: Fundamental pitch in Hz.
        duration_sec: Duration in seconds.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n = int(sample_rate * duration_sec)
    t = np.arange(n) / sample_rate

    # Mild pitch sweep: starts slightly above target, settles quickly
    decay = 20.0 / duration_sec
    freq_curve = pitch_hz + pitch_hz * 0.3 * np.exp(-decay * t)

    # Instantaneous phase
    phase = 2 * np.pi * np.cumsum(freq_curve) / sample_rate
    body = np.sin(phase)

    # Add second harmonic for body
    body += 0.3 * np.sin(2 * phase)

    # Amplitude envelope
    env = np.exp(-6.0 * t / duration_sec)
    signal = body * env

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Cymbal
# ---------------------------------------------------------------------------

def cymbal(
    duration_sec: float = 1.0,
    brightness: float = 0.5,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a cymbal.

    Layered filtered noise with slow decay. Brightness controls the
    highpass cutoff frequency.

    Args:
        duration_sec: Duration in seconds.
        brightness: Brightness of the cymbal (0.0 = dark, 1.0 = bright).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    n = int(sample_rate * duration_sec)
    t = np.arange(n) / sample_rate

    # Noise source
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, n)

    # Highpass filter: brightness controls cutoff (2kHz to 10kHz)
    brightness = np.clip(brightness, 0.0, 1.0)
    cutoff_hz = 2000 + 8000 * brightness
    nyquist = sample_rate / 2
    cutoff = min(cutoff_hz / nyquist, 0.99)
    cutoff = max(cutoff, 0.01)
    sos = butter(3, cutoff, btype="high", output="sos")
    signal = sosfilt(sos, noise)

    # Add some metallic tones (inharmonic sine components)
    signal += 0.15 * np.sin(2 * np.pi * 3500 * t)
    signal += 0.1 * np.sin(2 * np.pi * 5200 * t)
    signal += 0.08 * np.sin(2 * np.pi * 7300 * t)

    # Slow exponential decay with a faster initial transient
    env = 0.7 * np.exp(-3.0 * t / duration_sec) + 0.3 * np.exp(-0.8 * t / duration_sec)
    signal = signal * env

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = amplitude * signal / peak

    return signal


# ---------------------------------------------------------------------------
# Drum Kits
# ---------------------------------------------------------------------------

def drum_kit(
    style: str = "808",
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, np.ndarray]:
    """Return a pre-configured drum kit as a dict of named samples.

    Each kit contains: kick, snare, hihat_closed, hihat_open, clap, tom_low,
    tom_mid, tom_high, cymbal.

    Args:
        style: Kit style. Available: "808", "909", "acoustic", "lo-fi", "industrial".
        amplitude: Peak amplitude for all sounds.
        sample_rate: Sample rate in Hz.

    Returns:
        Dict mapping drum names to mono signals.
    """
    configs = _KIT_CONFIGS.get(style)
    if configs is None:
        raise ValueError(f"Unknown drum kit style: '{style}'. "
                         f"Available: {list(_KIT_CONFIGS.keys())}")

    kit = {}
    for name, (fn, kwargs) in configs.items():
        kit[name] = fn(amplitude=amplitude, sample_rate=sample_rate, **kwargs)

    return kit


# Kit configurations: style -> {name: (function, kwargs)}
_KIT_CONFIGS: dict[str, dict[str, tuple]] = {
    "808": {
        "kick":        (kick,   {"pitch_start": 180, "pitch_end": 35, "duration_sec": 0.4, "click_amount": 0.1, "drive": 1.5}),
        "snare":       (snare,  {"tone_hz": 180, "noise_amount": 0.5, "duration_sec": 0.25, "ring": 0.2}),
        "hihat_closed":(hihat,  {"duration_sec": 0.05, "openness": 0.0, "tone_hz": 8000}),
        "hihat_open":  (hihat,  {"duration_sec": 0.3, "openness": 0.8, "tone_hz": 8000}),
        "clap":        (clap,   {"duration_sec": 0.15, "spread": 0.4}),
        "tom_low":     (tom,    {"pitch_hz": 80, "duration_sec": 0.3}),
        "tom_mid":     (tom,    {"pitch_hz": 120, "duration_sec": 0.25}),
        "tom_high":    (tom,    {"pitch_hz": 180, "duration_sec": 0.2}),
        "cymbal":      (cymbal, {"duration_sec": 1.0, "brightness": 0.5}),
    },
    "909": {
        "kick":        (kick,   {"pitch_start": 140, "pitch_end": 45, "duration_sec": 0.25, "click_amount": 0.5, "drive": 1.2}),
        "snare":       (snare,  {"tone_hz": 240, "noise_amount": 0.65, "duration_sec": 0.18, "ring": 0.4}),
        "hihat_closed":(hihat,  {"duration_sec": 0.04, "openness": 0.0, "tone_hz": 10000}),
        "hihat_open":  (hihat,  {"duration_sec": 0.25, "openness": 0.75, "tone_hz": 10000}),
        "clap":        (clap,   {"duration_sec": 0.12, "spread": 0.5}),
        "tom_low":     (tom,    {"pitch_hz": 100, "duration_sec": 0.25}),
        "tom_mid":     (tom,    {"pitch_hz": 160, "duration_sec": 0.2}),
        "tom_high":    (tom,    {"pitch_hz": 220, "duration_sec": 0.15}),
        "cymbal":      (cymbal, {"duration_sec": 1.2, "brightness": 0.6}),
    },
    "acoustic": {
        "kick":        (kick,   {"pitch_start": 120, "pitch_end": 50, "duration_sec": 0.25, "click_amount": 0.5, "drive": 1.0}),
        "snare":       (snare,  {"tone_hz": 220, "noise_amount": 0.7, "duration_sec": 0.15, "ring": 0.5}),
        "hihat_closed":(hihat,  {"duration_sec": 0.04, "openness": 0.0, "tone_hz": 9000}),
        "hihat_open":  (hihat,  {"duration_sec": 0.25, "openness": 0.7, "tone_hz": 9000}),
        "clap":        (clap,   {"duration_sec": 0.12, "spread": 0.6}),
        "tom_low":     (tom,    {"pitch_hz": 90, "duration_sec": 0.3}),
        "tom_mid":     (tom,    {"pitch_hz": 140, "duration_sec": 0.25}),
        "tom_high":    (tom,    {"pitch_hz": 200, "duration_sec": 0.2}),
        "cymbal":      (cymbal, {"duration_sec": 1.5, "brightness": 0.4}),
    },
    "lo-fi": {
        "kick":        (kick,   {"pitch_start": 100, "pitch_end": 45, "duration_sec": 0.2, "click_amount": 0.2, "drive": 2.5}),
        "snare":       (snare,  {"tone_hz": 160, "noise_amount": 0.8, "duration_sec": 0.18, "ring": 0.1}),
        "hihat_closed":(hihat,  {"duration_sec": 0.03, "openness": 0.0, "tone_hz": 6000}),
        "hihat_open":  (hihat,  {"duration_sec": 0.2, "openness": 0.5, "tone_hz": 6000}),
        "clap":        (clap,   {"duration_sec": 0.1, "spread": 0.3}),
        "tom_low":     (tom,    {"pitch_hz": 70, "duration_sec": 0.25}),
        "tom_mid":     (tom,    {"pitch_hz": 110, "duration_sec": 0.2}),
        "tom_high":    (tom,    {"pitch_hz": 160, "duration_sec": 0.15}),
        "cymbal":      (cymbal, {"duration_sec": 0.8, "brightness": 0.3}),
    },
    "industrial": {
        "kick":        (kick,   {"pitch_start": 200, "pitch_end": 30, "duration_sec": 0.35, "click_amount": 0.6, "drive": 3.0}),
        "snare":       (snare,  {"tone_hz": 250, "noise_amount": 0.4, "duration_sec": 0.2, "ring": 0.7}),
        "hihat_closed":(hihat,  {"duration_sec": 0.04, "openness": 0.0, "tone_hz": 10000}),
        "hihat_open":  (hihat,  {"duration_sec": 0.35, "openness": 0.9, "tone_hz": 10000}),
        "clap":        (clap,   {"duration_sec": 0.2, "spread": 0.7}),
        "tom_low":     (tom,    {"pitch_hz": 60, "duration_sec": 0.35}),
        "tom_mid":     (tom,    {"pitch_hz": 100, "duration_sec": 0.3}),
        "tom_high":    (tom,    {"pitch_hz": 150, "duration_sec": 0.25}),
        "cymbal":      (cymbal, {"duration_sec": 1.5, "brightness": 0.7}),
    },
}
