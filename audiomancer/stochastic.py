"""Stochastic — random micro-event placement for organic ambient textures.

Scatters short texture bursts at random timestamps within a signal.
Each render with a different seed produces a unique arrangement.
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.envelope import ar as _ar_envelope
from audiomancer.textures import generate as _gen_texture
from audiomancer.utils import mono_to_stereo

# ---------------------------------------------------------------------------
# Core: scatter micro-events across a duration
# ---------------------------------------------------------------------------

def scatter_events(
    duration_sec: float,
    events: list[dict] | None = None,
    density: float = 0.5,
    seed: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Scatter short texture micro-events at random timestamps.

    Creates organic, non-repeating ambient layers by placing brief
    texture bursts (singing bowls, shimmers, etc.) at random positions.

    Args:
        duration_sec: Total duration in seconds.
        events: List of event dicts, each with:
            - "texture": texture name (e.g., "crystal_shimmer", "singing_bowl")
            - "duration": event duration in seconds (default 5.0)
            - "count": number of occurrences (default 3)
            - "volume_db": volume in dB (default -6.0)
            - "min_gap_sec": minimum gap between events (default 10.0)
            If None, uses a sensible default set.
        density: Overall density multiplier (0.0 = empty, 1.0 = normal, 2.0 = dense).
        seed: Random seed for reproducibility. None = unique each time.
        sample_rate: Sample rate.

    Returns:
        Stereo signal with scattered events.
    """
    if events is None:
        events = DEFAULT_EVENTS

    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration_sec)
    result = np.zeros((n_samples, 2), dtype=np.float64)

    for evt in events:
        tex_name = evt.get("texture", "crystal_shimmer")
        evt_dur = evt.get("duration", 5.0)
        count = max(1, int(evt.get("count", 3) * density))
        vol_db = evt.get("volume_db", -6.0)
        min_gap = evt.get("min_gap_sec", 10.0)

        gain = 10 ** (vol_db / 20)

        # Generate event positions with minimum gap constraint
        positions = _place_events(duration_sec, evt_dur, count, min_gap, rng)

        for pos in positions:
            # Generate the texture clip
            evt_seed = rng.integers(0, 100000) if seed is not None else None
            clip = _gen_texture(tex_name, evt_dur, seed=evt_seed,
                                sample_rate=sample_rate)
            if clip.ndim == 1:
                clip = mono_to_stereo(clip)

            # Apply AR envelope (fade in/out)
            env = _ar_envelope(evt_dur, attack=0.3, curve=2.0,
                               sample_rate=sample_rate)
            clip = clip * env[:, np.newaxis]
            clip = clip * gain

            # Place in output
            start = int(pos * sample_rate)
            end = min(start + clip.shape[0], n_samples)
            clip_len = end - start
            if clip_len > 0:
                result[start:end] += clip[:clip_len]

    return result


def _place_events(
    duration_sec: float,
    event_dur: float,
    count: int,
    min_gap: float,
    rng: np.random.Generator,
) -> list[float]:
    """Place events with minimum gap constraint."""
    max_start = duration_sec - event_dur
    if max_start <= 0:
        return [0.0]

    positions = []
    attempts = 0
    max_attempts = count * 20

    while len(positions) < count and attempts < max_attempts:
        pos = rng.uniform(0, max_start)
        # Check minimum gap to all existing positions
        if all(abs(pos - p) >= min_gap for p in positions):
            positions.append(pos)
        attempts += 1

    return sorted(positions)


# ---------------------------------------------------------------------------
# Default event set — ambient meditation
# ---------------------------------------------------------------------------

DEFAULT_EVENTS = [
    {
        "texture": "crystal_shimmer",
        "duration": 4.0,
        "count": 3,
        "volume_db": -10.0,
        "min_gap_sec": 30.0,
    },
    {
        "texture": "singing_bowl",
        "duration": 6.0,
        "count": 2,
        "volume_db": -12.0,
        "min_gap_sec": 45.0,
    },
    {
        "texture": "ethereal_wash",
        "duration": 8.0,
        "count": 2,
        "volume_db": -14.0,
        "min_gap_sec": 40.0,
    },
]


# ---------------------------------------------------------------------------
# Phase D4 — typed micro-events (harmonic_bloom, grain_burst, overtone_whisper)
# ---------------------------------------------------------------------------

def _make_harmonic_bloom(chord_freqs: list[float], event_dur: float,
                         volume_db: float, rng: np.random.Generator,
                         sample_rate: int) -> np.ndarray:
    """A chord harmonic emerges then fades — 3-8s raised-cosine."""
    freq = float(rng.choice(chord_freqs)) * rng.choice([2.0, 3.0, 4.0])
    n = int(event_dur * sample_rate)
    t = np.linspace(0, event_dur, n, endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    env = 0.5 * (1 - np.cos(2 * np.pi * t / event_dur))  # raised cosine
    gain = 10 ** (volume_db / 20)
    mono = tone * env * gain
    return mono_to_stereo(mono)


def _make_grain_burst(source: np.ndarray, event_dur: float,
                      volume_db: float, rng: np.random.Generator,
                      sample_rate: int) -> np.ndarray:
    """Scatter 3-5 small grains from a source buffer over event_dur."""
    if source.ndim == 2:
        source = source[:, 0]
    n = int(event_dur * sample_rate)
    out = np.zeros(n)
    grain_size = int(rng.uniform(0.05, 0.12) * sample_rate)
    n_grains = int(rng.integers(3, 6))
    src_len = len(source)
    for _ in range(n_grains):
        if src_len <= grain_size:
            continue
        src_start = int(rng.integers(0, src_len - grain_size))
        grain = source[src_start:src_start + grain_size]
        window = np.hanning(grain_size)
        grain = grain * window
        pos = int(rng.uniform(0, max(1, n - grain_size)))
        out[pos:pos + grain_size] += grain
    gain = 10 ** (volume_db / 20)
    return mono_to_stereo(out * gain)


def _make_overtone_whisper(chord_freqs: list[float], event_dur: float,
                           volume_db: float, rng: np.random.Generator,
                           sample_rate: int) -> np.ndarray:
    """Pure sine at a high overtone with slow swell, very quiet."""
    base = float(rng.choice(chord_freqs))
    overtone = base * rng.choice([5.0, 7.0, 9.0])
    n = int(event_dur * sample_rate)
    t = np.linspace(0, event_dur, n, endpoint=False)
    tone = np.sin(2 * np.pi * overtone * t)
    env = np.sin(np.pi * t / event_dur) ** 2  # smooth bell
    gain = 10 ** (volume_db / 20)
    return mono_to_stereo(tone * env * gain)


def _make_micro_silence(event_dur: float, sample_rate: int,
                        duck_db: float = -12.0) -> np.ndarray:
    """Multiplicative duck envelope (NOT additive).

    Returns a SUBTRACTIVE gain reduction value centered on 0. Caller must
    detect this by checking that `event_spec["type"] == "micro_silence"` and
    apply via multiplication on the pre-mix stem.
    """
    n = int(event_dur * sample_rate)
    t = np.linspace(0, event_dur, n, endpoint=False)
    # Raised-cosine dip
    dip = 0.5 * (1 - np.cos(2 * np.pi * t / event_dur))
    attenuation = 10 ** (duck_db / 20)
    # Return value in [attenuation, 1.0], shape (n, 1) broadcastable
    gain = 1.0 - (1.0 - attenuation) * dip
    return np.column_stack([gain, gain])


def micro_events(
    duration_sec: float,
    event_specs: list[dict],
    chord_freqs: list[float] | None = None,
    source: np.ndarray | None = None,
    seed: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Scatter typed micro-events over duration — the anti-static layer.

    Each event_spec dict:
        - "type": "harmonic_bloom" | "grain_burst" | "overtone_whisper"
                  ("micro_silence" handled separately — returns ducking env)
        - "rate_per_min": expected events per minute
        - "volume_db": event level
        - "duration_range": (min_sec, max_sec) per event (default per-type)

    Args:
        duration_sec: Stem duration.
        event_specs: List of event specs.
        chord_freqs: Required for harmonic_bloom / overtone_whisper.
        source: Required for grain_burst (mono buffer).
        seed: Random seed.
        sample_rate: Sample rate.

    Returns:
        Stereo signal to ADD to the mix. Does not include micro_silence ducks.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sample_rate)
    result = np.zeros((n_samples, 2), dtype=np.float64)

    for spec in event_specs:
        evt_type = spec["type"]
        if evt_type == "micro_silence":
            continue  # silence is multiplicative, not additive

        rate_per_min = spec.get("rate_per_min", 1.0)
        volume_db = spec.get("volume_db", -24.0)
        dur_range = spec.get("duration_range", (3.0, 6.0))

        n_events = int(rate_per_min * duration_sec / 60.0)
        if n_events <= 0:
            continue

        min_gap = duration_sec / (n_events + 1) * 0.5
        positions = _place_events(duration_sec, dur_range[1], n_events,
                                  min_gap, rng)

        for pos in positions:
            evt_dur = rng.uniform(*dur_range)

            if evt_type == "harmonic_bloom":
                if chord_freqs is None:
                    raise ValueError("harmonic_bloom needs chord_freqs")
                clip = _make_harmonic_bloom(chord_freqs, evt_dur, volume_db,
                                            rng, sample_rate)
            elif evt_type == "grain_burst":
                if source is None:
                    raise ValueError("grain_burst needs source buffer")
                clip = _make_grain_burst(source, evt_dur, volume_db,
                                         rng, sample_rate)
            elif evt_type == "overtone_whisper":
                if chord_freqs is None:
                    raise ValueError("overtone_whisper needs chord_freqs")
                clip = _make_overtone_whisper(chord_freqs, evt_dur, volume_db,
                                              rng, sample_rate)
            else:
                raise ValueError(f"Unknown event type: {evt_type!r}")

            start = int(pos * sample_rate)
            end = min(start + clip.shape[0], n_samples)
            chunk = end - start
            if chunk > 0:
                result[start:end] += clip[:chunk]

    return result


def micro_silence_env(
    duration_sec: float,
    rate_per_min: float = 0.2,
    duck_db: float = -12.0,
    event_dur_range: tuple[float, float] = (0.5, 1.0),
    seed: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Multiplicative envelope with random subtle ducks.

    Apply on the mix BEFORE LUFS normalize: `stem = stem * env[:, np.newaxis]`.
    Creates tiny "breath pauses" that the conscious ear barely catches.

    Returns:
        Envelope stereo, shape (n, 2), values in [~0.25, 1.0].
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sample_rate)
    env = np.ones((n_samples, 2), dtype=np.float64)

    n_events = int(rate_per_min * duration_sec / 60.0)
    if n_events <= 0:
        return env

    positions = _place_events(duration_sec, event_dur_range[1], n_events,
                              event_dur_range[1] * 3, rng)

    for pos in positions:
        evt_dur = rng.uniform(*event_dur_range)
        duck = _make_micro_silence(evt_dur, sample_rate, duck_db)
        start = int(pos * sample_rate)
        end = min(start + duck.shape[0], n_samples)
        chunk = end - start
        if chunk > 0:
            env[start:end] *= duck[:chunk]
    return env
