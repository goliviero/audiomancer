"""Stochastic — random micro-event placement for organic ambient textures.

Scatters short texture bursts at random timestamps within a signal.
Each render with a different seed produces a unique arrangement.
"""

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.textures import generate as _gen_texture
from audiomancer.envelope import ar as _ar_envelope
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
