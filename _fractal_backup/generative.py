"""Generative composition -- algorithmic music creation tools.

Random melodies, evolving parameters, ambient textures, and chord
progression renderers. All functions accept a `seed` parameter for
reproducibility: same seed = same output, always.
"""

import numpy as np

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE
from fractal.music_theory import scale_hz, chord_hz, progression_hz, note_to_hz
from fractal.presets import get_preset, SynthPreset
from fractal.modulation import LFO, apply_filter_sweep
from fractal.envelopes import SmoothFade, ADSR
from fractal.effects import Reverb, LowPassFilter, NormalizePeak, EffectChain
from fractal.signal import mix_signals, pad_to_length, silence


# ---------------------------------------------------------------------------
# Random Melody
# ---------------------------------------------------------------------------

def random_melody(
    scale_notes: list[float],
    n_notes: int = 8,
    note_duration: float = 0.5,
    rest_probability: float = 0.1,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """Generate a random melody from a set of scale frequencies.

    Returns a list of (frequency_hz, duration_sec) tuples. Rests are
    represented as (0.0, duration).

    Args:
        scale_notes: List of frequencies to choose from (e.g., from scale_hz()).
        n_notes: Number of notes to generate.
        note_duration: Duration of each note in seconds.
        rest_probability: Probability of a rest instead of a note (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        List of (frequency, duration) tuples.
    """
    rng = np.random.default_rng(seed)
    melody = []

    for _ in range(n_notes):
        if rng.random() < rest_probability:
            melody.append((0.0, note_duration))
        else:
            freq = rng.choice(scale_notes)
            melody.append((float(freq), note_duration))

    return melody


# ---------------------------------------------------------------------------
# Weighted Random Notes
# ---------------------------------------------------------------------------

def weighted_random_notes(
    scale_notes: list[float],
    weights: list[float] | None = None,
    n_notes: int = 8,
    seed: int | None = None,
) -> list[float]:
    """Pick notes from a scale with optional weighting.

    Default weights favor the root (index 0) and fifth (index 4 if exists).

    Args:
        scale_notes: List of frequencies.
        weights: Probability weights (same length as scale_notes).
            None uses default weighting favoring root and fifth.
        n_notes: Number of notes to pick.
        seed: Random seed.

    Returns:
        List of frequencies.
    """
    rng = np.random.default_rng(seed)

    if weights is None:
        # Default: favor root (1st) and fifth (5th degree)
        weights = np.ones(len(scale_notes))
        weights[0] = 3.0  # root
        if len(scale_notes) > 4:
            weights[4] = 2.0  # fifth

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    indices = rng.choice(len(scale_notes), size=n_notes, p=weights)
    return [float(scale_notes[i]) for i in indices]


# ---------------------------------------------------------------------------
# Evolving Parameter
# ---------------------------------------------------------------------------

def evolving_parameter(
    base_value: float,
    drift_range: float,
    duration_sec: float,
    speed: float = 0.1,
    smoothness: float = 0.95,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a slowly evolving parameter curve via filtered random walk.

    Useful for organic drift of cutoff, reverb mix, volume, etc.

    Args:
        base_value: Center value of the parameter.
        drift_range: Maximum deviation from base_value.
        duration_sec: Duration in seconds.
        speed: How fast the parameter changes (0.01 = glacial, 1.0 = fast).
        smoothness: Smoothing factor (0.0 = noisy, 0.99 = very smooth).
        sample_rate: Sample rate in Hz.
        seed: Random seed.

    Returns:
        1D array of parameter values.
    """
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng(seed)

    # Generate random walk
    steps = rng.standard_normal(n) * speed
    walk = np.cumsum(steps)

    # Apply smoothing (exponential moving average)
    smoothed = np.zeros(n)
    smoothed[0] = walk[0]
    for i in range(1, n):
        smoothed[i] = smoothness * smoothed[i - 1] + (1 - smoothness) * walk[i]

    # Normalize to [-1, 1] range
    peak = np.max(np.abs(smoothed))
    if peak > 0:
        smoothed = smoothed / peak

    return base_value + drift_range * smoothed


# ---------------------------------------------------------------------------
# Phrase Generator
# ---------------------------------------------------------------------------

def phrase_generator(
    scale_notes: list[float],
    preset: SynthPreset | str,
    tempo_bpm: float = 120.0,
    measures: int = 4,
    density: float = 0.5,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a musical phrase as rendered audio.

    Creates a rhythmic pattern of notes from the given scale, rendered
    with the specified preset.

    Args:
        scale_notes: List of frequencies to choose from.
        preset: SynthPreset instance or preset name string.
        tempo_bpm: Tempo in BPM.
        measures: Number of measures (4 beats each).
        density: Note density (0.0 = sparse, 1.0 = every beat).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.
        seed: Random seed.

    Returns:
        Mono signal.
    """
    if isinstance(preset, str):
        preset = get_preset(preset)

    rng = np.random.default_rng(seed)
    beat_sec = 60.0 / tempo_bpm
    total_beats = measures * 4
    total_samples = int(sample_rate * total_beats * beat_sec)
    output = np.zeros(total_samples, dtype=np.float64)

    # Decide which 8th-note positions get notes
    n_eighths = total_beats * 2
    for i in range(n_eighths):
        if rng.random() > density:
            continue

        pos_sec = i * beat_sec / 2
        pos_samples = int(pos_sec * sample_rate)

        # Pick a note
        freq = rng.choice(scale_notes)

        # Variable note duration (8th to quarter note)
        dur = beat_sec * rng.choice([0.5, 0.75, 1.0])
        note = preset.render(float(freq), dur, amplitude=amplitude,
                             sample_rate=sample_rate)

        # Place in output
        end = min(pos_samples + len(note), total_samples)
        length = end - pos_samples
        if length > 0:
            output[pos_samples:end] += note[:length]

    # Clip to prevent overs from overlapping notes
    peak = np.max(np.abs(output))
    if peak > 0 and peak > amplitude:
        output = amplitude * output / peak

    return output


# ---------------------------------------------------------------------------
# Ambient Texture
# ---------------------------------------------------------------------------

def ambient_texture(
    key: str = "D3",
    scale_type: str = "pentatonic_minor",
    duration_sec: float = 60.0,
    layers: int = 3,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an evolving ambient texture.

    Creates a multi-layer ambient piece with:
    - A sustained drone pad on the root
    - Sparse melodic notes from the scale
    - Filtered noise texture

    All with slow LFO-driven filter movement for organic evolution.

    Args:
        key: Root note (e.g., "D3", "A2").
        scale_type: Scale type for melodic content.
        duration_sec: Duration in seconds.
        layers: Number of melodic layers (1-5).
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.
        seed: Random seed for reproducibility.

    Returns:
        Mono signal.
    """
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_sec)
    notes = scale_hz(key, scale_type, octaves=2)
    root_hz = note_to_hz(key)

    all_layers = []

    # Layer 1: Drone pad on root
    drone_preset = get_preset("dark_ambient_pad")
    drone = drone_preset.render(root_hz, duration_sec,
                                amplitude=0.4, sample_rate=sample_rate)
    all_layers.append(drone)

    # Layer 2+: Sparse melodic layers
    for layer_i in range(min(layers, 4)):
        layer_seed = seed + layer_i + 1 if seed is not None else None
        layer_rng = np.random.default_rng(layer_seed)

        # Pick a preset for this layer
        preset_names = ["ethereal_pad", "glass_bell", "shimmer_pad", "pluck"]
        preset = get_preset(preset_names[layer_i % len(preset_names)])

        layer_signal = np.zeros(n, dtype=np.float64)

        # Place sparse notes
        time_pos = layer_rng.uniform(1.0, 5.0)  # start offset
        while time_pos < duration_sec - 5.0:
            freq = layer_rng.choice(notes)
            note_dur = layer_rng.uniform(2.0, 6.0)
            note_sig = preset.render(float(freq), note_dur,
                                     amplitude=0.2, sample_rate=sample_rate)

            pos = int(time_pos * sample_rate)
            end = min(pos + len(note_sig), n)
            length = end - pos
            if length > 0:
                layer_signal[pos:end] += note_sig[:length]

            # Next note: random gap
            time_pos += note_dur + layer_rng.uniform(2.0, 8.0)

        all_layers.append(layer_signal)

    # Mix all layers
    result = mix_signals(all_layers)

    # Global filter: slow breathing
    lfo = LFO(rate=0.05, shape="sine", depth=1.0, bipolar=False)
    cutoff_env = lfo.modulate_param(2000, 1000, n, sample_rate)
    from fractal.modulation import apply_param_automation
    lpf = LowPassFilter(cutoff_hz=2000)
    result = apply_param_automation(result, lpf, "cutoff_hz", cutoff_env,
                                   sample_rate=sample_rate)

    # Fade in/out
    fade = SmoothFade(fade_in=4.0, fade_out=4.0)
    result = fade.apply(result)

    # Master effects
    fx = EffectChain([
        Reverb(decay=0.7, mix=0.4),
        NormalizePeak(target_db=-3.0),
    ])
    result = fx.process(result, sample_rate)

    # Final amplitude
    peak = np.max(np.abs(result))
    if peak > 0:
        result = amplitude * result / peak

    return result


# ---------------------------------------------------------------------------
# Chord Progression Render
# ---------------------------------------------------------------------------

def chord_progression_render(
    key: str = "C4",
    prog_name: str = "I_V_vi_IV",
    preset: SynthPreset | str = "warm_analog_pad",
    bars_per_chord: int = 2,
    tempo_bpm: float = 80.0,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render a chord progression with a given preset.

    Each chord is sustained for bars_per_chord bars with crossfade transitions.

    Args:
        key: Root key (e.g., "C4", "D3").
        prog_name: Progression name from PROGRESSIONS dict.
        preset: SynthPreset instance or preset name string.
        bars_per_chord: Number of bars per chord.
        tempo_bpm: Tempo in BPM.
        amplitude: Peak amplitude.
        sample_rate: Sample rate in Hz.

    Returns:
        Mono signal.
    """
    if isinstance(preset, str):
        preset = get_preset(preset)

    chords = progression_hz(key, prog_name)
    beat_sec = 60.0 / tempo_bpm
    chord_duration = bars_per_chord * 4 * beat_sec
    crossfade_sec = min(1.0, chord_duration * 0.1)

    total_duration = chord_duration * len(chords)
    n_total = int(sample_rate * total_duration)
    output = np.zeros(n_total, dtype=np.float64)

    for i, chord_freqs in enumerate(chords):
        # Render each note of the chord
        chord_signals = []
        for freq in chord_freqs:
            note = preset.render(freq, chord_duration,
                                 amplitude=0.3, sample_rate=sample_rate)
            chord_signals.append(note)

        chord_mix = mix_signals(chord_signals)

        # Apply fade for smooth transitions
        fade = SmoothFade(fade_in=crossfade_sec, fade_out=crossfade_sec)
        chord_mix = fade.apply(chord_mix)

        # Place in output
        pos = int(i * chord_duration * sample_rate)
        end = min(pos + len(chord_mix), n_total)
        length = end - pos
        if length > 0:
            output[pos:end] += chord_mix[:length]

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = amplitude * output / peak

    return output
