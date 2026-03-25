"""Fractal — Python-native audio pipeline for composing and exporting music.

Public API re-exports for convenience:
    from fractal import SAMPLE_RATE, sine, Session, export_wav
"""

# Constants
from fractal.constants import SAMPLE_RATE, BIT_DEPTH, MAX_AMPLITUDE, DEFAULT_AMPLITUDE

# Signal utilities
from fractal.signal import (
    duration_samples, duration_seconds, is_mono, is_stereo,
    mono_to_stereo, stereo_to_mono, silence, normalize_peak,
    pad_to_length, trim, concat, mix_signals,
)

# Generators
from fractal.generators import (
    sine, square, sawtooth, triangle, white_noise, pink_noise,
    binaural, load_sample,
)

# Envelopes
from fractal.envelopes import (
    Envelope, FadeInOut, ADSR, SmoothFade, ExponentialFade,
    Swell, Gate, Tremolo, AutomationCurve,
)

# Effects
from fractal.effects import (
    Effect, LowPassFilter, HighPassFilter, BandPassFilter, EQ,
    StereoWidth, Reverb, Delay, Distortion, NormalizePeak, EffectChain,
)

# Track + Mixer
from fractal.track import Track, apply_pan
from fractal.mixer import Bus, Session

# Sequencer
from fractal.sequencer import Clip, Pattern, Sequencer

# Drums
from fractal.drums import kick, snare, hihat, clap, tom, cymbal, drum_kit

# Modulation
from fractal.modulation import LFO, apply_vibrato, apply_filter_sweep, apply_param_automation

# Presets
from fractal.presets import SynthPreset, SYNTH_PRESETS, DRUM_PRESETS, get_preset, list_presets

# Generative
from fractal.generative import (
    random_melody, weighted_random_notes, evolving_parameter,
    phrase_generator, ambient_texture, chord_progression_render,
)

# Synthesizers
from fractal.synth import (
    fm_synth, additive, wavetable, subtractive, pulse, unison,
    HARMONIC_PRESETS,
)

# Music theory
from fractal.music_theory import (
    note_to_hz, hz_to_note, interval_hz, transpose,
    scale, scale_hz, SCALES,
    chord, chord_hz, CHORD_TYPES,
    progression, progression_hz, PROGRESSIONS,
)

# Export
from fractal.export import export_wav, export_flac, export_auto

__all__ = [
    # Constants
    "SAMPLE_RATE", "BIT_DEPTH", "MAX_AMPLITUDE", "DEFAULT_AMPLITUDE",
    # Signal
    "duration_samples", "duration_seconds", "is_mono", "is_stereo",
    "mono_to_stereo", "stereo_to_mono", "silence", "normalize_peak",
    "pad_to_length", "trim", "concat", "mix_signals",
    # Generators
    "sine", "square", "sawtooth", "triangle", "white_noise", "pink_noise",
    "binaural", "load_sample",
    # Envelopes
    "Envelope", "FadeInOut", "ADSR", "SmoothFade", "ExponentialFade",
    "Swell", "Gate", "Tremolo", "AutomationCurve",
    # Effects
    "Effect", "LowPassFilter", "HighPassFilter", "BandPassFilter", "EQ",
    "StereoWidth", "Reverb", "Delay", "Distortion", "NormalizePeak", "EffectChain",
    # Track + Mixer
    "Track", "apply_pan", "Bus", "Session",
    # Sequencer
    "Clip", "Pattern", "Sequencer",
    # Drums
    "kick", "snare", "hihat", "clap", "tom", "cymbal", "drum_kit",
    # Modulation
    "LFO", "apply_vibrato", "apply_filter_sweep", "apply_param_automation",
    # Presets
    "SynthPreset", "SYNTH_PRESETS", "DRUM_PRESETS", "get_preset", "list_presets",
    # Generative
    "random_melody", "weighted_random_notes", "evolving_parameter",
    "phrase_generator", "ambient_texture", "chord_progression_render",
    # Synthesizers
    "fm_synth", "additive", "wavetable", "subtractive", "pulse", "unison",
    "HARMONIC_PRESETS",
    # Music theory
    "note_to_hz", "hz_to_note", "interval_hz", "transpose",
    "scale", "scale_hz", "SCALES",
    "chord", "chord_hz", "CHORD_TYPES",
    "progression", "progression_hz", "PROGRESSIONS",
    # Export
    "export_wav", "export_flac", "export_auto",
]
