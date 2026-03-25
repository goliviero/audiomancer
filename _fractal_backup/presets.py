"""Presets -- named synthesis recipes for instant sound design.

Each SynthPreset bundles a synth type, parameters, envelope, effects,
and optional modulation into a single callable that renders a note.
This is the vocabulary layer that lets Claude map "blade runner pad"
to concrete Fractal code.
"""

from dataclasses import dataclass, field
import numpy as np

from fractal.constants import SAMPLE_RATE, DEFAULT_AMPLITUDE
from fractal.synth import (
    fm_synth, additive, subtractive, wavetable, pulse, unison,
    HARMONIC_PRESETS,
)
from fractal.generators import sine, sawtooth, square, triangle
from fractal.drums import kick, snare, hihat, clap, tom, cymbal, drum_kit
from fractal.envelopes import ADSR, SmoothFade, Swell
from fractal.effects import (
    Effect, LowPassFilter, HighPassFilter, Reverb, Delay,
    Distortion, NormalizePeak, EffectChain,
)
from fractal.modulation import LFO, apply_vibrato, apply_filter_sweep
from fractal.music_theory import note_to_hz


# ---------------------------------------------------------------------------
# SynthPreset dataclass
# ---------------------------------------------------------------------------

@dataclass
class SynthPreset:
    """A named synthesis recipe.

    Bundles everything needed to render a musical note: synth engine,
    parameters, envelope, effects, and optional modulation.

    Usage:
        preset = get_preset("blade_runner_pad")
        signal = preset.render("D4", 8.0)
    """

    name: str
    category: str  # "pad", "lead", "bass", "key", "texture", "fx"
    description: str
    synth_type: str  # "fm", "subtractive", "additive", "wavetable", "basic", "unison"
    synth_params: dict = field(default_factory=dict)
    envelope: object | None = None
    effects: list[Effect] = field(default_factory=list)
    vibrato_rate: float = 0.0
    vibrato_depth_cents: float = 0.0
    filter_sweep: tuple | None = None  # (start_hz, end_hz, curve)

    def render(
        self,
        note: str | float,
        duration_sec: float,
        amplitude: float = DEFAULT_AMPLITUDE,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Render a note using this preset.

        Args:
            note: Note name ("C4", "A#3") or frequency in Hz.
            duration_sec: Duration in seconds.
            amplitude: Peak amplitude.
            sample_rate: Sample rate in Hz.

        Returns:
            Mono signal.
        """
        freq = note_to_hz(note) if isinstance(note, str) else float(note)
        n = int(sample_rate * duration_sec)

        # Generate raw signal based on synth type
        signal = self._generate(freq, duration_sec, sample_rate)

        # Apply envelope
        if self.envelope is not None:
            env_curve = self.envelope.generate(n, sample_rate)
            signal = signal[:n] * env_curve[:n]

        # Apply vibrato
        if self.vibrato_rate > 0 and self.vibrato_depth_cents > 0:
            signal = apply_vibrato(
                signal, rate=self.vibrato_rate,
                depth_cents=self.vibrato_depth_cents,
                sample_rate=sample_rate,
            )

        # Apply filter sweep
        if self.filter_sweep is not None:
            start_hz, end_hz, curve = self.filter_sweep
            signal = apply_filter_sweep(
                signal, start_hz, end_hz,
                filter_type="lowpass", curve=curve,
                sample_rate=sample_rate,
            )

        # Apply effects chain
        if self.effects:
            chain = EffectChain(self.effects)
            signal = chain.process(signal, sample_rate)

        # Normalize to target amplitude
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = amplitude * signal / peak

        return signal

    def _generate(
        self, freq: float, duration_sec: float, sample_rate: int,
    ) -> np.ndarray:
        """Generate raw signal from synth engine."""
        p = self.synth_params

        if self.synth_type == "fm":
            ratio = p.get("ratio", 1.0)
            mod_index = p.get("mod_index", 2.0)
            mod_env = None
            mod_decay = p.get("mod_decay", None)
            if mod_decay is not None:
                n = int(sample_rate * duration_sec)
                mod_env = np.exp(-mod_decay * np.linspace(0, 1, n))
            return fm_synth(
                freq, freq * ratio, mod_index, duration_sec,
                mod_envelope=mod_env, sample_rate=sample_rate,
            )

        elif self.synth_type == "subtractive":
            osc = p.get("oscillator", "saw")
            cutoff = p.get("cutoff_hz", 2000)
            resonance = p.get("resonance", 0.0)
            filter_env = None
            filter_decay = p.get("filter_decay", None)
            if filter_decay is not None:
                n = int(sample_rate * duration_sec)
                filter_env = np.exp(-filter_decay * np.linspace(0, 1, n))
            return subtractive(
                osc, freq, duration_sec, cutoff_hz=cutoff,
                resonance=resonance, filter_envelope=filter_env,
                sample_rate=sample_rate,
            )

        elif self.synth_type == "additive":
            harmonics = p.get("harmonics", HARMONIC_PRESETS.get("organ"))
            return additive(
                freq, harmonics, duration_sec, sample_rate=sample_rate,
            )

        elif self.synth_type == "unison":
            gen_name = p.get("generator", "sawtooth")
            gen_fn = {"sine": sine, "sawtooth": sawtooth, "square": square,
                      "triangle": triangle}.get(gen_name, sawtooth)
            voices = p.get("voices", 5)
            detune = p.get("detune_cents", 15.0)
            return unison(
                gen_fn, freq, duration_sec, voices=voices,
                detune_cents=detune, sample_rate=sample_rate,
            )

        elif self.synth_type == "basic":
            gen_name = p.get("generator", "sine")
            gen_fn = {"sine": sine, "sawtooth": sawtooth, "square": square,
                      "triangle": triangle}.get(gen_name, sine)
            return gen_fn(freq, duration_sec, sample_rate=sample_rate)

        elif self.synth_type == "pulse":
            duty = p.get("duty", 0.5)
            return pulse(freq, duration_sec, duty=duty, sample_rate=sample_rate)

        else:
            raise ValueError(f"Unknown synth_type: '{self.synth_type}'")


# ---------------------------------------------------------------------------
# Preset Library
# ---------------------------------------------------------------------------

SYNTH_PRESETS: dict[str, SynthPreset] = {

    # === Pads ===

    "blade_runner_pad": SynthPreset(
        name="blade_runner_pad",
        category="pad",
        description="Vangelis-inspired FM pad with slow modulation decay and reverb wash",
        synth_type="fm",
        synth_params={"ratio": 1.0, "mod_index": 3.5, "mod_decay": 2.0},
        envelope=ADSR(attack=2.0, decay=1.0, sustain=0.7, release=3.0),
        effects=[Reverb(decay=0.6, mix=0.4)],
        vibrato_rate=4.0, vibrato_depth_cents=8.0,
    ),

    "warm_analog_pad": SynthPreset(
        name="warm_analog_pad",
        category="pad",
        description="Warm subtractive pad with gentle filter sweep",
        synth_type="unison",
        synth_params={"generator": "sawtooth", "voices": 5, "detune_cents": 12},
        envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0),
        effects=[LowPassFilter(cutoff_hz=3000), Reverb(decay=0.5, mix=0.3)],
        vibrato_rate=3.5, vibrato_depth_cents=6.0,
    ),

    "ethereal_pad": SynthPreset(
        name="ethereal_pad",
        category="pad",
        description="Airy additive pad with string harmonics and long reverb",
        synth_type="additive",
        synth_params={"harmonics": HARMONIC_PRESETS["string"]},
        envelope=ADSR(attack=3.0, decay=1.0, sustain=0.6, release=4.0),
        effects=[Reverb(decay=0.7, mix=0.5)],
        vibrato_rate=3.0, vibrato_depth_cents=5.0,
    ),

    "dark_ambient_pad": SynthPreset(
        name="dark_ambient_pad",
        category="pad",
        description="Deep FM pad with metallic overtones for dark ambient",
        synth_type="fm",
        synth_params={"ratio": 1.41, "mod_index": 4.0, "mod_decay": 3.0},
        envelope=ADSR(attack=4.0, decay=2.0, sustain=0.5, release=5.0),
        effects=[LowPassFilter(cutoff_hz=2000), Reverb(decay=0.8, mix=0.5)],
    ),

    "shimmer_pad": SynthPreset(
        name="shimmer_pad",
        category="pad",
        description="Bright FM shimmer with fast mod decay and heavy reverb",
        synth_type="fm",
        synth_params={"ratio": 2.0, "mod_index": 5.0, "mod_decay": 6.0},
        envelope=ADSR(attack=1.0, decay=0.5, sustain=0.8, release=3.0),
        effects=[Reverb(decay=0.7, mix=0.6)],
        vibrato_rate=5.0, vibrato_depth_cents=4.0,
    ),

    # === Leads ===

    "glass_bell": SynthPreset(
        name="glass_bell",
        category="lead",
        description="DX7-style FM bell with fast mod decay",
        synth_type="fm",
        synth_params={"ratio": 1.0, "mod_index": 5.0, "mod_decay": 8.0},
        envelope=ADSR(attack=0.005, decay=0.3, sustain=0.15, release=1.5),
        effects=[Reverb(decay=0.5, mix=0.35)],
    ),

    "supersaw_lead": SynthPreset(
        name="supersaw_lead",
        category="lead",
        description="7-voice detuned sawtooth lead for trance/EDM",
        synth_type="unison",
        synth_params={"generator": "sawtooth", "voices": 7, "detune_cents": 25},
        envelope=ADSR(attack=0.01, decay=0.1, sustain=0.8, release=0.3),
        effects=[LowPassFilter(cutoff_hz=6000)],
    ),

    "pluck": SynthPreset(
        name="pluck",
        category="lead",
        description="Short subtractive pluck with fast filter decay",
        synth_type="subtractive",
        synth_params={"oscillator": "saw", "cutoff_hz": 5000, "filter_decay": 6.0},
        envelope=ADSR(attack=0.005, decay=0.2, sustain=0.1, release=0.3),
    ),

    "sine_lead": SynthPreset(
        name="sine_lead",
        category="lead",
        description="Pure sine lead with vibrato",
        synth_type="basic",
        synth_params={"generator": "sine"},
        envelope=ADSR(attack=0.05, decay=0.1, sustain=0.9, release=0.2),
        vibrato_rate=5.5, vibrato_depth_cents=15.0,
    ),

    "metallic_hit": SynthPreset(
        name="metallic_hit",
        category="lead",
        description="Inharmonic FM hit for metallic/percussive tones",
        synth_type="fm",
        synth_params={"ratio": 1.4, "mod_index": 6.0, "mod_decay": 10.0},
        envelope=ADSR(attack=0.001, decay=0.15, sustain=0.05, release=0.8),
        effects=[Reverb(decay=0.4, mix=0.25)],
    ),

    # === Basses ===

    "sub_bass": SynthPreset(
        name="sub_bass",
        category="bass",
        description="Deep pure sine sub bass",
        synth_type="basic",
        synth_params={"generator": "sine"},
        envelope=ADSR(attack=0.01, decay=0.05, sustain=0.9, release=0.1),
    ),

    "acid_squelch": SynthPreset(
        name="acid_squelch",
        category="bass",
        description="303-style acid bass with filter sweep",
        synth_type="subtractive",
        synth_params={"oscillator": "saw", "cutoff_hz": 4000, "resonance": 0.6,
                      "filter_decay": 5.0},
        envelope=ADSR(attack=0.005, decay=0.15, sustain=0.4, release=0.1),
        effects=[Distortion(drive=2.0, mix=0.3)],
    ),

    "warm_bass": SynthPreset(
        name="warm_bass",
        category="bass",
        description="Warm analog-style bass with gentle filter",
        synth_type="subtractive",
        synth_params={"oscillator": "saw", "cutoff_hz": 1500, "resonance": 0.2,
                      "filter_decay": 3.0},
        envelope=ADSR(attack=0.01, decay=0.2, sustain=0.6, release=0.2),
    ),

    "reese_bass": SynthPreset(
        name="reese_bass",
        category="bass",
        description="Detuned unison saw bass for DnB/dubstep",
        synth_type="unison",
        synth_params={"generator": "sawtooth", "voices": 3, "detune_cents": 8},
        envelope=ADSR(attack=0.01, decay=0.1, sustain=0.8, release=0.15),
        effects=[LowPassFilter(cutoff_hz=2000)],
    ),

    "square_bass": SynthPreset(
        name="square_bass",
        category="bass",
        description="Punchy square wave bass with bite",
        synth_type="subtractive",
        synth_params={"oscillator": "square", "cutoff_hz": 2000, "filter_decay": 4.0},
        envelope=ADSR(attack=0.005, decay=0.1, sustain=0.7, release=0.1),
    ),

    # === Keys ===

    "electric_piano": SynthPreset(
        name="electric_piano",
        category="key",
        description="FM electric piano (Rhodes-style)",
        synth_type="fm",
        synth_params={"ratio": 1.0, "mod_index": 1.5, "mod_decay": 4.0},
        envelope=ADSR(attack=0.005, decay=0.4, sustain=0.3, release=0.5),
        effects=[Reverb(decay=0.3, mix=0.2)],
    ),

    "organ": SynthPreset(
        name="organ",
        category="key",
        description="Additive organ with classic drawbar harmonics",
        synth_type="additive",
        synth_params={"harmonics": HARMONIC_PRESETS["organ"]},
        envelope=ADSR(attack=0.02, decay=0.05, sustain=0.9, release=0.1),
        vibrato_rate=6.0, vibrato_depth_cents=10.0,
    ),

    # === Textures ===

    "ambient_drone": SynthPreset(
        name="ambient_drone",
        category="texture",
        description="Slow-evolving additive drone for ambient backgrounds",
        synth_type="additive",
        synth_params={"harmonics": HARMONIC_PRESETS["string"]},
        envelope=Swell(rise_time=5.0),
        effects=[LowPassFilter(cutoff_hz=2500), Reverb(decay=0.7, mix=0.5)],
        vibrato_rate=2.0, vibrato_depth_cents=3.0,
    ),

    "strings_ensemble": SynthPreset(
        name="strings_ensemble",
        category="texture",
        description="Lush unison strings with slow attack and reverb",
        synth_type="unison",
        synth_params={"generator": "sawtooth", "voices": 5, "detune_cents": 10},
        envelope=ADSR(attack=1.0, decay=0.5, sustain=0.8, release=1.5),
        effects=[LowPassFilter(cutoff_hz=4000), Reverb(decay=0.5, mix=0.3)],
        vibrato_rate=4.5, vibrato_depth_cents=8.0,
    ),

    "noise_texture": SynthPreset(
        name="noise_texture",
        category="texture",
        description="Filtered noise texture for ambient layering",
        synth_type="subtractive",
        synth_params={"oscillator": "saw", "cutoff_hz": 800, "resonance": 0.3},
        envelope=SmoothFade(fade_in=2.0, fade_out=2.0),
        effects=[Reverb(decay=0.6, mix=0.4)],
    ),
}


# ---------------------------------------------------------------------------
# Drum Presets (reference to drum_kit styles)
# ---------------------------------------------------------------------------

DRUM_PRESETS = ["808", "909", "acoustic", "lo-fi", "industrial"]


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_preset(name: str) -> SynthPreset:
    """Look up a synth preset by name (supports partial matching).

    Args:
        name: Preset name or partial match (e.g., "blade" -> "blade_runner_pad").

    Returns:
        SynthPreset instance.

    Raises:
        ValueError: If no preset matches or multiple presets match.
    """
    # Exact match first
    if name in SYNTH_PRESETS:
        return SYNTH_PRESETS[name]

    # Partial match
    matches = [k for k in SYNTH_PRESETS if name.lower() in k.lower()]

    if len(matches) == 1:
        return SYNTH_PRESETS[matches[0]]
    elif len(matches) == 0:
        raise ValueError(f"No preset matching '{name}'. "
                         f"Available: {list(SYNTH_PRESETS.keys())}")
    else:
        raise ValueError(f"Ambiguous preset '{name}'. "
                         f"Matches: {matches}")


def list_presets(category: str | None = None) -> list[str]:
    """List available preset names, optionally filtered by category.

    Args:
        category: Filter by category ("pad", "lead", "bass", "key", "texture").
            None returns all presets.

    Returns:
        Sorted list of preset names.
    """
    if category is None:
        return sorted(SYNTH_PRESETS.keys())

    return sorted(
        name for name, preset in SYNTH_PRESETS.items()
        if preset.category == category
    )
