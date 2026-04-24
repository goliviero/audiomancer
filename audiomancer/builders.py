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
    lfo_sine,
    multi_lfo,
    random_walk,
)
from audiomancer.saturation import tape_saturate
from audiomancer.spatial import auto_pan, haas_width
from audiomancer.synth import chord_pad, sine, triangle
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
# Texture wrapper — any textures/ preset as a builder
# ---------------------------------------------------------------------------

def texture(duration: float, seed: int, sample_rate: int,
            texture_name: str, **texture_params) -> np.ndarray:
    """Wrap any preset from audiomancer.textures as a builder.

    Covers the 9 built-in textures (ethereal_wash, crystal_shimmer,
    breathing_pad, evolving_drone, singing_bowl, ocean_bed, earth_hum,
    deep_space, noise_wash) with a single config entry.

    Args:
        texture_name: Name of the texture (see textures.list_textures()).
        **texture_params: Passed through to the preset (e.g. frequency,
            base_freq, frequencies depending on the preset).
    """
    from audiomancer.textures import generate as _gen
    return _gen(texture_name, duration_sec=duration, seed=seed,
                sample_rate=sample_rate, **texture_params)


# ---------------------------------------------------------------------------
# Instrument synth / sampled — ethnic instruments as builders
# ---------------------------------------------------------------------------

def instrument_synth(duration: float, seed: int, sample_rate: int,
                     name: str, **params) -> np.ndarray:
    """Synthetic ethnic instrument: didgeridoo / handpan / oud / sitar / derbouka_pattern."""
    from audiomancer import instruments as _inst
    from audiomancer.utils import mono_to_stereo

    fn_map = {
        "didgeridoo": _inst.didgeridoo,
        "handpan": _inst.handpan,
        "oud": _inst.oud,
        "sitar": _inst.sitar,
        "derbouka_pattern": _inst.derbouka_pattern,
    }
    if name not in fn_map:
        raise ValueError(
            f"Unknown instrument {name!r}. Valid: {list(fn_map)}"
        )
    mono = fn_map[name](duration_sec=duration, seed=seed,
                        sample_rate=sample_rate, **params)
    return mono_to_stereo(mono)


def instrument_sampled(duration: float, seed: int, sample_rate: int,
                       source_path: str, source_hz: float, target_hz: float,
                       mode: str = "pad", **params) -> np.ndarray:
    """Load a CC0 sample, pitch-shift to target_hz, optionally paulstretch.

    Args:
        source_path: Path to WAV (relative to project root).
        source_hz: Native pitch of the sample.
        target_hz: Desired pitch.
        mode: 'note' (preserve timbre via pitch-shift) or 'pad' (pitch-shift
              + paulstretch into a long ambient pad).
    """
    from pathlib import Path

    from audiomancer.sampler import pitched_pad, play_note
    from audiomancer.utils import load_audio, mono_to_stereo

    sig, _ = load_audio(Path(source_path), target_sr=sample_rate)

    if mode == "note":
        out = play_note(sig, source_hz=source_hz, target_hz=target_hz,
                        duration_sec=duration, sample_rate=sample_rate,
                        **params)
    elif mode == "pad":
        out = pitched_pad(sig, source_hz=source_hz, target_hz=target_hz,
                          duration_sec=duration, seed=seed,
                          sample_rate=sample_rate, **params)
    else:
        raise ValueError(f"mode must be 'note' or 'pad', got {mode!r}")

    if out.ndim == 1:
        out = mono_to_stereo(out)
    return out


# ---------------------------------------------------------------------------
# Piano processed — load a recorded piano WAV + apply a piano_presets preset
# ---------------------------------------------------------------------------

def morph_textures(duration: float, seed: int, sample_rate: int,
                   texture_a: dict, texture_b: dict) -> np.ndarray:
    """Generate 2 textures and morph A -> B across the duration.

    Args:
        texture_a / texture_b: each a dict with keys:
            - "name": texture preset name (e.g. "crystal_shimmer")
            - "params": dict of texture-specific params (e.g. {"frequency": 396.0})

    Returns stereo ndarray.
    """
    from audiomancer.spectral import morph as _morph
    from audiomancer.textures import generate as _gen

    sig_a = _gen(
        texture_a["name"], duration_sec=duration, seed=seed,
        sample_rate=sample_rate, **texture_a.get("params", {}),
    )
    sig_b = _gen(
        texture_b["name"], duration_sec=duration, seed=seed + 1,
        sample_rate=sample_rate, **texture_b.get("params", {}),
    )
    min_len = min(sig_a.shape[0], sig_b.shape[0])
    return _morph(sig_a[:min_len], sig_b[:min_len], sample_rate=sample_rate)


def piano_processed(duration: float, seed: int, sample_rate: int,
                    source_path: str,
                    preset: str = "mid_pad") -> np.ndarray:
    """Load a piano WAV and apply one of the piano_presets presets.

    Args:
        source_path: Path to the raw piano .wav (relative to project root).
        preset: 'bass_drone' | 'mid_pad' | 'sparse_notes'.

    Returns stereo ndarray ready for LUFS + master chain.
    """
    from pathlib import Path

    from audiomancer.piano_presets import PRESETS
    from audiomancer.utils import load_audio

    path = Path(source_path)
    sig, _ = load_audio(path, target_sr=sample_rate)
    preset_fn, _default_lufs = PRESETS[preset]
    return preset_fn(sig, duration, sample_rate)


# ---------------------------------------------------------------------------
# Foundation drone — deep, zero-transient sine pair with slow amp breath
# ---------------------------------------------------------------------------

def foundation_drone(duration: float, seed: int, sample_rate: int,
                     freqs: list[float],
                     detune_cents: float = 3.0,
                     lp_hz: float = 500.0,
                     amp_mod_cycle_sec: float = 22.0,
                     amp_mod_depth_db: float = 1.0,
                     reverb_room: float = 0.55,
                     reverb_wet: float = 0.18) -> np.ndarray:
    """Continuous tectonic drone — detuned sine pairs, heavy lowpass, slow breath.

    Designed as the bedrock layer of a grounding ambient piece. No attack
    transient (pure continuous sines), no filter sweep, minimal reverb. The
    only motion is a ±amp_mod_depth_db volume breath over amp_mod_cycle_sec.

    Args:
        freqs: Fundamental frequencies (Hz). Each gets a detuned partner.
        detune_cents: Cents offset for the detuned partner (±).
        lp_hz: Lowpass cutoff. Keep ≤600Hz for sub-domain only.
        amp_mod_cycle_sec: Period of the slow amplitude breath.
        amp_mod_depth_db: ± dB of the breath modulation.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    signal = np.zeros(n)

    for i, freq in enumerate(freqs):
        phase_jitter = rng.uniform(0, 2 * np.pi)
        partner_cents = detune_cents * (1 if i % 2 == 0 else -1)
        partner_freq = freq * 2 ** (partner_cents / 1200)
        t = np.linspace(0, duration, n, endpoint=False)
        signal += np.sin(2 * np.pi * freq * t + phase_jitter)
        signal += np.sin(2 * np.pi * partner_freq * t + phase_jitter * 0.7)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * 0.8 / peak

    signal = lowpass(signal, cutoff_hz=lp_hz, sample_rate=sample_rate)
    stereo = mono_to_stereo(signal)

    # Slow volume breath: convert dB to linear depth
    depth_linear = 10 ** (amp_mod_depth_db / 20) - 1.0
    breath = lfo_sine(
        duration, rate_hz=1.0 / amp_mod_cycle_sec,
        depth=depth_linear, offset=1.0, sample_rate=sample_rate,
    )
    stereo = apply_amplitude_mod(stereo, breath)

    stereo = reverb(stereo, room_size=reverb_room, damping=0.75,
                    wet_level=reverb_wet, sample_rate=sample_rate)
    return stereo


# ---------------------------------------------------------------------------
# Ochre pad — warm chord with hinted accent note + tape saturation
# ---------------------------------------------------------------------------

def ochre_pad(duration: float, seed: int, sample_rate: int,
              chord: list[tuple[float, float]],
              voices: int = 4,
              detune_cents: float = 12.0,
              lp_hz: float = 3000.0,
              sat_drive: float = 1.1,
              breath_cycle_sec: float = 28.0,
              breath_depth: float = 0.04,
              reverb_room: float = 0.7,
              reverb_wet: float = 0.38) -> np.ndarray:
    """Warm organic pad — stacked detuned voices with tape saturation.

    Each chord entry is ``(hz, volume_db)``: 0 dB for the main voice, negative
    for hinted-below-audibility accents. Chain: per-note chord_pad (detuned
    sines) → sum → tape saturation (even harmonics = warmth) → lowpass →
    slow breathing → reverb.

    Args:
        chord: List of (freq_hz, volume_db) tuples. The main audible pitch
            should have volume_db close to 0; accent tones go negative
            (e.g. -20dB for "felt more than heard" perfect-fifth hint).
        sat_drive: Tape saturation drive (1.0 = gentle, 1.5-1.7 = warm body).
    """
    signal = np.zeros(int(duration * sample_rate))
    for i, (freq, vol_db) in enumerate(chord):
        gain = 10 ** (vol_db / 20)
        voice = chord_pad([freq], duration, voices=voices,
                          detune_cents=detune_cents, amplitude=0.7,
                          seed=seed + i * 101, jitter_cents=2.0,
                          sample_rate=sample_rate)
        signal += gain * voice

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * 0.85 / peak

    # Tape saturation — even-order harmonics for warmth, no brightness
    signal = tape_saturate(signal, drive=sat_drive, asymmetry=0.18)

    # Lowpass after saturation to tame any harmonics saturation added
    signal = lowpass(signal, cutoff_hz=lp_hz, sample_rate=sample_rate)
    stereo = mono_to_stereo(signal)

    # Slow breath — barely moving
    breath = lfo_sine(
        duration, rate_hz=1.0 / breath_cycle_sec,
        depth=breath_depth, offset=1.0, sample_rate=sample_rate,
    )
    stereo = apply_amplitude_mod(stereo, breath)

    stereo = reverb(stereo, room_size=reverb_room, damping=0.55,
                    wet_level=reverb_wet, sample_rate=sample_rate)
    return stereo


# ---------------------------------------------------------------------------
# Sparse sample events — place 1-N pitched sample copies with long fades
# ---------------------------------------------------------------------------

def sparse_sample_events(duration: float, seed: int, sample_rate: int,
                         source_path: str,
                         source_hz: float,
                         target_hz: float,
                         event_count: int = 2,
                         event_dur_range: tuple[float, float] = (18.0, 25.0),
                         fade_in_sec: float = 18.0,
                         fade_out_sec: float = 20.0,
                         pitch_drift_cents: float = 30.0,
                         stereo_width: float = 0.15,
                         hp_hz: float = 40.0,
                         lp_hz: float = 4000.0,
                         reverb_room: float = 0.6,
                         reverb_wet: float = 0.35) -> np.ndarray:
    """Place N sparse long-fade sample events across a duration.

    Each event: pitch-shift source by ±pitch_drift_cents, tile-or-trim to a
    random duration in event_dur_range, apply long fade_in/fade_out, pan by
    ±stereo_width, HP/LP band-limit. Events are placed at non-overlapping
    random positions. Silence between events is intentional.

    Args:
        source_path: Path to source WAV (relative to project root).
        source_hz: Native pitch of the sample.
        target_hz: Center pitch for events.
        event_count: Number of events to place (1-N).
        event_dur_range: (min, max) duration of each event in seconds.
        fade_in_sec / fade_out_sec: Long linear fades per event.
        pitch_drift_cents: ± cents jitter per event.
        stereo_width: Pan range per event (0.0 = centered, 0.15 = ±15%).
    """
    from pathlib import Path

    from audiomancer.sampler import play_note
    from audiomancer.utils import load_audio

    rng = np.random.default_rng(seed)
    n_total = int(duration * sample_rate)
    output = np.zeros((n_total, 2))

    sig, _ = load_audio(Path(source_path), target_sr=sample_rate)
    if sig.ndim == 2:
        sig = sig.mean(axis=1)

    # Schedule non-overlapping event windows
    events = []
    attempts = 0
    while len(events) < event_count and attempts < 200:
        attempts += 1
        evt_dur = rng.uniform(*event_dur_range)
        start_max = duration - evt_dur
        if start_max <= 0:
            break
        start = rng.uniform(0.0, start_max)
        window = (start, start + evt_dur)
        overlap = any(not (window[1] < e[0] or window[0] > e[1])
                      for e in events)
        if overlap:
            continue
        events.append(window)

    for start, end in events:
        evt_dur = end - start
        drift = rng.uniform(-pitch_drift_cents, pitch_drift_cents)
        evt_hz = target_hz * 2 ** (drift / 1200)

        # Pitch-shift + tile/trim to evt_dur
        shifted = play_note(sig, source_hz=source_hz, target_hz=evt_hz,
                            duration_sec=evt_dur, amplitude=0.85,
                            sample_rate=sample_rate)

        # Band-limit
        shifted = highpass(shifted, cutoff_hz=hp_hz, sample_rate=sample_rate)
        shifted = lowpass(shifted, cutoff_hz=lp_hz, sample_rate=sample_rate)

        # Long fades
        n_evt = int(evt_dur * sample_rate)
        n_fi = min(int(fade_in_sec * sample_rate), n_evt // 2)
        n_fo = min(int(fade_out_sec * sample_rate), n_evt - n_fi)
        env = np.ones(n_evt)
        env[:n_fi] = np.linspace(0.0, 1.0, n_fi)
        env[-n_fo:] = np.linspace(1.0, 0.0, n_fo)
        shifted = shifted[:n_evt] * env

        # Pan ±stereo_width (constant-power)
        pan = rng.uniform(-stereo_width, stereo_width)
        left_gain = np.cos((pan + 1) * np.pi / 4)
        right_gain = np.sin((pan + 1) * np.pi / 4)
        stereo_evt = np.column_stack([shifted * left_gain,
                                      shifted * right_gain])

        # Place
        start_s = int(start * sample_rate)
        end_s = min(start_s + len(stereo_evt), n_total)
        output[start_s:end_s] += stereo_evt[:end_s - start_s]

    if reverb_wet > 0:
        output = reverb(output, room_size=reverb_room, damping=0.6,
                        wet_level=reverb_wet, sample_rate=sample_rate)
    return output


# ---------------------------------------------------------------------------
# Subliminal sine — felt-more-than-heard low hum with slow tremolo
# ---------------------------------------------------------------------------

def subliminal_sine(duration: float, seed: int, sample_rate: int,
                    freq: float = 60.0,
                    tremolo_cycle_sec: float = 25.0,
                    tremolo_depth_db: float = 2.0) -> np.ndarray:
    """Pure low sine with slow amplitude tremolo — meant to sit far below mix.

    Output is peak-normalized to 0.5 — the mix level controls how subliminal
    it actually becomes (target -30dB below main layers).

    Args:
        freq: Fundamental (typically 60Hz for Earth / Schumann-adjacent).
        tremolo_cycle_sec: Period of the tremolo cycle.
        tremolo_depth_db: ± dB depth of the tremolo.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t + phase)

    stereo = mono_to_stereo(signal)
    depth_linear = 10 ** (tremolo_depth_db / 20) - 1.0
    trem = lfo_sine(duration, rate_hz=1.0 / tremolo_cycle_sec,
                    depth=depth_linear, offset=1.0, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, trem)
    return stereo


# ---------------------------------------------------------------------------
# Registry — string key -> builder function
# ---------------------------------------------------------------------------

REGISTRY = {
    "pad_alive": pad_alive,
    "pendulum_bass": pendulum_bass,
    "arpege_bass": arpege_bass,
    "binaural_beat": binaural_beat,
    "texture": texture,
    "piano_processed": piano_processed,
    "morph_textures": morph_textures,
    "instrument_synth": instrument_synth,
    "instrument_sampled": instrument_sampled,
    "foundation_drone": foundation_drone,
    "ochre_pad": ochre_pad,
    "sparse_sample_events": sparse_sample_events,
    "subliminal_sine": subliminal_sine,
}
