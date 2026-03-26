"""quick — API one-liner pour se faire pas chier.

Au lieu de chaîner 5 fonctions pour un drone, tu fais juste :

    from audiomancer import quick

    sig = quick.drone(136.1, 300)
    sig = quick.pad([261.63, 329.63, 392.0], 300)
    sig = quick.binaural("theta_deep", 300)
    sig = quick.texture("deep_space", 300, seed=42)
    sig = quick.save(sig, "my_drone")

Toutes les fonctions retournent du stéréo normalisé à -1 dB, prêt à exporter.
"""

from pathlib import Path

import numpy as np

from audiomancer import SAMPLE_RATE
from audiomancer.binaural import binaural as _binaural
from audiomancer.binaural import from_preset as _from_preset
from audiomancer.effects import chorus_subtle, lowpass, reverb_cathedral, reverb_hall
from audiomancer.layers import mix as _mix
from audiomancer.layers import normalize_lufs
from audiomancer.modulation import apply_amplitude_mod, evolving_lfo
from audiomancer.synth import chord_pad as _chord_pad
from audiomancer.synth import drone as _drone
from audiomancer.textures import generate as _texture_generate
from audiomancer.utils import (
    export_wav,
    fade_in,
    fade_out,
    mono_to_stereo,
    normalize,
)

# ---------------------------------------------------------------------------
# Note → fréquence (A4 = 440 Hz, tempérament égal)
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note(name: str) -> float:
    """Convert a note name to Hz. Examples: 'A4' → 440.0, 'C3' → 130.81.

    Args:
        name: Note name with octave (e.g. 'A4', 'C#3', 'Bb2').

    Returns:
        Frequency in Hz.
    """
    name = name.replace("b", "#").replace("Bb", "A#").replace("Eb", "D#") \
               .replace("Ab", "G#").replace("Db", "C#").replace("Gb", "F#")
    octave = int(name[-1])
    pitch = name[:-1].upper()
    if pitch not in _NOTE_NAMES:
        raise ValueError(f"Unknown note: {pitch!r}")
    semitone = _NOTE_NAMES.index(pitch) + (octave + 1) * 12
    return 440.0 * 2 ** ((semitone - 69) / 12)


# ---------------------------------------------------------------------------
# Fréquences sacrées / communes — pour s'en souvenir sans googler
# ---------------------------------------------------------------------------

FREQS = {
    # Solfège frequencies
    "ut":        174.0,
    "re":        285.0,
    "mi":        396.0,
    "fa":        417.0,
    "sol":       528.0,
    "la":        639.0,
    "si":        741.0,
    "do":        852.0,
    "divine":    963.0,
    # Special
    "om":        136.1,
    "holy":      111.0,
    "schumann":  7.83,
    "gamma":     40.0,
    # Standard notes
    "a4":        440.0,
    "a3":        220.0,
    "a2":        110.0,
    "c4":        261.63,
    "c3":        130.81,
}


# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------

# Harmonics presets
HARMONICS_WARM = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06), (6, 0.03)]
HARMONICS_BRIGHT = [(1, 1.0), (2, 0.7), (3, 0.5), (4, 0.3), (5, 0.15)]
HARMONICS_DARK = [(1, 1.0), (2, 0.3), (3, 0.1)]
HARMONICS_BOWL = [(1, 1.0), (2.71, 0.6), (5.40, 0.35), (8.93, 0.15)]  # inharmonique


def drone(
    frequency: float,
    duration_sec: float,
    harmonics: list[tuple[float, float]] | None = None,
    cutoff_hz: float = 1800,
    seed: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Drone complet prêt à l'emploi : synthesis + filter + reverb + evolving mod.

    Args:
        frequency: Fondamentale en Hz. Utilise FREQS["om"] ou note("A3") si besoin.
        duration_sec: Durée.
        harmonics: Série harmonique. Défaut : HARMONICS_WARM.
        cutoff_hz: Fréquence de coupure du lowpass. Baisse pour plus sombre.
        seed: Graine pour la modulation.
        sample_rate: Sample rate.

    Returns:
        Stéréo normalisé à -1 dB.
    """
    if harmonics is None:
        harmonics = HARMONICS_WARM

    raw = _drone(frequency, duration_sec, harmonics=harmonics,
                 amplitude=0.7, sample_rate=sample_rate)
    raw = lowpass(raw, cutoff_hz=cutoff_hz, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)

    # Légère modulation d'amplitude
    mod = evolving_lfo(duration_sec, rate_hz=0.03, depth=0.06,
                       offset=1.0, drift_speed=0.04,
                       seed=seed, sample_rate=sample_rate)
    stereo = apply_amplitude_mod(stereo, mod)

    stereo = reverb_cathedral(stereo, sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


# ---------------------------------------------------------------------------
# Pads
# ---------------------------------------------------------------------------

def pad(
    frequencies: list[float],
    duration_sec: float,
    voices: int = 4,
    detune_cents: float = 10.0,
    dark: bool = False,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Chord pad complet : synthesis + chorus + reverb.

    Args:
        frequencies: Liste de fréquences (ex: [261.63, 329.63, 392.0] = Do majeur).
                     Utilise note("C3") pour convertir.
        duration_sec: Durée.
        voices: Voix désaccordées par note.
        detune_cents: Spread de désaccordage.
        dark: True = lowpass 2000 Hz + hall reverb (sombre). False = 4000 Hz + cathédrale.
        sample_rate: Sample rate.

    Returns:
        Stéréo normalisé à -1 dB.
    """
    raw = _chord_pad(frequencies, duration_sec, voices=voices,
                     detune_cents=detune_cents, amplitude=0.5,
                     sample_rate=sample_rate)
    cutoff = 2000 if dark else 4000
    raw = lowpass(raw, cutoff_hz=cutoff, sample_rate=sample_rate)
    stereo = mono_to_stereo(raw)
    stereo = chorus_subtle(stereo, sample_rate=sample_rate)
    reverb_fn = reverb_hall if dark else reverb_cathedral
    stereo = reverb_fn(stereo, sample_rate=sample_rate)
    return normalize(stereo, target_db=-1.0)


# ---------------------------------------------------------------------------
# Binaural
# ---------------------------------------------------------------------------

def binaural(preset: str, duration_sec: float,
             volume_db: float = -6.0,
             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Binaural beat depuis un preset nommé.

    Presets disponibles: theta_deep, alpha_relax, delta_sleep,
                         solfeggio_528, solfeggio_432, om_theta.

    Args:
        preset: Nom du preset.
        duration_sec: Durée.
        volume_db: Volume (binaural doit être discret, -6 à -20 dB).
        sample_rate: Sample rate.

    Returns:
        Stéréo.
    """
    sig = _from_preset(preset, duration_sec, sample_rate=sample_rate)
    gain = 10 ** (volume_db / 20)
    return sig * gain


def binaural_custom(carrier_hz: float, beat_hz: float, duration_sec: float,
                    volume_db: float = -6.0,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Binaural beat custom (carrier + beat frequency).

    Args:
        carrier_hz: Fréquence porteuse.
        beat_hz: Fréquence du beat binaural (différence L/R).
        duration_sec: Durée.
        volume_db: Volume.
        sample_rate: Sample rate.
    """
    sig = _binaural(carrier_hz, beat_hz, duration_sec,
                    amplitude=0.4, sample_rate=sample_rate)
    gain = 10 ** (volume_db / 20)
    return sig * gain


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

def texture(name: str, duration_sec: float, seed: int | None = None,
            sample_rate: int = SAMPLE_RATE, **kwargs) -> np.ndarray:
    """Texture évolutive depuis la banque de presets.

    Presets: evolving_drone, breathing_pad, deep_space, ocean_bed,
             crystal_shimmer, earth_hum, ethereal_wash, singing_bowl, noise_wash.

    Returns:
        Stéréo normalisé.
    """
    return _texture_generate(name, duration_sec=duration_sec, seed=seed,
                             sample_rate=sample_rate, **kwargs)


# ---------------------------------------------------------------------------
# Mix
# ---------------------------------------------------------------------------

def mix(layers: list[tuple], duration_sec: float | None = None,
        target_lufs: float = -14.0,
        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mixe plusieurs layers avec volumes en dB.

    Args:
        layers: Liste de (signal, volume_db). Ex:
                [(drone_sig, 0.0), (binaural_sig, -8.0), (pad_sig, -12.0)]
        duration_sec: Si fourni, tronque/pad au bon nombre de samples.
        target_lufs: Normalisation LUFS finale.
        sample_rate: Sample rate.

    Returns:
        Mix stéréo normalisé.
    """
    signals = [s for s, _ in layers]
    volumes = [v for _, v in layers]
    result = _mix(signals, volumes_db=volumes)
    if duration_sec is not None:
        n = int(sample_rate * duration_sec)
        if result.shape[0] > n:
            result = result[:n]
    return normalize_lufs(result, target_lufs=target_lufs)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save(signal: np.ndarray, name: str,
         folder: str | Path = "output",
         fade_sec: float = 3.0) -> Path:
    """Export un signal en WAV avec fade in/out.

    Args:
        signal: Signal audio.
        name: Nom du fichier (sans extension).
        folder: Dossier de sortie.
        fade_sec: Durée des fades.

    Returns:
        Path du fichier créé.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sig = fade_in(fade_out(signal, fade_sec), fade_sec)
    path = folder / f"{name}.wav"
    export_wav(sig, path)
    print(f"Saved: {path}")
    return path
