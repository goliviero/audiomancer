"""V004 mix — full cycle preview (4 min).

Arpeggio palindrome: G2 -> Bb2 -> D3 -> G3 -> D3 -> Bb2 (6 notes x 40s = 240s)
Mid pad G3, crystal shimmer sparse (45-60s spacing), binaural 4 Hz theta.

Usage:
    python scripts/19_v004_mix_preview.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.compose import make_loopable, tremolo, verify_loop
from audiomancer.effects import chorus_subtle, delay, lowpass, reverb
from audiomancer.envelope import breathing
from audiomancer.layers import mix, normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.modulation import apply_amplitude_mod
from audiomancer.spatial import haas_width
from audiomancer.synth import chord_pad, drone, sawtooth, sine
from audiomancer.utils import export_wav, mono_to_stereo
import audiomancer.quick as q

SR = 48000
OUT = project_root / "output" / "V004"
DUR = 240  # 4 min = one full palindrome cycle

# G minor JI
ARPEGGIO_PALINDROME = [98.0, 117.6, 147.0, 196.0, 147.0, 117.6]
# G2, Bb2, D3, G3, D3, Bb2 (no duplicates at extremes)
NOTE_DUR = 40.0
XFADE = 5.0

HARMONICS_BASS = [(1, 1.0), (2, 0.35), (3, 0.12), (4, 0.05)]

# Shimmer cluster (G4 based)
CHIME_ROOT = 392.0
CHIME_FREQS = [
    CHIME_ROOT,
    CHIME_ROOT * 1.005,
    CHIME_ROOT * 1.498,
    CHIME_ROOT * 2.003,
    CHIME_ROOT * 2.51,
]


# ---------------------------------------------------------------------------
# Stem builders
# ---------------------------------------------------------------------------

def build_bass(duration: int) -> np.ndarray:
    """Bass arpeggio palindrome with crossfade glides."""
    n_samples = int(duration * SR)
    xfade_samples = int(XFADE * SR)
    note_samples = int(NOTE_DUR * SR)

    output = np.zeros(n_samples)
    pos = 0
    note_idx = 0

    while pos < n_samples:
        freq = ARPEGGIO_PALINDROME[note_idx % len(ARPEGGIO_PALINDROME)]
        note_idx += 1

        # Generate note with extra length for crossfade tail
        note_len_sec = NOTE_DUR + XFADE
        # Warm sine drone + subtle saw undertone
        main = drone(freq, note_len_sec, harmonics=HARMONICS_BASS,
                     amplitude=0.7, sample_rate=SR)
        saw = sawtooth(freq, note_len_sec, amplitude=0.15, sample_rate=SR)
        saw = lowpass(saw, cutoff_hz=300, sample_rate=SR)
        note = main + saw
        note = lowpass(note, cutoff_hz=250, sample_rate=SR)

        # Smooth fade envelope
        fade_in_env = np.linspace(0, 1, xfade_samples)
        fade_out_env = np.linspace(1, 0, xfade_samples)
        hold_samples = note_samples - xfade_samples
        env = np.concatenate([fade_in_env, np.ones(hold_samples), fade_out_env])
        env = env[:len(note)]
        if len(env) < len(note):
            note = note[:len(env)]
        note = note * env

        end = min(pos + len(note), n_samples)
        chunk = end - pos
        output[pos:end] += note[:chunk]
        pos += note_samples

    # Stereo + Haas 5ms
    stereo = mono_to_stereo(output)
    stereo = haas_width(stereo, delay_ms=5.0, sample_rate=SR)

    # Warm reverb
    stereo = reverb(stereo, room_size=0.8, damping=0.5, wet_level=0.45,
                    sample_rate=SR)

    # Breathing +-3dB, 30s cycle
    breath_env = breathing(duration, breath_rate=1.0 / 30.0, depth=0.29,
                           floor=0.71, sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, breath_env)

    return stereo


def build_pad(duration: int) -> np.ndarray:
    """Mid pad G3 (196 Hz) — warm sustained."""
    raw = chord_pad([196.0], duration, voices=4, detune_cents=5.0,
                    amplitude=0.5, sample_rate=SR)
    raw = lowpass(raw, cutoff_hz=2000, sample_rate=SR)
    stereo = mono_to_stereo(raw)
    stereo = chorus_subtle(stereo, sample_rate=SR)
    stereo = reverb(stereo, room_size=0.90, damping=0.4, wet_level=0.6,
                    sample_rate=SR)

    breath_env = breathing(duration, breath_rate=1.0 / 30.0, depth=0.25,
                           floor=0.75, sample_rate=SR)
    stereo = apply_amplitude_mod(stereo, breath_env)
    return stereo


def build_shimmer(duration: int) -> np.ndarray:
    """Crystal shimmer — sparse chime hits every 45-60s."""
    n_samples = int(duration * SR)
    rng = np.random.default_rng(42)

    output = np.zeros(n_samples)
    pos = 0.0
    hit_count = 0

    while pos < duration:
        gap = rng.uniform(45.0, 60.0)
        pos += gap
        if pos >= duration:
            break

        freq = rng.choice(CHIME_FREQS)
        chime_dur = rng.uniform(3.0, 5.0)

        # Build chime: short attack + long decay
        n = int(chime_dur * SR)
        t = np.linspace(0, chime_dur, n, endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        attack_n = int(0.01 * SR)
        env = np.zeros(n)
        env[:attack_n] = np.linspace(0, 1, attack_n)
        env[attack_n:] = np.exp(-(t[attack_n:] - t[attack_n]) / 1.2)
        chime = tone * env * rng.uniform(0.5, 1.0)

        start = int(pos * SR)
        end = min(start + len(chime), n_samples)
        chunk = end - start
        output[start:end] += chime[:chunk]
        hit_count += 1

    print(f"    Shimmer: {hit_count} chime hits over {duration}s")

    output = lowpass(output, cutoff_hz=4000, sample_rate=SR)
    stereo = mono_to_stereo(output)
    stereo = reverb(stereo, room_size=0.95, damping=0.3, wet_level=0.8,
                    sample_rate=SR)
    stereo = delay(stereo, delay_seconds=0.5, feedback=0.35, mix=0.3,
                   sample_rate=SR)

    # Very quiet
    gain = 10 ** (-25.0 / 20)
    stereo = stereo * gain
    return stereo


def build_binaural(duration: int) -> np.ndarray:
    """Binaural 4 Hz theta beat."""
    return q.binaural_custom(carrier_hz=196.0, beat_hz=4.0,
                             duration_sec=duration, volume_db=-14.0,
                             sample_rate=SR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== V004 Full Cycle Mix — {DUR}s (4 min) ===")
    print(f"  Arpeggio: G2->Bb2->D3->G3->D3->Bb2 (palindrome)")
    print(f"  Notes: {NOTE_DUR}s each, {XFADE}s crossfades")
    print(f"  Pad: G3 (196 Hz), breathing 30s")
    print(f"  Shimmer: sparse (45-60s spacing)")
    print(f"  Binaural: 4 Hz theta")
    print(f"  Target: -14 LUFS, 48 kHz WAV")
    print()

    print("  [1/4] Bass arpeggio...")
    bass = build_bass(DUR)
    print("  [2/4] Mid pad...")
    pad = build_pad(DUR)
    print("  [3/4] Shimmer...")
    shimmer = build_shimmer(DUR)
    print("  [4/4] Binaural 4 Hz...")
    binaural = build_binaural(DUR)

    # Match lengths
    min_len = min(bass.shape[0], pad.shape[0], shimmer.shape[0],
                  binaural.shape[0])
    bass = bass[:min_len]
    pad = pad[:min_len]
    shimmer = shimmer[:min_len]
    binaural = binaural[:min_len]

    # Mix: bass lead, pad warmth, shimmer texture, binaural subliminal
    print("  Mixing...")
    stem = mix([bass, pad, shimmer, binaural],
               volumes_db=[0.0, -3.0, -6.0, -8.0])
    stem = normalize_lufs(stem, target_lufs=-14.0, sample_rate=SR)
    stem = master_chain(stem, sample_rate=SR)
    stem = make_loopable(stem, crossfade_sec=5.0, sample_rate=SR)

    # Loop check
    score, report = verify_loop(stem, crossfade_sec=5.0, sample_rate=SR)
    quality = "EXCELLENT" if score > 0.85 else "GOOD" if score > 0.7 else "CHECK"
    print(f"  Loop: {quality} ({score:.3f}) | "
          f"jump={report['jump_amplitude']:.6f} corr={report['correlation']:.4f}")

    path = OUT / "V004_mix_4min.wav"
    export_wav(stem, path, sample_rate=SR)
    peak_db = 20 * np.log10(np.max(np.abs(stem)) + 1e-10)
    print(f"\n  -> {path.name}  ({stem.shape[0] / SR:.0f}s, peak={peak_db:.1f} dBFS)")
    print()
    print("ffmpeg loop 20min:")
    print(f"  ffmpeg -stream_loop 4 -i {path.name} -t 1200 V004_20min.wav")


if __name__ == "__main__":
    main()
