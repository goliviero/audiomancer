"""Gallery v4 — ethnic instruments (synthetic + sampler).

10 clips exercising the 5 synth instruments (didgeridoo, handpan, oud, sitar,
derbouka) + the sampler workflow (pitch-shift + paulstretched pad) +
combined ambient scene.

Usage:
    python scripts/52_gallery_v4.py
    python scripts/52_gallery_v4.py --only 01_didgeridoo_drone

Note: clip 06 (sampler pad) uses a SYNTHETIC sample (sine at A3) because
no CC0 handpan/oud WAV ships with the repo. Swap in your own sample
(samples/own/handpan_D3.wav etc.) to hear the real thing.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from audiomancer.effects import reverb_hall
from audiomancer.instruments import (
    derbouka_pattern,
    didgeridoo,
    handpan,
    oud,
    sitar,
)
from audiomancer.ir_reverb import reverb_from_synthetic
from audiomancer.layers import mix, normalize_lufs
from audiomancer.mastering import master_chain
from audiomancer.sampler import pitched_pad, play_note
from audiomancer.synth import karplus_strong, sine
from audiomancer.utils import export_wav, fade_in, fade_out, mono_to_stereo

SR = 48000
OUT = project_root / "output" / "gallery_v4"


def _finalize(sig: np.ndarray, target_lufs: float = -14.0,
              fade_sec: float = 0.5) -> np.ndarray:
    if sig.ndim == 1:
        sig = mono_to_stereo(sig)
    sig = fade_in(sig, fade_sec, sample_rate=SR)
    sig = fade_out(sig, fade_sec, sample_rate=SR)
    sig = normalize_lufs(sig, target_lufs=target_lufs, sample_rate=SR)
    sig = master_chain(sig, sample_rate=SR)
    return sig


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def clip_01_didgeridoo_drone() -> np.ndarray:
    """15s didgeridoo drone at 73 Hz with breath rhythm + IR cathedral reverb."""
    didj = didgeridoo(73.0, 15.0, breath_rate=0.35, formant_shift=0.0,
                      seed=42, sample_rate=SR)
    stereo = mono_to_stereo(didj)
    stereo = reverb_from_synthetic(stereo, space="cathedral", wet=0.25,
                                   seed=42, sample_rate=SR)
    return _finalize(stereo)


def clip_02_handpan_progression() -> np.ndarray:
    """6 handpan hits: D3, A3, F3, C4, E3, D3 (D minor pentatonic-ish)."""
    sequence = [
        (146.83, 0.0),   # D3
        (220.00, 1.2),   # A3
        (174.61, 2.4),   # F3
        (261.63, 3.6),   # C4
        (164.81, 4.8),   # E3
        (146.83, 6.0),   # D3
    ]
    total = 9.0
    out = np.zeros(int(total * SR))
    for freq, start_s in sequence:
        hit = handpan(freq, 3.5, inharmonicity=0.06, decay=0.998,
                      seed=int(freq * 10), sample_rate=SR)
        start = int(start_s * SR)
        end = min(start + len(hit), len(out))
        out[start:end] += hit[:end - start]
    stereo = mono_to_stereo(out)
    stereo = reverb_from_synthetic(stereo, space="hall", wet=0.5,
                                   seed=42, sample_rate=SR)
    return _finalize(stereo)


def clip_03_oud_phrase() -> np.ndarray:
    """Oud arpeggio: 5 notes with pluck envelope + body resonance."""
    # D minor: D A F D A
    notes = [146.83, 220.0, 174.61, 146.83, 220.0]
    total = 7.0
    out = np.zeros(int(total * SR))
    for i, freq in enumerate(notes):
        note = oud(freq, 2.5, body_resonance_hz=400,
                   seed=42 + i * 17, sample_rate=SR)
        start = int(i * 1.2 * SR)
        end = min(start + len(note), len(out))
        out[start:end] += note[:end - start] * 0.6
    stereo = mono_to_stereo(out)
    stereo = reverb_from_synthetic(stereo, space="room", wet=0.35,
                                   seed=42, sample_rate=SR)
    return _finalize(stereo)


def clip_04_sitar_cmaj9() -> np.ndarray:
    """Sitar: Cmaj9 arpège up-down with buzz + sympathetic strings."""
    from audiomancer.harmony import arpeggio_from_chord
    freqs = arpeggio_from_chord("Cmaj9", octaves=1, pattern="up_down",
                                root_octave=3)
    total = 8.0
    out = np.zeros(int(total * SR))
    for i, freq in enumerate(freqs):
        note = sitar(freq, 3.0, buzz_amount=0.35,
                     sympathetic_strings=True,
                     seed=42 + i * 11, sample_rate=SR)
        start = int(i * 0.4 * SR)
        end = min(start + len(note), len(out))
        out[start:end] += note[:end - start] * 0.4
    stereo = mono_to_stereo(out)
    stereo = reverb_from_synthetic(stereo, space="hall", wet=0.45,
                                   seed=42, sample_rate=SR)
    return _finalize(stereo)


def clip_05_derbouka_pattern() -> np.ndarray:
    """Derbouka rhythm: 'D t t D t t D t' @ 120 BPM for 8 seconds."""
    pattern = "D t t D t t D t"
    sig = derbouka_pattern(pattern, bpm=120, duration_sec=8.0,
                           seed=42, sample_rate=SR)
    stereo = mono_to_stereo(sig)
    stereo = reverb_hall(stereo, sample_rate=SR)
    return _finalize(stereo)


def clip_06_sampler_pitch_free() -> np.ndarray:
    """play_note A/B: same source sample (synthetic handpan A3=220 Hz) played at 4 different pitches.

    Uses clip_02's handpan at A3 as source, then pitch-shifts to D3, F3, A3, C4.
    Demonstrates how a single-sample instrument can play any note.
    """
    # Create a synthetic "handpan A3" sample (our stand-in)
    source = handpan(220.0, 3.5, inharmonicity=0.07, decay=0.998,
                     seed=42, sample_rate=SR)

    # Play at 4 different target pitches, 2.5s apart
    targets = [146.83, 174.61, 220.0, 261.63]  # D3, F3, A3, C4
    total = 10.0
    out = np.zeros(int(total * SR))
    for i, target in enumerate(targets):
        note = play_note(source, source_hz=220.0, target_hz=target,
                         amplitude=0.6, sample_rate=SR)
        start = int(i * 2.5 * SR)
        end = min(start + len(note), len(out))
        out[start:end] += note[:end - start]
    stereo = mono_to_stereo(out)
    stereo = reverb_from_synthetic(stereo, space="hall", wet=0.4,
                                   seed=42, sample_rate=SR)
    return _finalize(stereo)


def clip_07_sampler_pad() -> np.ndarray:
    """pitched_pad: turn a 3s handpan into a 15s sustained pad at target freq.

    Source: synthetic handpan at A3. Target: F3 (174.61 Hz). Duration: 15s.
    """
    source = handpan(220.0, 3.0, inharmonicity=0.08, decay=0.999,
                     seed=42, sample_rate=SR)
    pad = pitched_pad(source, source_hz=220.0, target_hz=174.61,
                      duration_sec=15.0, window_sec=0.4,
                      amplitude=0.65, seed=42, sample_rate=SR)
    pad = reverb_from_synthetic(pad, space="cathedral", wet=0.5,
                                seed=42, sample_rate=SR)
    return _finalize(pad, fade_sec=2.0)


def clip_08_karplus_ethnic_blend() -> np.ndarray:
    """Karplus + oud + sitar blend: Cmaj triad played with 3 techniques."""
    freqs = [130.81, 164.81, 196.0]  # Cmaj
    section = 5.0
    kp_parts = []
    oud_parts = []
    sit_parts = []
    for i, f in enumerate(freqs):
        kp_parts.append(karplus_strong(f, section, decay=0.998,
                                       brightness=0.6,
                                       seed=42 + i, sample_rate=SR))
        oud_parts.append(oud(f, section, body_resonance_hz=400,
                             seed=42 + i * 17, sample_rate=SR))
        sit_parts.append(sitar(f, section, buzz_amount=0.2,
                               sympathetic_strings=True,
                               seed=42 + i * 29, sample_rate=SR))

    kp = sum(kp_parts) / len(kp_parts)
    oud_mix = sum(oud_parts) / len(oud_parts)
    sit_mix = sum(sit_parts) / len(sit_parts)

    kp_s = mono_to_stereo(kp)
    oud_s = mono_to_stereo(oud_mix)
    sit_s = mono_to_stereo(sit_mix)

    combined = mix([kp_s, oud_s, sit_s], volumes_db=[-6, -6, -6])
    combined = reverb_from_synthetic(combined, space="hall", wet=0.4,
                                     seed=42, sample_rate=SR)
    return _finalize(combined)


def clip_09_didgeridoo_handpan_derbouka() -> np.ndarray:
    """Full ethnic scene: didgeridoo drone + sparse handpan hits + derbouka pattern."""
    duration = 15.0

    # Didgeridoo drone (mono)
    didj = didgeridoo(73.0, duration, breath_rate=0.35,
                      formant_shift=0.0, seed=42, sample_rate=SR)
    didj_s = mono_to_stereo(didj)
    didj_s = reverb_from_synthetic(didj_s, space="cathedral", wet=0.3,
                                   seed=42, sample_rate=SR)

    # Handpan hits: 4 notes spread across duration
    hp = np.zeros(int(duration * SR))
    hits = [(2.0, 146.83), (5.5, 220.0), (9.0, 174.61), (12.5, 261.63)]
    for pos, f in hits:
        note = handpan(f, 3.5, inharmonicity=0.06, decay=0.998,
                       seed=int(f * 10), sample_rate=SR)
        start = int(pos * SR)
        end = min(start + len(note), len(hp))
        hp[start:end] += note[:end - start]
    hp_s = mono_to_stereo(hp)
    hp_s = reverb_from_synthetic(hp_s, space="cathedral", wet=0.5,
                                 seed=42, sample_rate=SR)

    # Derbouka: soft pattern underneath
    drum = derbouka_pattern("D . t . D . t .", bpm=80,
                            duration_sec=duration, seed=42, sample_rate=SR)
    drum_s = mono_to_stereo(drum)
    drum_s = reverb_hall(drum_s, sample_rate=SR)

    min_len = min(didj_s.shape[0], hp_s.shape[0], drum_s.shape[0])
    combined = mix(
        [didj_s[:min_len], hp_s[:min_len], drum_s[:min_len]],
        volumes_db=[-6.0, -10.0, -16.0],
    )
    return _finalize(combined, fade_sec=1.5)


def clip_10_handpan_pad_cathedral() -> np.ndarray:
    """Pitched_pad de handpan A3 -> D3, 20s, cathedral IR + subtle flutter.

    The sampler showpiece: turn a 3s single note into a 20s ambient pad
    at any target pitch, with authentic convolution reverb.
    """
    from audiomancer.saturation import vinyl_wow
    source = handpan(220.0, 3.0, inharmonicity=0.07, decay=0.999,
                     seed=42, sample_rate=SR)
    pad = pitched_pad(source, source_hz=220.0, target_hz=146.83,
                      duration_sec=20.0, window_sec=0.5,
                      amplitude=0.7, seed=42, sample_rate=SR)
    # Vinyl flutter for organic feel
    pad = vinyl_wow(pad, depth=0.0003, rate_hz=0.2, seed=7, sample_rate=SR)
    pad = reverb_from_synthetic(pad, space="cathedral", wet=0.55,
                                seed=42, sample_rate=SR)
    return _finalize(pad, fade_sec=2.5)


CLIPS = {
    "01_didgeridoo_drone": (clip_01_didgeridoo_drone,
                            "Didgeridoo at 73 Hz + breath rhythm + cathedral IR"),
    "02_handpan_progression": (clip_02_handpan_progression,
                               "6-note handpan progression (D minor pentatonic)"),
    "03_oud_phrase": (clip_03_oud_phrase,
                      "Oud arpeggio: D A F D A (Karplus + body resonance)"),
    "04_sitar_cmaj9": (clip_04_sitar_cmaj9,
                      "Sitar Cmaj9 up-down arpeggio with jawari buzz"),
    "05_derbouka_pattern": (clip_05_derbouka_pattern,
                            "Derbouka pattern 'D t t D t t D t' @ 120 BPM"),
    "06_sampler_pitch_free": (clip_06_sampler_pitch_free,
                              "Same source sample played at 4 pitches (D3/F3/A3/C4)"),
    "07_sampler_pad": (clip_07_sampler_pad,
                      "3s source -> 15s pitched pad via pitched_pad"),
    "08_karplus_ethnic_blend": (clip_08_karplus_ethnic_blend,
                                "Cmaj triad: Karplus + oud + sitar stacked"),
    "09_didgeridoo_handpan_derbouka": (clip_09_didgeridoo_handpan_derbouka,
                                       "Full ethnic scene: didj drone + handpan + derbouka"),
    "10_handpan_pad_cathedral": (clip_10_handpan_pad_cathedral,
                                 "Handpan single note -> 20s pitched pad + cathedral"),
}


def main():
    parser = argparse.ArgumentParser(description="Gallery v4 — ethnic instruments.")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    selected = {args.only: CLIPS[args.only]} if args.only else CLIPS

    print(f"=== Gallery v4 - {len(selected)} clip(s) ===")
    print(f"  Target: -14 LUFS, 48 kHz WAV, {OUT}/")
    print()

    for label, (fn, desc) in selected.items():
        print(f"  [{label}] {desc}")
        sig = fn()
        path = OUT / f"{label}.wav"
        export_wav(sig, path, sample_rate=SR)
        peak_db = 20 * np.log10(np.max(np.abs(sig)) + 1e-10)
        print(f"    -> {path.name}  ({sig.shape[0] / SR:.1f}s, peak={peak_db:.1f} dBFS)")

    print()
    print(f"Done - {len(selected)} clip(s) in {OUT}/")


if __name__ == "__main__":
    main()
