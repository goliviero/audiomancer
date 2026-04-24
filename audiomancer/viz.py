"""Visualization helpers — waveform and spectrum PNGs for QA/docs.

Requires matplotlib (optional extra: `pip install audiomancer[viz]`).

Typical usage:

    from audiomancer.viz import plot_stem
    plot_stem(stem, "output/qa/V006_warm_pad.png",
              sample_rate=48000, title="V006 warm pad")
"""

from pathlib import Path

import numpy as np

from audiomancer import SAMPLE_RATE


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for audiomancer.viz. Install:\n"
            "    pip install matplotlib\n"
            "or:\n"
            "    pip install audiomancer[viz]"
        )


def _as_mono(signal: np.ndarray) -> np.ndarray:
    return signal if signal.ndim == 1 else np.mean(signal, axis=1)


def plot_waveform(signal: np.ndarray, output_path: str | Path,
                  sample_rate: int = SAMPLE_RATE,
                  title: str | None = None) -> Path:
    """Render a waveform PNG (mono or stereo overlay).

    Returns the output path on success.
    """
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = signal.shape[0]
    t = np.arange(n) / sample_rate

    fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
    if signal.ndim == 2:
        ax.plot(t, signal[:, 0], alpha=0.6, linewidth=0.5, label="L")
        ax.plot(t, signal[:, 1], alpha=0.6, linewidth=0.5, label="R")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.plot(t, signal, linewidth=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_spectrum(signal: np.ndarray, output_path: str | Path,
                  sample_rate: int = SAMPLE_RATE,
                  title: str | None = None) -> Path:
    """Render an FFT magnitude PNG (log-log).

    Returns the output path on success.
    """
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mono = _as_mono(signal)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sample_rate)
    # dB scale (floor -100)
    spec_db = 20 * np.log10(np.maximum(spec / (np.max(spec) + 1e-12), 1e-5))

    fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
    ax.semilogx(freqs[1:], spec_db[1:], linewidth=0.5)  # skip DC
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(20, sample_rate // 2)
    ax.set_ylim(-80, 5)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_stem(signal: np.ndarray, output_path: str | Path,
              sample_rate: int = SAMPLE_RATE,
              title: str | None = None) -> Path:
    """Combined waveform + spectrum PNG (2 subplots stacked).

    Best single-image QA view.
    """
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = signal.shape[0]
    t = np.arange(n) / sample_rate
    mono = _as_mono(signal)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=100)

    # Waveform
    if signal.ndim == 2:
        ax1.plot(t, signal[:, 0], alpha=0.6, linewidth=0.5, label="L")
        ax1.plot(t, signal[:, 1], alpha=0.6, linewidth=0.5, label="R")
        ax1.legend(loc="upper right", fontsize=8)
    else:
        ax1.plot(t, signal, linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_title((title or "Stem") + "  -  Waveform")

    # Spectrum
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sample_rate)
    spec_db = 20 * np.log10(np.maximum(spec / (np.max(spec) + 1e-12), 1e-5))
    ax2.semilogx(freqs[1:], spec_db[1:], linewidth=0.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_xlim(20, sample_rate // 2)
    ax2.set_ylim(-80, 5)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("Spectrum")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
