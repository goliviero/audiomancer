"""Fractal constants — single source of truth for shared values."""

# Audio defaults
SAMPLE_RATE = 44100       # Hz — standard CD quality
BIT_DEPTH = 16            # bits per sample for integer export
MAX_AMPLITUDE = 0.95      # headroom ceiling to prevent clipping
DEFAULT_AMPLITUDE = 0.5   # safe default for generators

# Fade defaults
DEFAULT_FADE_SECONDS = 10  # seconds of fade in/out

# Export
OUTPUTS_DIR = "outputs/audio"
SOUNDS_RAW_DIR = "sounds/raw"
SOUNDS_PROCESSED_DIR = "sounds/processed"
