"""Smoke test — verify project imports and basic setup."""


def test_import():
    """Verify the fractal package is importable."""
    import fractal  # noqa: F401


def test_sample_rate_constant():
    """Verify default sample rate is defined."""
    from fractal import SAMPLE_RATE

    assert isinstance(SAMPLE_RATE, int)
    assert SAMPLE_RATE == 44100


def test_all_modules_importable():
    """Verify all Phase 1 modules import without errors."""
    from fractal import constants  # noqa: F401
    from fractal import signal  # noqa: F401
    from fractal import generators  # noqa: F401
    from fractal import export  # noqa: F401
