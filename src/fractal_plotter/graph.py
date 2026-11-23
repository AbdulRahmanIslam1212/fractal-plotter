import numpy as np


def make_palette(size: int = 256) -> np.ndarray:
    """Return a (size, 3) uint8 lookup table built from three phase-shifted sines."""
    t = np.linspace(0, 1, size, endpoint=False)
    r = 0.5 + 0.5 * np.sin(2 * np.pi * (t + 0.00))
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (t + 0.33))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * (t + 0.67))
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


PALETTE = 255 - make_palette()


def colorize(counts: np.ndarray, max_iter: int) -> np.ndarray:
    """(height, width) escape counts -> (height, width, 3) uint8 RGB."""
    rgb = PALETTE[counts % len(PALETTE)]  # fancy indexing makes a copy
    rgb[counts == max_iter] = 0  # interior points -> black
    return rgb
