import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _kernel(height, width, cx, cy, scale, max_iter):
    counts = np.empty((height, width), dtype=np.int32)
    x0 = cx - (width - 1) / 2 * scale
    y0 = cy + (height - 1) / 2 * scale
    for row in prange(height):  # rows run in parallel
        ci = y0 - row * scale
        for col in range(width):
            cr = x0 + col * scale
            counts[row, col] = max_iter  # interior default

            # skip the main cardioid and period-2 bulb: known to never escape
            q = (cr - 0.25) ** 2 + ci * ci
            if q * (q + (cr - 0.25)) <= 0.25 * ci * ci:
                continue
            if (cr + 1.0) ** 2 + ci * ci <= 0.0625:
                continue

            zr = 0.0
            zi = 0.0
            for n in range(max_iter):
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                if zr * zr + zi * zi > 4.0:
                    counts[row, col] = n
                    break
    return counts


def pixel_to_complex(
    px: float,
    py: float,
    width: int,
    height: int,
    center: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    """Map a pixel position to (real, imag). Row 0 is the top of the view."""
    re = center[0] + (px - (width - 1) / 2) * scale
    im = center[1] + ((height - 1) / 2 - py) * scale
    return re, im


def mandelbrot(height, width, center=(-0.5, 0.0), scale=None, max_iter=200):
    if scale is None:
        scale = 3.0 / width
    return _kernel(height, width, center[0], center[1], scale, max_iter)
