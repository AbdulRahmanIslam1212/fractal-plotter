import numpy as np
from numba import njit, prange


MANDELBROT, JULIA, SHIP = 0, 1, 2


@njit(parallel=True, fastmath=True, cache=True)
def _kernel(height, width, cx, cy, scale, max_iter, mode, jr, ji):
    counts = np.empty((height, width), dtype=np.int32)
    x0 = cx - (width - 1) / 2 * scale
    y0 = cy + (height - 1) / 2 * scale
    for row in prange(height):
        ci = y0 - row * scale
        for col in range(width):
            cr = x0 + col * scale
            counts[row, col] = max_iter

            if mode == 1:  # Julia: pixel is z, c is fixed
                zr, zi = cr, ci
                kr, ki = jr, ji
            else:  # Mandelbrot / Ship: pixel is c
                zr, zi = 0.0, 0.0
                kr, ki = cr, ci

            if mode == 0:  # shortcuts only valid for Mandelbrot
                q = (cr - 0.25) ** 2 + ci * ci
                if q * (q + (cr - 0.25)) <= 0.25 * ci * ci:
                    continue
                if (cr + 1.0) ** 2 + ci * ci <= 0.0625:
                    continue

            for n in range(max_iter):
                if mode == 2:
                    zr = abs(zr)
                    zi = -abs(zi)  # minus = upright ship (see below)
                zr, zi = zr * zr - zi * zi + kr, 2.0 * zr * zi + ki
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


def render(
    height, width, center, scale, max_iter, mode=MANDELBROT, julia_c=(-0.8, 0.156)
):
    return _kernel(
        height,
        width,
        center[0],
        center[1],
        scale,
        max_iter,
        mode,
        julia_c[0],
        julia_c[1],
    )
