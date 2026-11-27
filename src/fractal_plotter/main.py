import pygame
from compute import render, pixel_to_complex, MANDELBROT, JULIA, SHIP
from graph import colorize
import math
import numpy as np
import json
import time
from pathlib import Path

EXPORT_DIR = Path("exports")

C_SPEED = 0.4  # units of c per second
C_FINE = 0.1  # multiplier while holding Ctrl, for precise tuning

PAN_SPEED = 400  # screen pixels per second
BOOST = 3.0  # multiplier while holding shift

WIDTH, HEIGHT = 1280, 720
BASE_ITER = 1024
DEFAULT_SCALE = 3.0 / WIDTH

VIEWS = {
    MANDELBROT: ((-0.5, 0.0), 3.0 / WIDTH),
    JULIA: ((0.0, 0.0), 3.0 / WIDTH),
    SHIP: ((-0.5, 0.5), 3.5 / WIDTH),
}
NAMES = {"mandelbrot": MANDELBROT, "julia": JULIA, "ship": SHIP}
KEY_TO_MODE = {pygame.K_1: MANDELBROT, pygame.K_2: JULIA, pygame.K_3: SHIP}

MODE_NAMES = {v: k for k, v in NAMES.items()}


def export_view(surface, mode, center, scale, max_iter, julia_c, with_image=True):
    EXPORT_DIR.mkdir(exist_ok=True)
    stem = EXPORT_DIR / f"{MODE_NAMES[mode]}_{time.strftime('%Y%m%d_%H%M%S')}"

    meta = {
        "version": 1,
        "fractal": MODE_NAMES[mode],
        "julia_c": list(julia_c) if mode == JULIA else None,
        "center": list(center),
        "scale": scale,  # source of truth
        "vertical_span": scale * HEIGHT,  # informational
        "zoom": DEFAULT_SCALE / scale,  # informational
        "max_iter": max_iter,
        "width": WIDTH,
        "height": HEIGHT,
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    if with_image:
        pygame.image.save(surface, str(stem.with_suffix(".png")))
    return stem


def main() -> None:
    input_mode = (
        input("Fractal (mandelbrot/julia/ship) [mandelbrot]: ").strip().lower()
        or "mandelbrot"
    )
    if input_mode not in NAMES:
        raise SystemExit(
            f"Unknown fractal '{input_mode}'. Choose from: {', '.join(NAMES)}"
        )
    mode = NAMES[input_mode]
    julia_c = (-0.12256, 0.74486)  # Default: Douady rabbit
    input_span = input("Enter a span (leave empty if you dont want to): ")
    input_real = input("Enter a real component (leave empty if you dont want to): ")
    input_imag = input(
        "Enter an imaginary component (leave empty if you dont want to): "
    )

    if not input_span or not input_real or not input_imag:
        center, scale = VIEWS[mode]
    else:
        try:
            center = (float(input_real), -float(input_imag))
            scale = float(input_span) / HEIGHT

        except:
            raise Exception("Something went wrong parsing your inputs")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    dirty = True
    surface = None
    max_iter = BASE_ITER

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                p = pixel_to_complex(mx, my, WIDTH, HEIGHT, center, scale)

                scale *= 0.8**event.y

                center = (
                    p[0] - (mx - (WIDTH - 1) / 2) * scale,
                    p[1] - ((HEIGHT - 1) / 2 - my) * scale,
                )
                max_iter = int(BASE_ITER + 50 * math.log2(DEFAULT_SCALE / scale))
                dirty = True

            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_MODE:
                    mode = KEY_TO_MODE[event.key]
                    center, scale = VIEWS[mode]
                    max_iter = BASE_ITER
                    dirty = True
                elif event.key == pygame.K_r:
                    center, scale = VIEWS[mode]
                    max_iter = BASE_ITER
                    dirty = True
                elif event.key == pygame.K_e and surface is not None:
                    image = not (event.mod & pygame.KMOD_SHIFT)
                    stem = export_view(
                        surface, mode, center, scale, max_iter, julia_c, image
                    )
                    pygame.display.set_caption(f"Saved {stem.name}!")
        if dirty:
            counts = render(HEIGHT, WIDTH, center, scale, max_iter, mode, julia_c)
            rgb = colorize(counts, max_iter)
            surface = pygame.image.frombuffer(
                np.ascontiguousarray(rgb).tobytes(), (WIDTH, HEIGHT), "RGB"
            )
            dirty = False

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        dt = min(clock.tick(60) / 1000, 0.1)

        keys = pygame.key.get_pressed()
        dx = keys[pygame.K_d] - keys[pygame.K_a]
        dy = keys[pygame.K_w] - keys[pygame.K_s]

        if dx or dy:
            speed = PAN_SPEED * (BOOST if keys[pygame.K_LSHIFT] else 1.0)
            norm = math.hypot(dx, dy)
            step = speed * dt * scale / norm
            center = (center[0] + dx * step, center[1] + dy * step)
            dirty = True

        if mode == JULIA:
            ax = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
            ay = keys[pygame.K_UP] - keys[pygame.K_DOWN]
            if ax or ay:
                speed = C_SPEED * (C_FINE if keys[pygame.K_LCTRL] else 1.0)
                norm = math.hypot(ax, ay)
                step = speed * dt / norm
                cr = max(-2.0, min(2.0, julia_c[0] + ax * step))
                ci = max(-2.0, min(2.0, julia_c[1] + ay * step))
                julia_c = (cr, ci)
                pygame.display.set_caption(f"Julia  c = {cr:.4f} {ci:+.4f}i")
                dirty = True

    pygame.quit()


if __name__ == "__main__":
    main()
