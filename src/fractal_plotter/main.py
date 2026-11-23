import pygame
from compute import mandelbrot, pixel_to_complex
from graph import colorize
import math
import numpy as np

PAN_SPEED = 400  # screen pixels per second
BOOST = 3.0  # multiplier while holding shift

WIDTH, HEIGHT = 1920, 1080
BASE_ITER = 1024
DEFAULT_SCALE = 3.0 / WIDTH


def main() -> None:
    input_span = input("Enter a span (leave empty if you dont want to): ")
    input_real = input("Enter a real component (leave empty if you dont want to): ")
    input_imag = input(
        "Enter an imaginary component (leave empty if you dont want to): "
    )

    if not input_span or not input_real or not input_imag:
        center = (-0.5, 0.0)
        scale = DEFAULT_SCALE

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

        if dirty:
            counts = mandelbrot(HEIGHT, WIDTH, center, scale, max_iter)
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

    pygame.quit()


if __name__ == "__main__":
    main()
